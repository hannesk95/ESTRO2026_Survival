"""
Soft tissue sarcoma survival modelling — four training regimes in one run.

  'specific'  train on the primary-histology training fold only
  'agnostic'  pooled training, histology NOT a feature          (regime 2)
  'aware'     pooled training, histology one-hot AS a feature   (regime 3)
  'matched'   pooled training subsampled to the size and event count of the
              specific training fold, repeated N_MATCHED_REPEATS times

All four share the SAME outer folds and the SAME test patients, so every
comparison is paired. Paired bootstrap CIs on the C-index differences are the
inferential quantity.

Regime 3 is the one that maps onto SARCULATOR, which is itself a pooled model
with histological subtype as an input. The hypothesis it tests: pooling failed
for synovial sarcoma because the model was forced to average across subtypes;
given the subtype label it should be able to keep a synovial-specific rule
while still learning general structure from the other 192 patients.

Model types: rsf, extra_trees, gradient_boosting, cox_ph, cox_elastic, svm,
             lipschitz_svm, ensemble

Two bugs from the original script remain fixed — see [FIX 1] and [FIX 2].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from scipy.stats import rankdata

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sksurv.ensemble import (
    RandomSurvivalForest,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

from joblib import Memory
from utils import score_survival_model


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

N_INNER_REPEATS = 3
N_INNER_REPEATS_LARGE_GRID = 1
LARGE_GRID_THRESHOLD = 20

# --- B2 (matched) ------------------------------------------------------------
N_MATCHED_REPEATS = 100
MATCH_EVENT_COUNT = True
MATCHED_PRIMARY_FRACTION = None
MATCHED_RETUNE = False
MIN_EVENTS_IN_TRAIN = 3

# --- regime 3 (aware) --------------------------------------------------------
# Fixed category order so the one-hot columns mean the same thing in every run,
# whichever subtype happens to be primary.
HISTOLOGY_CATEGORIES = ["SYN", "MFH", "Liposarcoma"]

# How much does the aware model actually rely on knowing the subtype? Permute
# the histology columns of the test fold and measure the C-index drop.
N_PERMUTATION_REPEATS = 20

PCA_VARIANCE_GRID = [None, 0.80, 0.90, 0.95, 0.99]

DATA_CSV = "/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/journal_extension_data.csv"
MEMORY = Memory(
    location="/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/sklearn_cache", verbose=0
)

EVENT_FIELD = "event"
TIME_FIELD = "duration"
REGIMES = ("specific", "agnostic", "aware", "matched")

HISTOLOGY_LABELS = {"syn": "SYN", "mfh": "MFH", "lipo": "Liposarcoma"}

COVARIATE_TOKENS = {"volume", "clinical"}
IMAGING_TOKENS = {"radiomics", "ctfm", "omnislicer"}


# =============================================================================
# Data loading
# =============================================================================

_DF_CACHE = {}


def _load_df(path=DATA_CSV):
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(path)
    return _DF_CACHE[path]


def feature_set_blocks(feature_set):
    """Return (has_covariate_block, has_imaging_block)."""
    tokens = set(feature_set.split("_"))
    unknown = tokens - COVARIATE_TOKENS - IMAGING_TOKENS
    if unknown:
        raise ValueError(f"Unsupported feature set token(s): {sorted(unknown)}")
    return bool(tokens & COVARIATE_TOKENS), bool(tokens & IMAGING_TOKENS)


def _feature_columns(df, feature_set):
    """
    Return (covariate_columns, imaging_columns).

    Column ORDER matters downstream: the pipeline and the ensemble both assume
        [covariates ...][imaging ...]
    """
    tokens = feature_set.split("_")
    feature_set_blocks(feature_set)

    covariate_cols = []
    if "volume" in tokens:
        covariate_cols += ["radiomics_original_shape_VoxelVolume"]
    if "clinical" in tokens:
        covariate_cols += ["Age", "Grading", "TNMT", "TNMN"]

    imaging_cols = []
    if "radiomics" in tokens:
        imaging_cols += [
            c for c in df.columns if c.startswith("radiomics_") and "shape" not in c
        ]
    if "ctfm" in tokens:
        imaging_cols += [c for c in df.columns if c.startswith("ctfm_")]
    if "omnislicer" in tokens:
        imaging_cols += [c for c in df.columns if c.startswith("omnislicer_")]

    return covariate_cols, imaging_cols


def _histology_dummies(sub):
    """
    One-hot, NOT ordinal.

    Encoding the subtype as 0/1/2 would tell the model that MFH sits 'between'
    SYN and Liposarcoma and that the gap SYN->MFH equals MFH->Liposarcoma. Both
    are meaningless. A tree would then only be able to split on that invented
    ordering, e.g. '{SYN} vs {MFH, Liposarcoma}', and could never isolate MFH
    alone with a single split.

    Categories are pinned so the columns are identical across runs.
    """
    cat = pd.Categorical(sub["histology"], categories=HISTOLOGY_CATEGORIES)
    if cat.isna().any():
        bad = sorted(set(sub["histology"]) - set(HISTOLOGY_CATEGORIES))
        raise ValueError(f"Unmapped histology label(s): {bad}")
    return pd.get_dummies(cat, prefix="hist").to_numpy(dtype=np.float64)


def get_data(histology, feature_set, include_histology=False):
    """
    Primary-histology patients first, then all other histologies.

    include_histology=True inserts the one-hot subtype columns at the END of the
    covariate block, i.e. inside the PCA passthrough region:

        [volume?, Age, Grading, TNMT, TNMN?, hist_SYN, hist_MFH, hist_Lipo][imaging ...]

    Putting them anywhere after n_covariates would feed them into PCA, which
    would mix a categorical indicator into continuous imaging components and
    destroy the thing regime 3 is trying to test.
    """
    if histology not in HISTOLOGY_LABELS:
        raise ValueError(f"Unsupported histology: {histology}")

    df = _load_df()
    label = HISTOLOGY_LABELS[histology]

    primary_ids = sorted(df.loc[df["histology"] == label, "Pseudonym"].tolist())
    secondary_ids = sorted(df.loc[df["histology"] != label, "Pseudonym"].tolist())
    patient_ids = primary_ids + secondary_ids

    covariate_cols, imaging_cols = _feature_columns(df, feature_set)
    sub = df.set_index("Pseudonym").loc[patient_ids]

    blocks = [sub[covariate_cols].to_numpy(dtype=np.float64)] if covariate_cols else []
    n_covariates = len(covariate_cols)

    if include_histology:
        blocks.append(_histology_dummies(sub))
        n_covariates += len(HISTOLOGY_CATEGORIES)

    if imaging_cols:
        blocks.append(sub[imaging_cols].to_numpy(dtype=np.float64))

    X = np.hstack(blocks) if blocks else np.empty((len(sub), 0))
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    events = sub["event"].to_numpy(dtype=bool)
    times = sub["time"].to_numpy(dtype=np.float64)
    histologies = sub["histology"].to_numpy()

    y = np.array(
        list(zip(events, times)),
        dtype=[(EVENT_FIELD, bool), (TIME_FIELD, np.float64)],
    )

    return (X, y, times, events, histologies,
            len(primary_ids), len(secondary_ids), n_covariates)


# =============================================================================
# Late-fusion ensemble
# =============================================================================

class SurvivalEnsemble(BaseEstimator):
    """
    RandomSurvivalForest on the covariate block + ExtraSurvivalTrees on the
    imaging block, combined as a weighted average of z-standardised risk scores
    using training-set statistics (no leakage at predict time).

        combined = weight * risk_covariate_z + (1 - weight) * risk_imaging_z
    """

    def __init__(self, clinical_idx=None, imaging_idx=None, weight=0.5,
                 imaging_pca_components=None, rsf_kwargs=None, ext_kwargs=None,
                 random_state=RANDOM_STATE):
        self.clinical_idx = clinical_idx
        self.imaging_idx = imaging_idx
        self.weight = weight
        self.imaging_pca_components = imaging_pca_components
        self.rsf_kwargs = rsf_kwargs
        self.ext_kwargs = ext_kwargs
        self.random_state = random_state

    def fit(self, X, y):
        rsf_kwargs = self.rsf_kwargs or {}
        ext_kwargs = self.ext_kwargs or {}

        X = np.asarray(X)
        X_clin, X_img = X[:, self.clinical_idx], X[:, self.imaging_idx]

        if self.imaging_pca_components is not None:
            self.imaging_pca_ = PCA(n_components=self.imaging_pca_components,
                                    svd_solver="full", random_state=self.random_state)
            X_img = self.imaging_pca_.fit_transform(X_img)
        else:
            self.imaging_pca_ = None

        self.rsf_ = RandomSurvivalForest(random_state=self.random_state, **rsf_kwargs)
        self.ext_ = ExtraSurvivalTrees(random_state=self.random_state, **ext_kwargs)
        self.rsf_.fit(X_clin, y)
        self.ext_.fit(X_img, y)

        tr_clin, tr_img = self.rsf_.predict(X_clin), self.ext_.predict(X_img)
        self.clin_mean_, self.clin_std_ = tr_clin.mean(), tr_clin.std() + 1e-12
        self.img_mean_, self.img_std_ = tr_img.mean(), tr_img.std() + 1e-12
        return self

    def predict(self, X):
        X = np.asarray(X)
        X_clin, X_img = X[:, self.clinical_idx], X[:, self.imaging_idx]
        if self.imaging_pca_ is not None:
            X_img = self.imaging_pca_.transform(X_img)
        risk_clin = (self.rsf_.predict(X_clin) - self.clin_mean_) / self.clin_std_
        risk_img = (self.ext_.predict(X_img) - self.img_mean_) / self.img_std_
        return self.weight * risk_clin + (1 - self.weight) * risk_img


# =============================================================================
# Model / pipeline
# =============================================================================

def get_model_and_param_grid(model_type, feature_set, n_covariates):
    """
    [FIX 1] PCA was applied to the WRONG COLUMNS in the original script.

    get_data() puts covariates FIRST. The original ColumnTransformer sent
    slice(0, n_passthrough) to PCA and the remainder to "passthrough", i.e. it
    compressed the clinical covariates and let the full high-dimensional imaging
    block through raw. Every combined feature set was affected.

    n_covariates here already includes the histology one-hot columns when the
    aware regime is being fitted, so they land in the passthrough branch.
    """
    has_covariates, has_imaging = feature_set_blocks(feature_set)
    use_pca = has_imaging and model_type not in ("cox_elastic", "cox_ph", "ensemble")

    if use_pca:
        pca = PCA(random_state=RANDOM_STATE, svd_solver="full")
        if n_covariates == 0:
            preprocessor = pca
            pca_param_name = "preprocessor__n_components"
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("covariates", "passthrough", slice(0, n_covariates)),
                    ("pca", pca, slice(n_covariates, None)),
                ],
                remainder="drop",
            )
            pca_param_name = "preprocessor__pca__n_components"
    else:
        preprocessor = "passthrough"
        pca_param_name = None

    param_grid = {}

    if model_type == "rsf":
        clf = RandomSurvivalForest(random_state=RANDOM_STATE)

        param_grid = {
            "clf__n_estimators": [100, 200, 500],
            "clf__max_depth": [None, 3, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
        }

    elif model_type == "extra_trees":
        clf = ExtraSurvivalTrees(random_state=RANDOM_STATE)
    elif model_type == "gradient_boosting":
        clf = GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE)
    elif model_type == "cox_ph":
        clf = CoxPHSurvivalAnalysis()
    elif model_type == "cox_elastic":
        clf = CoxnetSurvivalAnalysis()

        param_grid = {
            "clf__l1_ratio": [0.0, 0.5, 1.0],
            "clf__alphas": [[0.01, 0.1, 1.0, 10.0]],
        }

    elif model_type == "svm":
        clf = FastSurvivalSVM(random_state=RANDOM_STATE)
    elif model_type == "lipschitz_svm":
        clf = MinlipSurvivalAnalysis()
    elif model_type == "ensemble":
        if not has_covariates or not has_imaging:
            raise ValueError(
                f"feature_set='{feature_set}' lacks a covariate or imaging block; "
                "the ensemble needs both."
            )
        clf = SurvivalEnsemble(
            clinical_idx=slice(0, n_covariates),
            imaging_idx=slice(n_covariates, None),
            random_state=RANDOM_STATE,
        )
        param_grid = {
            "clf__weight": np.round(np.linspace(0.1, 0.9, 9), 2).tolist(),
            "clf__imaging_pca_components": PCA_VARIANCE_GRID,
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("preprocessor", preprocessor), ("clf", clf)],
        memory=MEMORY,
    )
    if use_pca:
        param_grid[pca_param_name] = PCA_VARIANCE_GRID

    return pipeline, param_grid


def _n_grid_candidates(param_grid):
    n = 1
    for values in param_grid.values():
        n *= len(values)
    return n


def fit_model(model_type, feature_set, n_covariates, X_train, y_train,
              fixed_params=None):
    pipeline, param_grid = get_model_and_param_grid(model_type, feature_set, n_covariates)

    if fixed_params is not None:
        if fixed_params:
            pipeline.set_params(**fixed_params)
        pipeline.fit(X_train, y_train)
        return pipeline, dict(fixed_params)

    if not param_grid:
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    n_repeats = (N_INNER_REPEATS_LARGE_GRID
                 if _n_grid_candidates(param_grid) > LARGE_GRID_THRESHOLD
                 else N_INNER_REPEATS)
    cv_inner = RepeatedKFold(n_splits=N_INNER_SPLITS, n_repeats=n_repeats,
                             random_state=RANDOM_STATE)
    grid_search = GridSearchCV(pipeline, param_grid, scoring=score_survival_model,
                               cv=cv_inner, refit=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def log_fitted_model_info(best_model, model_type, regime, fold_idx):
    preprocessor = best_model.named_steps["preprocessor"]
    n_pca = None
    if isinstance(preprocessor, PCA):
        n_pca = preprocessor.n_components_
    elif isinstance(preprocessor, ColumnTransformer):
        n_pca = preprocessor.named_transformers_["pca"].n_components_
    if n_pca is not None:
        mlflow.log_metric(f"{regime}_n_pca_components_fold_{fold_idx}", int(n_pca))

    if model_type == "ensemble":
        clf = best_model.named_steps["clf"]
        mlflow.log_metric(f"{regime}_ensemble_weight_fold_{fold_idx}", float(clf.weight))
        mlflow.log_param(f"{regime}_ensemble_imaging_pca_fold_{fold_idx}",
                         clf.imaging_pca_components)


# =============================================================================
# Regime 3 diagnostic: does the model actually use the subtype label?
# =============================================================================

def histology_permutation_importance(model, X_test, y_test, hist_slice,
                                     n_repeats=N_PERMUTATION_REPEATS,
                                     random_state=RANDOM_STATE):
    """
    C-index drop when the histology one-hot block is shuffled across test rows.

    Rows are permuted JOINTLY so every shuffled row is still a valid one-hot
    vector — shuffling each column independently would produce impossible
    patients (two subtypes at once, or none) and overstate the importance.

    ~0  -> the label was ignored; regime 3 collapses to regime 2
    >0  -> the model genuinely conditions on subtype
    """
    base = concordance_index_censored(
        y_test[EVENT_FIELD], y_test[TIME_FIELD], model.predict(X_test))[0]

    rng = np.random.RandomState(random_state)
    drops = []
    for _ in range(n_repeats):
        X_perm = X_test.copy()
        X_perm[:, hist_slice] = X_test[rng.permutation(len(X_test)), hist_slice]
        c = concordance_index_censored(
            y_test[EVENT_FIELD], y_test[TIME_FIELD], model.predict(X_perm))[0]
        drops.append(base - c)
    return base, float(np.mean(drops)), float(np.std(drops))


# =============================================================================
# B2: sample-size-matched subsampling
# =============================================================================

def _draw_without_replacement(rng, candidates, n):
    n = int(min(n, len(candidates)))
    if n <= 0:
        return np.array([], dtype=int)
    return rng.choice(candidates, size=n, replace=False)


def _stratified_draw(rng, candidates, pool_events, n_events, n_censored):
    ev = candidates[pool_events[candidates]]
    cn = candidates[~pool_events[candidates]]
    return np.concatenate([_draw_without_replacement(rng, ev, n_events),
                           _draw_without_replacement(rng, cn, n_censored)])


def draw_matched_indices(rng, pool_is_primary, pool_events, n_target, n_events_target,
                         primary_fraction=None, match_event_count=True):
    all_idx = np.arange(len(pool_events))
    n_censored_target = n_target - n_events_target

    if primary_fraction is None:
        if match_event_count:
            return _stratified_draw(rng, all_idx, pool_events,
                                    n_events_target, n_censored_target)
        return _draw_without_replacement(rng, all_idx, n_target)

    n_primary_budget = int(round(primary_fraction * n_target))
    primary_idx = all_idx[pool_is_primary]
    secondary_idx = all_idx[~pool_is_primary]
    n_primary_budget = min(n_primary_budget, len(primary_idx))
    n_secondary_budget = min(n_target - n_primary_budget, len(secondary_idx))

    if not match_event_count:
        return np.concatenate([
            _draw_without_replacement(rng, primary_idx, n_primary_budget),
            _draw_without_replacement(rng, secondary_idx, n_secondary_budget)])

    event_rate = n_events_target / max(n_target, 1)
    n_ev_primary = int(round(event_rate * n_primary_budget))
    n_ev_secondary = int(round(event_rate * n_secondary_budget))
    return np.concatenate([
        _stratified_draw(rng, primary_idx, pool_events,
                         n_ev_primary, n_primary_budget - n_ev_primary),
        _stratified_draw(rng, secondary_idx, pool_events,
                         n_ev_secondary, n_secondary_budget - n_ev_secondary)])


# =============================================================================
# Evaluation helpers
# =============================================================================

def bootstrap_c_index(y_primary, risk_scores, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    n = len(y_primary)
    scores = []
    for _ in range(n_bootstrap):
        sample = rng.choice(np.arange(n), size=n, replace=True)
        if y_primary[EVENT_FIELD][sample].sum() < 2:
            continue
        scores.append(concordance_index_censored(
            y_primary[EVENT_FIELD][sample], y_primary[TIME_FIELD][sample],
            risk_scores[sample])[0])
    return np.array(scores)


def paired_bootstrap_delta(y_primary, risk_a, risk_b,
                           n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Paired bootstrap of C-index(a) - C-index(b) on the SAME resampled patients."""
    rng = np.random.RandomState(seed)
    n = len(y_primary)
    deltas = []
    for _ in range(n_bootstrap):
        sample = rng.choice(np.arange(n), size=n, replace=True)
        if y_primary[EVENT_FIELD][sample].sum() < 2:
            continue
        ev, tm = y_primary[EVENT_FIELD][sample], y_primary[TIME_FIELD][sample]
        deltas.append(concordance_index_censored(ev, tm, risk_a[sample])[0]
                      - concordance_index_censored(ev, tm, risk_b[sample])[0])
    return np.array(deltas)


def kaplan_meier_by_risk(y_primary, risk_scores, tag, title):
    median_risk = np.median(risk_scores)
    risk_group = np.where(risk_scores >= median_risk, "high", "low")
    chisq, pvalue = compare_survival(y_primary, risk_group)

    plt.figure(figsize=(8, 6))
    for group_label in np.unique(risk_group):
        mask = risk_group == group_label
        time_g, surv_g, conf_int = kaplan_meier_estimator(
            y_primary[EVENT_FIELD][mask], y_primary[TIME_FIELD][mask], conf_type="log-log")
        plt.step(time_g, surv_g, where="post", label=f"{group_label} risk (n={mask.sum()})")
        plt.fill_between(time_g, conf_int[0], conf_int[1], alpha=0.2, step="post")

    plt.ylim(0, 1)
    plt.xlabel("Time"); plt.ylabel("Survival probability")
    plt.title(f"{title}\nLog-rank p = {pvalue:.4g}")
    plt.legend(loc="best"); plt.grid(alpha=0.3); plt.tight_layout()
    filename = f"kaplan_meier_{tag}.png"
    plt.savefig(filename, dpi=300); plt.close()
    mlflow.log_artifact(filename)
    return chisq, pvalue


# =============================================================================
# Main
# =============================================================================

def main(model_type, histology, feature_set):

    for k, v in [("model_type", model_type), ("histology", histology),
                 ("feature_set", feature_set), ("n_outer_splits", N_OUTER_SPLITS),
                 ("n_inner_splits", N_INNER_SPLITS), ("n_bootstrap", N_BOOTSTRAP),
                 ("random_state", RANDOM_STATE),
                 ("n_matched_repeats", N_MATCHED_REPEATS),
                 ("match_event_count", MATCH_EVENT_COUNT),
                 ("matched_primary_fraction", MATCHED_PRIMARY_FRACTION),
                 ("matched_retune", MATCHED_RETUNE)]:
        mlflow.log_param(k, v)

    # -------------------------------------------------------------------------
    # 1. Load data — blind matrix, and the same data with histology one-hots
    # -------------------------------------------------------------------------
    (X, y, times, events, histologies,
     n_primary, n_secondary, n_covariates) = get_data(histology, feature_set)

    (X_aw, _, _, _, _, _, _, n_cov_aw) = get_data(histology, feature_set,
                                                  include_histology=True)

    # where the one-hot block sits inside X_aw
    n_hist = len(HISTOLOGY_CATEGORIES)
    hist_slice = slice(n_cov_aw - n_hist, n_cov_aw)

    X_primary, X_secondary = X[:n_primary], X[n_primary:]
    Xaw_primary, Xaw_secondary = X_aw[:n_primary], X_aw[n_primary:]
    y_primary, y_secondary = y[:n_primary], y[n_primary:]
    events_primary, events_secondary = events[:n_primary], events[n_primary:]

    mlflow.log_param("n_primary", n_primary)
    mlflow.log_param("n_secondary", n_secondary)
    mlflow.log_metric("n_events_primary", int(events_primary.sum()))
    mlflow.log_metric("n_events_secondary", int(events_secondary.sum()))
    mlflow.log_metric("n_features", X.shape[1])
    mlflow.log_metric("n_covariates", n_covariates)

    if n_secondary == 0:
        raise ValueError("No secondary histologies — agnostic/aware/matched impossible.")

    # -------------------------------------------------------------------------
    # 2. Outer CV — identical folds shared by all four regimes
    # -------------------------------------------------------------------------
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True,
                               random_state=RANDOM_STATE)
    folds = list(outer_cv.split(X_primary, events_primary))

    # [FIX 2] risk score vectors are sized to the PRIMARY cohort only. The
    # original allocated np.zeros(primary + secondary) but only ever wrote into
    # primary indices, leaving every secondary patient at exactly 0.0 — and then
    # computed the bootstrap CI, median split, log-rank test and KM plot over the
    # FULL y. For every pooled run those numbers came from a cohort in which the
    # majority of patients had a constant risk score.
    risk = {r: np.zeros(n_primary) for r in ("specific", "agnostic", "aware")}
    fold_c = {r: [] for r in ("specific", "agnostic", "aware")}
    risk_matched = np.zeros((N_MATCHED_REPEATS, n_primary))
    perm_drops = []

    # -------------------------------------------------------------------------
    # 3. Fit all four regimes per fold
    # -------------------------------------------------------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):

        X_tr, y_tr = X_primary[train_idx], y_primary[train_idx]
        X_test = X_primary[test_idx]

        # --- regime 1: specific ---------------------------------------------
        # Fitted WITHOUT the histology columns on purpose: every patient here is
        # the same subtype, so the one-hots would be constant and carry nothing.
        model_spec, params_spec = fit_model(
            model_type, feature_set, n_covariates, X_tr, y_tr)
        risk["specific"][test_idx] = model_spec.predict(X_test)
        log_fitted_model_info(model_spec, model_type, "specific", fold_idx)

        # --- regime 2: agnostic-blind ---------------------------------------
        X_tr_pooled = np.vstack([X_tr, X_secondary])
        y_tr_pooled = np.hstack([y_tr, y_secondary])
        model_agn, params_agn = fit_model(
            model_type, feature_set, n_covariates, X_tr_pooled, y_tr_pooled)
        risk["agnostic"][test_idx] = model_agn.predict(X_test)
        log_fitted_model_info(model_agn, model_type, "agnostic", fold_idx)

        # --- regime 3: agnostic-aware ---------------------------------------
        Xaw_tr_pooled = np.vstack([Xaw_primary[train_idx], Xaw_secondary])
        Xaw_test = Xaw_primary[test_idx]
        model_aw, params_aw = fit_model(
            model_type, feature_set, n_cov_aw, Xaw_tr_pooled, y_tr_pooled)
        risk["aware"][test_idx] = model_aw.predict(Xaw_test)
        log_fitted_model_info(model_aw, model_type, "aware", fold_idx)

        base_c, drop_mean, drop_std = histology_permutation_importance(
            model_aw, Xaw_test, y_primary[test_idx], hist_slice)
        perm_drops.append(drop_mean)
        mlflow.log_metric(f"aware_hist_perm_drop_fold_{fold_idx}", drop_mean)
        mlflow.log_metric(f"aware_hist_perm_drop_std_fold_{fold_idx}", drop_std)

        for regime in ("specific", "agnostic", "aware"):
            c = concordance_index_censored(
                y_primary[EVENT_FIELD][test_idx], y_primary[TIME_FIELD][test_idx],
                risk[regime][test_idx])[0]
            fold_c[regime].append(c)
            mlflow.log_metric(f"{regime}_c_index_fold_{fold_idx}", c)

        mlflow.log_param(f"specific_best_params_fold_{fold_idx}", params_spec)
        mlflow.log_param(f"agnostic_best_params_fold_{fold_idx}", params_agn)
        mlflow.log_param(f"aware_best_params_fold_{fold_idx}", params_aw)
        mlflow.log_metric(f"specific_n_train_fold_{fold_idx}", len(y_tr))
        mlflow.log_metric(f"agnostic_n_train_fold_{fold_idx}", len(y_tr_pooled))

        print(f"Fold {fold_idx}: specific {fold_c['specific'][-1]:.3f} | "
              f"agnostic {fold_c['agnostic'][-1]:.3f} | "
              f"aware {fold_c['aware'][-1]:.3f} "
              f"(histology permutation drop {drop_mean:+.3f})")

        # --- regime 4-equivalent control: matched ---------------------------
        pool_X, pool_y = X_tr_pooled, y_tr_pooled
        pool_events = np.hstack([events_primary[train_idx], events_secondary])
        pool_is_primary = np.hstack([np.ones(len(train_idx), dtype=bool),
                                     np.zeros(n_secondary, dtype=bool)])
        n_target = len(train_idx)
        n_events_target = int(events_primary[train_idx].sum())
        fixed = None if MATCHED_RETUNE else params_agn

        for rep in range(N_MATCHED_REPEATS):
            rng = np.random.RandomState(RANDOM_STATE + 10_000 * fold_idx + rep)
            for _attempt in range(20):
                sel = draw_matched_indices(
                    rng=rng, pool_is_primary=pool_is_primary, pool_events=pool_events,
                    n_target=n_target, n_events_target=n_events_target,
                    primary_fraction=MATCHED_PRIMARY_FRACTION,
                    match_event_count=MATCH_EVENT_COUNT)
                if pool_events[sel].sum() >= MIN_EVENTS_IN_TRAIN:
                    break
            model_matched, _ = fit_model(model_type, feature_set, n_covariates,
                                         pool_X[sel], pool_y[sel], fixed_params=fixed)
            risk_matched[rep, test_idx] = model_matched.predict(X_test)

        mlflow.log_metric(f"matched_n_target_fold_{fold_idx}", n_target)
        mlflow.log_metric(f"matched_n_events_target_fold_{fold_idx}", n_events_target)

    # -------------------------------------------------------------------------
    # 4. Regime 3 diagnostic summary
    # -------------------------------------------------------------------------
    mlflow.log_metric("aware_hist_perm_drop_mean", float(np.mean(perm_drops)))
    print(f"\nHistology permutation importance (aware regime): "
          f"{np.mean(perm_drops):+.3f} mean C-index drop")
    if abs(np.mean(perm_drops)) < 0.005:
        print("  -> the subtype label is being ignored; regime 3 has collapsed "
              "onto regime 2 for this configuration.")

    # -------------------------------------------------------------------------
    # 5. Matched: distribution over subsampling repetitions
    # -------------------------------------------------------------------------
    matched_rep_c = np.array([
        concordance_index_censored(y_primary[EVENT_FIELD], y_primary[TIME_FIELD],
                                   risk_matched[rep])[0]
        for rep in range(N_MATCHED_REPEATS)])

    for name, val in [("mean", matched_rep_c.mean()), ("std", matched_rep_c.std()),
                      ("median", np.median(matched_rep_c)),
                      ("p2_5", np.percentile(matched_rep_c, 2.5)),
                      ("p97_5", np.percentile(matched_rep_c, 97.5))]:
        mlflow.log_metric(f"matched_rep_c_index_{name}", float(val))

    # Different repetitions produce risk scores on different scales, so
    # rank-normalise each before averaging.
    risk["matched"] = np.mean(
        [rankdata(risk_matched[rep]) for rep in range(N_MATCHED_REPEATS)], axis=0)

    plt.figure(figsize=(7, 5))
    plt.hist(matched_rep_c, bins=25, alpha=0.8, label="matched (per repetition)")
    for reg, color in [("specific", "C1"), ("agnostic", "C2"), ("aware", "C3")]:
        plt.axvline(np.mean(fold_c[reg]), color=color, linestyle="--",
                    label=f"{reg} = {np.mean(fold_c[reg]):.3f}")
    plt.xlabel("C-index"); plt.ylabel("Repetitions")
    plt.title(f"{histology} | {feature_set} | {model_type}\nsize-matched control")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("matched_c_index_distribution.png", dpi=300); plt.close()
    mlflow.log_artifact("matched_c_index_distribution.png")

    # -------------------------------------------------------------------------
    # 6. Per-regime metrics
    # -------------------------------------------------------------------------
    boot = {}
    print()
    for regime in REGIMES:
        if regime in fold_c:
            mlflow.log_metric(f"{regime}_c_index_mean", float(np.mean(fold_c[regime])))
            mlflow.log_metric(f"{regime}_c_index_std", float(np.std(fold_c[regime])))

        boot[regime] = bootstrap_c_index(y_primary, risk[regime])
        lo, hi = np.percentile(boot[regime], [2.5, 97.5])
        mlflow.log_metric(f"{regime}_bootstrap_c_index_mean", float(boot[regime].mean()))
        mlflow.log_metric(f"{regime}_bootstrap_c_index_lower", float(lo))
        mlflow.log_metric(f"{regime}_bootstrap_c_index_upper", float(hi))
        print(f"{regime:>9}: bootstrap C = {boot[regime].mean():.3f} "
              f"(95% CI [{lo:.3f}, {hi:.3f}])")

        try:
            chisq, pvalue = kaplan_meier_by_risk(
                y_primary, risk[regime],
                tag=f"{histology}_{feature_set}_{model_type}_{regime}",
                title=f"{histology} | {regime} | {feature_set} | {model_type}")
            mlflow.log_metric(f"{regime}_logrank_chisq", float(chisq))
            mlflow.log_metric(f"{regime}_logrank_pvalue", float(pvalue))
            mlflow.log_param(f"{regime}_significant", bool(pvalue < 0.05 and lo > 0.5))
        except Exception as exc:
            print(f"  KM / log-rank failed for {regime}: {exc}")
            mlflow.log_param(f"{regime}_km_error", str(exc))

    # -------------------------------------------------------------------------
    # 7. Paired comparisons
    # -------------------------------------------------------------------------
    print()
    comparisons = [
        ("aware", "agnostic"),    # does the subtype label help?      <- regime 3
        ("aware", "specific"),    # does aware pooling beat going alone?
        ("agnostic", "specific"),
        ("matched", "specific"),
        ("agnostic", "matched"),
    ]
    for a, b in comparisons:
        deltas = paired_bootstrap_delta(y_primary, risk[a], risk[b])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        name = f"delta_{a}_vs_{b}"
        mlflow.log_metric(f"{name}_mean", float(deltas.mean()))
        mlflow.log_metric(f"{name}_lower", float(lo))
        mlflow.log_metric(f"{name}_upper", float(hi))
        mlflow.log_metric(f"{name}_pvalue", float(p))
        print(f"  {a:9s} - {b:9s}: {deltas.mean():+.3f} "
              f"(95% CI [{lo:+.3f}, {hi:+.3f}], p = {p:.3g})")

    # -------------------------------------------------------------------------
    # 8. Persist out-of-fold predictions
    # -------------------------------------------------------------------------
    np.savez("oof_risk_scores.npz",
             specific=risk["specific"], agnostic=risk["agnostic"],
             aware=risk["aware"], matched=risk["matched"],
             matched_matrix=risk_matched, matched_rep_c=matched_rep_c,
             hist_perm_drops=np.array(perm_drops),
             event=y_primary[EVENT_FIELD], duration=y_primary[TIME_FIELD])
    mlflow.log_artifact("oof_risk_scores.npz")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    FEATURE_SETS = [
        "volume", "clinical", "radiomics", "ctfm", "omnislicer",
        "volume_clinical", "volume_radiomics", "volume_ctfm", "volume_omnislicer",
        "clinical_radiomics", "clinical_ctfm", "clinical_omnislicer",
        "volume_clinical_radiomics", "volume_clinical_ctfm", "volume_clinical_omnislicer",
    ]
    MODEL_TYPES = ["cox_elastic"]
                # "rsf", "extra_trees", "gradient_boosting",
                #  "cox_elastic", "svm", "lipschitz_svm", "ensemble"]
    HISTOLOGIES = ["syn", "mfh", "lipo"]

    mlflow.set_experiment("DEGRO_journal_extension_aware")

    for feature_set in FEATURE_SETS:
        has_covariates, has_imaging = feature_set_blocks(feature_set)
        for model_type in MODEL_TYPES:
            if model_type == "ensemble" and not (has_covariates and has_imaging):
                continue
            for histology in HISTOLOGIES:
                print("\n" + "#" * 79)
                print(f"# {model_type} | {histology} | {feature_set} "
                      f"| specific + agnostic + aware + matched")
                print("#" * 79 + "\n")
                mlflow.start_run()
                try:
                    main(model_type=model_type, histology=histology,
                         feature_set=feature_set)
                except Exception as exc:
                    print(f"RUN FAILED: {exc}")
                    mlflow.log_param("run_error", str(exc))
                finally:
                    mlflow.end_run()