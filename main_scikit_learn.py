"""
Soft tissue sarcoma survival modelling — histology-specific vs. histology-agnostic
vs. sample-size-matched (experiment B2), all three evaluated in a single run.

For every (model_type, histology, feature_set) combination this script fits:

  'specific'  train on the primary-histology training fold only
  'agnostic'  train on the primary-histology training fold + ALL other histologies
  'matched'   train on a random subsample of (training fold + other histologies)
              whose SIZE and EVENT COUNT equal the 'specific' training fold,
              repeated N_MATCHED_REPEATS times

All three share the SAME outer folds and the SAME test patients, so the comparison
is paired. The script reports paired bootstrap confidence intervals on the
C-index differences, which is the number that goes in the paper.

Reading the result:
  matched ~= specific  -> the agnostic gain is explained by training-set size alone
  matched >  specific  -> genuine transferable signal across histologies
  matched <  specific  -> other histologies actively dilute the subtype-specific signal

--------------------------------------------------------------------------------
Two bugs from the original script remain fixed here — see [FIX 1] and [FIX 2].
--------------------------------------------------------------------------------
"""

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata

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

# Inner-CV repeats. The original value of 100 meant 500 fits per candidate; with
# most param grids empty that was 500 fits to choose between one option. Empty
# grids now skip GridSearchCV entirely (see fit_model).
N_INNER_REPEATS = 3

# --- B2 parameters -----------------------------------------------------------
N_MATCHED_REPEATS = 100         # subsampling repetitions per outer fold
MATCH_EVENT_COUNT = True        # match number of EVENTS, not just number of patients.
                                # C-index precision is driven by events, so this is
                                # the tighter control and the one to report.
MATCHED_PRIMARY_FRACTION = None # None -> free draw from the pooled training set
                                # 0.0  -> pure cross-histology transfer (no primary patients)
                                # 0.5  -> half the budget forced to be primary histology
MATCHED_RETUNE = False          # False: reuse the hyperparameters selected by the agnostic
                                #        fit of the same fold (fast; isolates the effect of
                                #        training-set size from tuning noise)
                                # True:  re-run the inner CV for every repetition
                                #        (~20x slower, fully nested)
MIN_EVENTS_IN_TRAIN = 3         # redraw guard

DATA_CSV = "/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/journal_extension_data.csv"
MEMORY = Memory(
    location="/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/sklearn_cache", verbose=0
)

EVENT_FIELD = "event"
TIME_FIELD = "duration"
REGIMES = ("specific", "agnostic", "matched")

HISTOLOGY_LABELS = {"syn": "SYN", "mfh": "MFH", "lipo": "Liposarcoma"}


# =============================================================================
# Data loading
# =============================================================================

def _feature_columns(df, feature_set):
    """
    Return (covariate_columns, imaging_columns) for a feature set.

    Column ORDER matters downstream: the pipeline assumes
        [covariates ...][imaging ...]
    which is the order the original script built its arrays in.
    """
    tokens = feature_set.split("_")
    known = {"volume", "clinical", "radiomics", "ctfm", "omnislicer"}
    unknown = set(tokens) - known
    if unknown:
        raise ValueError(f"Unsupported feature set token(s): {sorted(unknown)}")

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


def get_data(histology, feature_set):
    """
    Primary-histology patients first, then all other histologies.
    Equivalent to the original get_data(), without the per-feature-set duplication.
    """
    if histology not in HISTOLOGY_LABELS:
        raise ValueError(f"Unsupported histology: {histology}")

    df = pd.read_csv(DATA_CSV)
    label = HISTOLOGY_LABELS[histology]

    primary_ids = sorted(df.loc[df["histology"] == label, "Pseudonym"].tolist())
    secondary_ids = sorted(df.loc[df["histology"] != label, "Pseudonym"].tolist())
    patient_ids = primary_ids + secondary_ids

    covariate_cols, imaging_cols = _feature_columns(df, feature_set)
    feature_cols = covariate_cols + imaging_cols

    sub = df.set_index("Pseudonym").loc[patient_ids]

    X = sub[feature_cols].to_numpy(dtype=np.float64)
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
            len(primary_ids), len(secondary_ids), len(covariate_cols))


# =============================================================================
# Model / pipeline
# =============================================================================

def get_model_and_param_grid(model_type, feature_set, n_covariates):
    """
    [FIX 1] PCA was applied to the WRONG COLUMNS in the original script.

    get_data() builds combined feature vectors as
        [volume?, Age, Grading, TNMT, TNMN?] + [imaging features ...]
    i.e. covariates FIRST. The original ColumnTransformer sent
    slice(0, n_passthrough) to PCA and slice(n_passthrough, None) to
    "passthrough" — so PCA compressed the 4-5 clinical covariates while the full
    high-dimensional radiomics / ctfm / omnislicer block went into the model raw
    and unreduced. Every combined feature set is affected.
    """
    has_imaging = feature_set not in ("volume", "clinical", "volume_clinical")
    use_pca = has_imaging and model_type not in ("cox_elastic", "cox_ph")

    if use_pca:
        # svd_solver='full' is required for fractional n_components (0.80, 0.90, ...)
        pca = PCA(random_state=RANDOM_STATE, svd_solver="full")

        if n_covariates == 0:
            preprocessor = pca
            pca_param_name = "preprocessor__n_components"
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    # covariates pass through untouched ...
                    ("covariates", "passthrough", slice(0, n_covariates)),
                    # ... the imaging block is what gets compressed
                    ("pca", pca, slice(n_covariates, None)),
                ],
                remainder="drop",
            )
            pca_param_name = "preprocessor__pca__n_components"
    else:
        preprocessor = "passthrough"
        pca_param_name = None

    if model_type == "rsf":
        clf = RandomSurvivalForest(random_state=RANDOM_STATE)
    elif model_type == "extra_trees":
        clf = ExtraSurvivalTrees(random_state=RANDOM_STATE)
    elif model_type == "gradient_boosting":
        clf = GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE)
    elif model_type == "cox_ph":
        clf = CoxPHSurvivalAnalysis()
    elif model_type == "cox_elastic":
        clf = CoxnetSurvivalAnalysis()
    elif model_type == "svm":
        clf = FastSurvivalSVM(random_state=RANDOM_STATE)
    elif model_type == "lipschitz_svm":
        clf = MinlipSurvivalAnalysis()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("preprocessor", preprocessor), ("clf", clf)],
        memory=MEMORY,
    )

    param_grid = {}
    if use_pca:
        param_grid[pca_param_name] = [0.80, 0.90, 0.95, 0.99]

    return pipeline, param_grid


def fit_model(model_type, feature_set, n_covariates, X_train, y_train,
              fixed_params=None, n_inner_repeats=N_INNER_REPEATS):
    """
    Fit the pipeline. If fixed_params is given, skip tuning and use them directly.
    If the param grid is empty, skip GridSearchCV — the original script ran 500
    inner fits to select between zero hyperparameters.
    """
    pipeline, param_grid = get_model_and_param_grid(model_type, feature_set, n_covariates)

    if fixed_params is not None:
        if fixed_params:
            pipeline.set_params(**fixed_params)
        pipeline.fit(X_train, y_train)
        return pipeline, dict(fixed_params)

    if not param_grid:
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    cv_inner = RepeatedKFold(
        n_splits=N_INNER_SPLITS, n_repeats=n_inner_repeats, random_state=RANDOM_STATE
    )
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring=score_survival_model,
        cv=cv_inner, refit=True, n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# =============================================================================
# B2: sample-size-matched subsampling
# =============================================================================

def _draw_without_replacement(rng, candidates, n):
    n = int(min(n, len(candidates)))
    if n <= 0:
        return np.array([], dtype=int)
    return rng.choice(candidates, size=n, replace=False)


def _stratified_draw(rng, candidates, pool_events, n_events, n_censored):
    """Draw n_events event-patients and n_censored censored patients."""
    ev = candidates[pool_events[candidates]]
    cn = candidates[~pool_events[candidates]]
    return np.concatenate(
        [_draw_without_replacement(rng, ev, n_events),
         _draw_without_replacement(rng, cn, n_censored)]
    )


def draw_matched_indices(rng, pool_is_primary, pool_events, n_target, n_events_target,
                         primary_fraction=None, match_event_count=True):
    """
    Draw indices into the pooled training set (primary training fold + all other
    histologies) so that the resulting training set has the same size — and
    optionally the same number of events — as the histology-specific training fold.
    """
    all_idx = np.arange(len(pool_events))
    n_censored_target = n_target - n_events_target

    if primary_fraction is None:
        if match_event_count:
            return _stratified_draw(rng, all_idx, pool_events,
                                    n_events_target, n_censored_target)
        return _draw_without_replacement(rng, all_idx, n_target)

    # Forced composition: a fixed share of the budget must come from the primary histology.
    n_primary_budget = int(round(primary_fraction * n_target))
    primary_idx = all_idx[pool_is_primary]
    secondary_idx = all_idx[~pool_is_primary]

    n_primary_budget = min(n_primary_budget, len(primary_idx))
    n_secondary_budget = min(n_target - n_primary_budget, len(secondary_idx))

    if not match_event_count:
        return np.concatenate([
            _draw_without_replacement(rng, primary_idx, n_primary_budget),
            _draw_without_replacement(rng, secondary_idx, n_secondary_budget),
        ])

    event_rate = n_events_target / max(n_target, 1)
    n_ev_primary = int(round(event_rate * n_primary_budget))
    n_ev_secondary = int(round(event_rate * n_secondary_budget))

    return np.concatenate([
        _stratified_draw(rng, primary_idx, pool_events,
                         n_ev_primary, n_primary_budget - n_ev_primary),
        _stratified_draw(rng, secondary_idx, pool_events,
                         n_ev_secondary, n_secondary_budget - n_ev_secondary),
    ])


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
        scores.append(
            concordance_index_censored(
                y_primary[EVENT_FIELD][sample],
                y_primary[TIME_FIELD][sample],
                risk_scores[sample],
            )[0]
        )
    return np.array(scores)


def paired_bootstrap_delta(y_primary, risk_a, risk_b,
                           n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Paired bootstrap of C-index(a) - C-index(b) on the SAME resampled patients.
    This is the comparison to report: the regimes share outer folds and test
    patients, so an unpaired comparison of two means throws away that pairing
    and gives needlessly wide intervals.
    """
    rng = np.random.RandomState(seed)
    n = len(y_primary)
    deltas = []
    for _ in range(n_bootstrap):
        sample = rng.choice(np.arange(n), size=n, replace=True)
        if y_primary[EVENT_FIELD][sample].sum() < 2:
            continue
        ev, tm = y_primary[EVENT_FIELD][sample], y_primary[TIME_FIELD][sample]
        c_a = concordance_index_censored(ev, tm, risk_a[sample])[0]
        c_b = concordance_index_censored(ev, tm, risk_b[sample])[0]
        deltas.append(c_a - c_b)
    return np.array(deltas)


def kaplan_meier_by_risk(y_primary, risk_scores, tag, title):
    """Median-split risk groups, log-rank test, KM plot. Returns (chisq, p)."""
    median_risk = np.median(risk_scores)
    risk_group = np.where(risk_scores >= median_risk, "high", "low")

    chisq, pvalue = compare_survival(y_primary, risk_group)

    plt.figure(figsize=(8, 6))
    for group_label in np.unique(risk_group):
        mask = risk_group == group_label
        time_g, surv_g, conf_int = kaplan_meier_estimator(
            y_primary[EVENT_FIELD][mask], y_primary[TIME_FIELD][mask], conf_type="log-log"
        )
        plt.step(time_g, surv_g, where="post", label=f"{group_label} risk (n={mask.sum()})")
        plt.fill_between(time_g, conf_int[0], conf_int[1], alpha=0.2, step="post")

    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(f"{title}\nLog-rank p = {pvalue:.4g}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filename = f"kaplan_meier_{tag}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    mlflow.log_artifact(filename)

    return chisq, pvalue


# =============================================================================
# Main
# =============================================================================

def main(model_type, histology, feature_set):

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("histology", histology)
    mlflow.log_param("feature_set", feature_set)
    mlflow.log_param("n_outer_splits", N_OUTER_SPLITS)
    mlflow.log_param("n_inner_splits", N_INNER_SPLITS)
    mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_matched_repeats", N_MATCHED_REPEATS)
    mlflow.log_param("match_event_count", MATCH_EVENT_COUNT)
    mlflow.log_param("matched_primary_fraction", MATCHED_PRIMARY_FRACTION)
    mlflow.log_param("matched_retune", MATCHED_RETUNE)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    (X, y, times, events, histologies,
     n_primary, n_secondary, n_covariates) = get_data(histology, feature_set)

    X_primary, X_secondary = X[:n_primary], X[n_primary:]
    y_primary, y_secondary = y[:n_primary], y[n_primary:]
    events_primary, events_secondary = events[:n_primary], events[n_primary:]

    mlflow.log_param("n_primary", n_primary)
    mlflow.log_param("n_secondary", n_secondary)
    mlflow.log_metric("n_events_primary", int(events_primary.sum()))
    mlflow.log_metric("n_events_secondary", int(events_secondary.sum()))
    mlflow.log_metric("n_features", X.shape[1])

    if n_secondary == 0:
        raise ValueError("No secondary histologies found — agnostic/matched impossible.")

    # -------------------------------------------------------------------------
    # 2. Outer CV — identical folds shared by all three regimes
    # -------------------------------------------------------------------------
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(outer_cv.split(X_primary, events_primary))

    # [FIX 2] risk score vectors are sized to the PRIMARY cohort only.
    #
    # The original allocated np.zeros(n_samples) with n_samples = primary + secondary,
    # but the outer CV only ever writes into primary indices. Every secondary patient
    # kept a risk score of exactly 0.0 — and the bootstrap CI, the median risk split,
    # the log-rank test and the KM plot were all computed over the FULL y. So for
    # every 'all_*' run those numbers were computed on a cohort where the majority of
    # patients had a constant risk score.
    risk = {
        "specific": np.zeros(n_primary),
        "agnostic": np.zeros(n_primary),
    }
    fold_c = {"specific": [], "agnostic": []}
    risk_matched = np.zeros((N_MATCHED_REPEATS, n_primary))

    # -------------------------------------------------------------------------
    # 3. Fit all three regimes per fold
    # -------------------------------------------------------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):

        X_tr_primary, y_tr_primary = X_primary[train_idx], y_primary[train_idx]
        X_test = X_primary[test_idx]

        # --- specific --------------------------------------------------------
        model_spec, params_spec = fit_model(
            model_type, feature_set, n_covariates, X_tr_primary, y_tr_primary
        )
        risk["specific"][test_idx] = model_spec.predict(X_test)

        # --- agnostic --------------------------------------------------------
        X_tr_pooled = np.vstack([X_tr_primary, X_secondary])
        y_tr_pooled = np.hstack([y_tr_primary, y_secondary])

        model_agn, params_agn = fit_model(
            model_type, feature_set, n_covariates, X_tr_pooled, y_tr_pooled
        )
        risk["agnostic"][test_idx] = model_agn.predict(X_test)

        for regime in ("specific", "agnostic"):
            c = concordance_index_censored(
                y_primary[EVENT_FIELD][test_idx],
                y_primary[TIME_FIELD][test_idx],
                risk[regime][test_idx],
            )[0]
            fold_c[regime].append(c)
            mlflow.log_metric(f"{regime}_c_index_fold_{fold_idx}", c)

        mlflow.log_param(f"specific_best_params_fold_{fold_idx}", params_spec)
        mlflow.log_param(f"agnostic_best_params_fold_{fold_idx}", params_agn)
        mlflow.log_metric(f"specific_n_train_fold_{fold_idx}", len(y_tr_primary))
        mlflow.log_metric(f"agnostic_n_train_fold_{fold_idx}", len(y_tr_pooled))

        print(
            f"Fold {fold_idx}: specific C = {fold_c['specific'][-1]:.3f} "
            f"(n={len(y_tr_primary)}) | agnostic C = {fold_c['agnostic'][-1]:.3f} "
            f"(n={len(y_tr_pooled)})"
        )

        # --- matched (B2) ----------------------------------------------------
        pool_X = X_tr_pooled
        pool_y = y_tr_pooled
        pool_events = np.hstack([events_primary[train_idx], events_secondary])
        pool_is_primary = np.hstack([
            np.ones(len(train_idx), dtype=bool),
            np.zeros(n_secondary, dtype=bool),
        ])

        # the budget: exactly what the histology-specific model gets
        n_target = len(train_idx)
        n_events_target = int(events_primary[train_idx].sum())

        # Reusing the agnostic fold's hyperparameters isolates the effect of
        # training-set size from tuning noise, and avoids re-running the inner CV
        # N_MATCHED_REPEATS times. No test-fold leakage: the agnostic fit never
        # saw these test patients either.
        fixed = None if MATCHED_RETUNE else params_agn

        print(f"  matched: n={n_target} ({n_events_target} events) "
              f"from a pool of {len(pool_y)}, {N_MATCHED_REPEATS} repetitions")

        for rep in range(N_MATCHED_REPEATS):
            rng = np.random.RandomState(RANDOM_STATE + 10_000 * fold_idx + rep)

            for _attempt in range(20):
                sel = draw_matched_indices(
                    rng=rng,
                    pool_is_primary=pool_is_primary,
                    pool_events=pool_events,
                    n_target=n_target,
                    n_events_target=n_events_target,
                    primary_fraction=MATCHED_PRIMARY_FRACTION,
                    match_event_count=MATCH_EVENT_COUNT,
                )
                if pool_events[sel].sum() >= MIN_EVENTS_IN_TRAIN:
                    break

            model_matched, _ = fit_model(
                model_type, feature_set, n_covariates,
                pool_X[sel], pool_y[sel], fixed_params=fixed,
            )
            risk_matched[rep, test_idx] = model_matched.predict(X_test)

        mlflow.log_metric(f"matched_n_target_fold_{fold_idx}", n_target)
        mlflow.log_metric(f"matched_n_events_target_fold_{fold_idx}", n_events_target)

    # -------------------------------------------------------------------------
    # 4. Matched: distribution of the C-index over subsampling repetitions
    # -------------------------------------------------------------------------
    matched_rep_c = np.array([
        concordance_index_censored(
            y_primary[EVENT_FIELD], y_primary[TIME_FIELD], risk_matched[rep]
        )[0]
        for rep in range(N_MATCHED_REPEATS)
    ])

    mlflow.log_metric("matched_rep_c_index_mean", float(matched_rep_c.mean()))
    mlflow.log_metric("matched_rep_c_index_std", float(matched_rep_c.std()))
    mlflow.log_metric("matched_rep_c_index_median", float(np.median(matched_rep_c)))
    mlflow.log_metric("matched_rep_c_index_p2_5", float(np.percentile(matched_rep_c, 2.5)))
    mlflow.log_metric("matched_rep_c_index_p97_5", float(np.percentile(matched_rep_c, 97.5)))

    # Risk scores are on different scales across repetitions (and across models),
    # so rank-normalise each repetition before averaging into one vector.
    risk["matched"] = np.mean(
        [rankdata(risk_matched[rep]) for rep in range(N_MATCHED_REPEATS)], axis=0
    )

    plt.figure(figsize=(7, 5))
    plt.hist(matched_rep_c, bins=25, alpha=0.8, label="matched (per repetition)")
    plt.axvline(np.mean(fold_c["specific"]), color="C1", linestyle="--",
                label=f"specific = {np.mean(fold_c['specific']):.3f}")
    plt.axvline(np.mean(fold_c["agnostic"]), color="C2", linestyle="--",
                label=f"agnostic = {np.mean(fold_c['agnostic']):.3f}")
    plt.xlabel("C-index")
    plt.ylabel("Repetitions")
    plt.title(f"{histology} | {feature_set} | {model_type}\nB2 size-matched control")
    plt.legend()
    plt.tight_layout()
    plt.savefig("matched_c_index_distribution.png", dpi=300)
    plt.close()
    mlflow.log_artifact("matched_c_index_distribution.png")

    # -------------------------------------------------------------------------
    # 5. Per-regime metrics
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
                title=f"{histology} | {regime} | {feature_set} | {model_type}",
            )
            mlflow.log_metric(f"{regime}_logrank_chisq", float(chisq))
            mlflow.log_metric(f"{regime}_logrank_pvalue", float(pvalue))
            mlflow.log_param(f"{regime}_significant", bool(pvalue < 0.05 and lo > 0.5))
        except Exception as exc:
            print(f"  KM / log-rank failed for {regime}: {exc}")
            mlflow.log_param(f"{regime}_km_error", str(exc))

    # -------------------------------------------------------------------------
    # 6. Paired comparisons — this is the B2 result
    # -------------------------------------------------------------------------
    print()
    comparisons = [
        ("agnostic", "specific"),
        ("matched", "specific"),
        ("agnostic", "matched"),
    ]
    for a, b in comparisons:
        deltas = paired_bootstrap_delta(y_primary, risk[a], risk[b])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        # two-sided bootstrap p-value for delta = 0
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

        name = f"delta_{a}_vs_{b}"
        mlflow.log_metric(f"{name}_mean", float(deltas.mean()))
        mlflow.log_metric(f"{name}_lower", float(lo))
        mlflow.log_metric(f"{name}_upper", float(hi))
        mlflow.log_metric(f"{name}_pvalue", float(p))

        print(f"  {a} - {b}: {deltas.mean():+.3f} "
              f"(95% CI [{lo:+.3f}, {hi:+.3f}], p = {p:.3g})")

    # -------------------------------------------------------------------------
    # 7. Persist out-of-fold predictions for later paired / pooled analyses
    # -------------------------------------------------------------------------
    np.savez(
        "oof_risk_scores.npz",
        specific=risk["specific"],
        agnostic=risk["agnostic"],
        matched=risk["matched"],
        matched_matrix=risk_matched,
        matched_rep_c=matched_rep_c,
        event=y_primary[EVENT_FIELD],
        duration=y_primary[TIME_FIELD],
    )
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
    MODEL_TYPES = ["rsf", "extra_trees", "gradient_boosting",
                   "cox_elastic", "svm", "lipschitz_svm"]
    HISTOLOGIES = ["syn", "mfh", "lipo"]

    mlflow.set_experiment("DEGRO_journal_extension_matched")

    for feature_set in FEATURE_SETS:
        for model_type in MODEL_TYPES:
            for histology in HISTOLOGIES:

                print("\n" + "#" * 79)
                print(f"# {model_type} | {histology} | {feature_set} "
                      f"| specific + agnostic + matched")
                print("#" * 79 + "\n")

                mlflow.start_run()
                try:
                    main(
                        model_type=model_type,
                        histology=histology,
                        feature_set=feature_set,
                    )
                except Exception as exc:
                    print(f"RUN FAILED: {exc}")
                    mlflow.log_param("run_error", str(exc))
                finally:
                    mlflow.end_run()