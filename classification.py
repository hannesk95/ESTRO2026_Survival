from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from glob import glob
import torch
import numpy as np
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from xgboost import XGBClassifier
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


def get_model_and_param_grid(model_type):
    
    if model_type == 'rf':
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(random_state=42))
        ])
        param_grid = {       
            # --- Core ensemble parameters ---
            'clf__n_estimators': [100, 300, 500, 1000],
            'clf__criterion': ['gini', 'entropy', 'log_loss'],

            # --- Tree structure ---
            'clf__max_depth': [None, 5, 10, 20],

            # --- PCA components ---
            "pca__n_components": [0.95, 0.99]
        }
    elif model_type == "svm":
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("clf", SVC(probability=True, random_state=42))
        ])
        param_grid = {
            # --- Core model type ---
            'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            # --- Regularization strength ---
            'clf__C': [0.01, 0.1, 1, 10, 100],

            # --- PCA components ---
            "pca__n_components": [0.95, 0.99]
        }
    
    elif model_type == "xgb":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(random_state=42))
        ])
        param_grid = {
            # --- 1. Core ensemble setup ---
            'clf__n_estimators': [100, 200, 300, 500],
            'clf__booster': ['gbtree', 'dart'],  # 'gblinear' usually worse for classification

            # --- 2. Tree complexity ---
            'clf__max_depth': [3, 5, 7, 9, 12],

            # --- PCA components ---
            "pca__n_components": [0.95, 0.99]
        }
    
    else:
        raise ValueError("Unsupported model type")

    return pipeline, param_grid

def main(model_type:str):

    mlflow.log_param("model_type", model_type)

    data = sorted(glob("/media/johannes/SSD_2.0TB/Duschinger/data/classification/duschinger/final_dataset/*.pt"))

    features = []
    labels = []
    for file in data:
        dictionary = torch.load(file)
        features.append(list(dictionary.values()))
        subtype = file.split("/")[-1].split("_")[-3]
        labels.append(0 if subtype == 'SYN' else 1 if subtype == 'MFH' else 2)

    X = np.array(features)
    y = np.array(labels)

    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    outer_results_bacc = list()
    outer_results_f1 = list()
    outer_results_mcc = list()
    outer_results_auroc = list()

    outer_true = list()
    outer_pred = list()

    for i, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # define the model
        pipeline, param_grid = get_model_and_param_grid(model_type=model_type)
        
        # define search
        search = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)

        outer_true.extend(y_test)
        outer_pred.extend(yhat)
        
        # evaluate the model        
        bacc = balanced_accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average='weighted')
        mcc = matthews_corrcoef(y_test, yhat)
        auroc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

        mlflow.log_metric(f"bacc_fold_{i}", bacc)
        mlflow.log_metric(f"f1_fold_{i}", f1)
        mlflow.log_metric(f"mcc_fold_{i}", mcc)
        mlflow.log_metric(f"auroc_fold_{i}", auroc)

        outer_results_bacc.append(bacc)
        outer_results_f1.append(f1)
        outer_results_mcc.append(mcc)
        outer_results_auroc.append(auroc)
        
        # report progress
        print('>bacc=%.3f, est=%.3f, cfg=%s' % (bacc, result.best_score_, result.best_params_))

    # summarize the estimated performance of the model
    print('Balanced Accuracy:   %.3f (%.3f)' % (mean(outer_results_bacc), std(outer_results_bacc)))
    print('F1 Score:            %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
    print('MCC:                 %.3f (%.3f)' % (mean(outer_results_mcc), std(outer_results_mcc)))
    print('AUROC:               %.3f (%.3f)' % (mean(outer_results_auroc), std(outer_results_auroc)))

    mlflow.log_metric("bacc_mean", mean(outer_results_bacc))
    mlflow.log_metric("bacc_std", std(outer_results_bacc))
    mlflow.log_metric("f1_mean", mean(outer_results_f1))
    mlflow.log_metric("f1_std", std(outer_results_f1))
    mlflow.log_metric("mcc_mean", mean(outer_results_mcc))
    mlflow.log_metric("mcc_std", std(outer_results_mcc))
    mlflow.log_metric("auroc_mean", mean(outer_results_auroc))
    mlflow.log_metric("auroc_std", std(outer_results_auroc))

    # ---------------------------------------------------------------------
    # Confusion matrices for visualization
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(confusion_matrix(outer_true, outer_pred), annot=True, fmt="d", cmap="Blues", ax=axes)
    axes.set_title(f"Confusion Matrix | Model: {model_type}")
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    os.remove("confusion_matrix.png")

if __name__ == "__main__":

    for model_type in ['svm', "rf"]:
        mlflow.set_experiment("classification")
        mlflow.start_run()
        print(f"Model type: {model_type}")
        main(model_type=model_type)
        mlflow.end_run()
