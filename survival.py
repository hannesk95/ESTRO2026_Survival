from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from glob import glob
from matplotlib.lines import Line2D
import torch
import numpy as np
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from xgboost import DMatrix, XGBRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored
import pandas as pd
import mlflow
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

RANDOM_STATE = 42

def bootstrap_confidence_interval(estimator, X_test, y_test, n_bootstrap=10000, random_state=RANDOM_STATE):
    
    np.random.seed(random_state)  

    bootstrap_scores = []
    n_samples = len(X_test)
      

    for i in range(n_bootstrap):
        # Bootstrap sampling with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_bootstrap = X_test[bootstrap_indices]
        y_bootstrap = y_test[bootstrap_indices]        
        
        score = estimator.score(X_bootstrap, y_bootstrap) 
        score = score.item()
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)

    lower_bound, upper_bound = np.percentile(bootstrap_scores, [2.5, 97.5])

    lower_bound = lower_bound.item()
    upper_bound = upper_bound.item()

    return lower_bound, upper_bound

def get_model_and_param_grid(model_type):
    
    if model_type == 'rf':
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            # ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomSurvivalForest(random_state=RANDOM_STATE))
        ])
        param_grid = {
            "clf__n_estimators": [10, 50, 100, 300, 500, 1000],
            # "clf__max_depth": [None, 10, 20],
            # "clf__min_samples_split": [2, 5, 6, 10],
            # "clf__min_samples_leaf": [1, 3, 5],
            # "clf__max_features": [None, "sqrt", "log2"],
            "pca__n_components": [0.90, 0.95, 0.99]
        }
    elif model_type == "svm":
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            # ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", FastSurvivalSVM(random_state=RANDOM_STATE))
        ])
        param_grid = {
            "clf__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__max_iter": [10000],
            # "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "pca__n_components": [0.90, 0.95, 0.99]
        }
    elif model_type == "xgb":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            # ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBRegressor(objective="survival:cox", eval_metric="cox-nloglik"))
        ])
        
        param_grid = {
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__n_estimators": [100, 200, 500],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "pca__n_components": [0.95, 0.99]
        }
    
    else:
        raise ValueError("Unsupported model type")

    return pipeline, param_grid

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["event"], y["duration"], prediction)
    return result[0]

def get_data(histology:str, feature_set:str):
    
    match histology:
        case "syn":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*SYN*.pt"))
        case "mfh":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*MFH*.pt"))
        case "lipo":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*Lipo*.pt"))
        case "all":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*.pt"))
        case "all_syn":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*.pt"))
        case "all_mfh":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*.pt"))
        case "all_lipo":
            data = sorted(glob("/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset/*.pt"))
    
    match feature_set:
        case "volume":
            
            features = []
            labels_event = []
            labels_time = []
            histologies = []

            for i, file in enumerate(data):
                # Load features
                dictionary = torch.load(file, map_location='cpu')

                # volume only
                features.append(dictionary["original_shape_VoxelVolume"])        
                
                # Extract survival information from filename
                filename_parts = file.split("/")[-1].split("_")
                event = bool(int(filename_parts[-2]))
                time = int(filename_parts[-3])      
                time = max(float(filename_parts[-3]), 1e-6)  # Time (ensure positive for SVM)      
                histology = filename_parts[3]   
                histologies.append(histology)      

                labels_event.append(event)
                labels_time.append(time)              

            X = np.array(features)
            X = X.reshape(X.shape[0], -1)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float32)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
        
            return X, y, events, histologies

        case "clinical":
            features = []
            labels_event = []
            labels_time = []
            ages = []
            t_list = []
            n_list = []
            m_list = []
            grading_list = []
            histologies = []

            df = pd.read_csv('clinical_cleaned.csv')

            for i, file in enumerate(data):
                filename_parts = file.split("/")[-1].split("_")
                histology = filename_parts[3]   
                histologies.append(histology) 

                try:
                    age = df[df['Pseudonym'] == filename_parts[0]].Age.item()
                    grading = df[df['Pseudonym'] == filename_parts[0]].Grading.item()
                    t = df[df['Pseudonym'] == filename_parts[0]].TNMT.item()
                    n = df[df['Pseudonym'] == filename_parts[0]].TNMN.item()
                    m = df[df['Pseudonym'] == filename_parts[0]].TNMM.item()

                    ages.append(age)
                    grading_list.append(grading)
                    t_list.append(t)
                    n_list.append(n)
                    m_list.append(m)

                    event = bool(int(filename_parts[-2]))
                    time = int(filename_parts[-3])
                    time = max(float(filename_parts[-3]), 1e-6)  # Time (ensure positive for SVM)                

                    labels_event.append(event)
                    labels_time.append(time)
                except:
                    continue

            ages = np.array(ages).reshape(-1, 1)
            grading_list = np.array(grading_list).reshape(-1, 1)
            t_list = np.array(t_list).reshape(-1, 1)
            n_list = np.array(n_list).reshape(-1, 1)
            m_list = np.array(m_list).reshape(-1, 1)
            X = np.hstack([ages, grading_list, t_list, n_list, m_list])
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float32)])
            events = np.array(labels_event)
            histologies = np.array(histologies)

            return X, y, events, histologies

        case "radiomics":

            features = []
            labels_event = []
            labels_time = []
            histologies = []

            for i, file in enumerate(data):
                # Load features
                dictionary = torch.load(file, map_location='cpu')

                # all features
                features.append(list(dictionary.values()))
                
                # Extract survival information from filename
                filename_parts = file.split("/")[-1].split("_")
                event = bool(int(filename_parts[-2]))
                time = int(filename_parts[-3])         
                time = max(float(filename_parts[-3]), 1e-6)  # Time (ensure positive for SVM)   
                histology = filename_parts[3]   
                histologies.append(histology)          

                labels_event.append(event)
                labels_time.append(time)              

            X = np.array(features)
            X = X.reshape(X.shape[0], -1)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float32)])
            events = np.array(labels_event)
            histologies = np.array(histologies)

            return X, y, events, histologies

        case "combined":
            features = []
            labels_event = []
            labels_time = []
            ages = []
            t_list = []
            n_list = []
            m_list = []
            grading_list = []
            histologies = []

            df = pd.read_csv('clinical_cleaned.csv')

            for i, file in enumerate(data):
                filename_parts = file.split("/")[-1].split("_")
                histology = filename_parts[3]   
                histologies.append(histology) 

                try:
                    # Load features
                    dictionary = torch.load(file, map_location='cpu')

                    # all features
                    feature = list(dictionary.values())
                    age = df[df['Pseudonym'] == filename_parts[0]].Age.item()
                    grading = df[df['Pseudonym'] == filename_parts[0]].Grading.item()
                    t = df[df['Pseudonym'] == filename_parts[0]].TNMT.item()
                    n = df[df['Pseudonym'] == filename_parts[0]].TNMN.item()
                    m = df[df['Pseudonym'] == filename_parts[0]].TNMM.item()

                    features.append(feature)
                    ages.append(age)
                    grading_list.append(grading)
                    t_list.append(t)
                    n_list.append(n)
                    m_list.append(m)

                    event = bool(int(filename_parts[-2]))
                    time = int(filename_parts[-3])     
                    time = max(float(filename_parts[-3]), 1e-6)  # Time (ensure positive for SVM)                

                    labels_event.append(event)
                    labels_time.append(time)
                except:
                    continue

            features = np.array(features)
            ages = np.array(ages).reshape(-1, 1)
            grading_list = np.array(grading_list).reshape(-1, 1)
            t_list = np.array(t_list).reshape(-1, 1)
            n_list = np.array(n_list).reshape(-1, 1)
            m_list = np.array(m_list).reshape(-1, 1)
            X = np.hstack([features, ages, grading_list, t_list, n_list, m_list])
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float32)])
            events = np.array(labels_event)
            histologies = np.array(histologies)

            return X, y, events, histologies

def main(model_type:str, histology:str, feature_set:str):

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("histology", histology)
    mlflow.log_param("feature_set", feature_set)

    X, y, events, histologies = get_data(histology=histology, feature_set=feature_set)

    # if histology in ["all_syn", "all_mfh", "all_lipo"]:
    events = [a + b for a, b in zip(list(events.astype(int).astype(str)), list(histologies))]

    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    outer_results_c_index = list()
    survival_times = list()
    high_risk_low_risk_labels = list()

    for fold_idx, (train_ix, test_ix) in enumerate(cv_outer.split(X, events)):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        histologies_train, histologies_test = histologies[train_ix], histologies[test_ix]

        if histology == "all_syn":       

            keep_indices = []
            drop_indices = []
            for idx, h in enumerate(histologies_test):
                if h == "SYN":
                    keep_indices.append(idx)
                else:
                    drop_indices.append(idx)

            keep_index_test = np.array(keep_indices)
            drop_index_test = np.array(drop_indices)
            
            X_test_temp = X_test[drop_index_test, :]
            y_test_temp = y_test[drop_index_test]
            X_train = np.vstack([X_train, X_test_temp])
            y_train = np.hstack([y_train, y_test_temp])

            X_test = X_test[keep_index_test, :]
            y_test = y_test[keep_index_test]
        
        if histology == "all_mfh":       

            keep_indices = []
            drop_indices = []
            for idx, h in enumerate(histologies_test):
                if h == "MFH":
                    keep_indices.append(idx)
                else:
                    drop_indices.append(idx)

            keep_index_test = np.array(keep_indices)
            drop_index_test = np.array(drop_indices)
            
            X_test_temp = X_test[drop_index_test, :]
            y_test_temp = y_test[drop_index_test]
            X_train = np.vstack([X_train, X_test_temp])
            y_train = np.hstack([y_train, y_test_temp])

            X_test = X_test[keep_index_test, :]
            y_test = y_test[keep_index_test]
        
        if histology == "all_lipo":       

            keep_indices = []
            drop_indices = []
            for idx, h in enumerate(histologies_test):
                if h == "Liposarcoma":
                    keep_indices.append(idx)
                else:
                    drop_indices.append(idx)

            keep_index_test = np.array(keep_indices)
            drop_index_test = np.array(drop_indices)
            
            X_test_temp = X_test[drop_index_test, :]
            y_test_temp = y_test[drop_index_test]
            X_train = np.vstack([X_train, X_test_temp])
            y_train = np.hstack([y_train, y_test_temp])

            X_test = X_test[keep_index_test, :]
            y_test = y_test[keep_index_test]

        
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        # define the model
        pipeline, param_grid = get_model_and_param_grid(model_type=model_type)
        
        # define search
        search = GridSearchCV(pipeline, param_grid, scoring=score_survival_model, cv=cv_inner, refit=True)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_       

        risk_scores_train = best_model.predict(X_train)
        median_risk_score = np.median(risk_scores_train).item()

        risk_scores_test = best_model.predict(X_test)
        risk_scores_test_binary = (risk_scores_test >= median_risk_score).astype(int)

        survival_times.extend(list(y_test["duration"]))        
        high_risk_low_risk_labels.extend(list(risk_scores_test_binary))

        c_index = best_model.score(X_test, y_test)
        mlflow.log_metric(f"c_index_{fold_idx + 1}", c_index)
        outer_results_c_index.append(c_index)

        # report progress
        print(f"Fold {fold_idx + 1}:")
        print('>c_index=%.3f, est=%.3f, cfg=%s' % (c_index, result.best_score_, result.best_params_))

        mlflow.log_param(f"best_params_fold_{fold_idx + 1}", result.best_params_)

    # summarize the estimated performance of the model
    print('Concordance Index:   %.3f (%.3f)\n' % (mean(outer_results_c_index), std(outer_results_c_index)))
    mlflow.log_metric("c_index_mean", mean(outer_results_c_index))
    mlflow.log_metric("c_index_std", std(outer_results_c_index))

    # Plot Kaplan-Meier curves and perform log-rank test
    survival_times = [time.item() for time in survival_times]
    high_risk_low_risk_labels = [label.item() for label in high_risk_low_risk_labels]

    high_risk_survival_times = [survival_times[i] for i in range(len(survival_times)) if high_risk_low_risk_labels[i] == 1]
    low_risk_survival_times = [survival_times[i] for i in range(len(survival_times)) if high_risk_low_risk_labels[i] == 0]

    results = logrank_test(
        high_risk_survival_times, low_risk_survival_times,
        event_observed_A=np.ones(len(high_risk_survival_times)),
        event_observed_B=np.ones(len(low_risk_survival_times))
    )

    p_value = results.p_value

    mlflow.log_metric("logrank_p_value", p_value)

    if p_value >= 0.001:
        p_value = round(p_value, 4)
    else:
        p_value = "<0.001"

    plt.figure(figsize=(8,6))
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=high_risk_survival_times, event_observed=np.ones(len(high_risk_survival_times)), label="high_risk")
    ax = kmf.plot(ci_show=True)   
    kmf.fit(durations=low_risk_survival_times, event_observed=np.ones(len(low_risk_survival_times)), label="low_risk")
    kmf.plot(ax=ax, ci_show=True)

    plt.title(f"Model: {model_type} | Histology: {histology} | Features: {feature_set}")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.plot([], [], ' ', label=f"p-value: {p_value}")
    plt.legend()
    plt.savefig(f"km_curve_model_{model_type}_histology_{histology}_features_{feature_set}.png")

    mlflow.log_artifact(f"km_curve_model_{model_type}_histology_{histology}_features_{feature_set}.png")
    os.remove(f"km_curve_model_{model_type}_histology_{histology}_features_{feature_set}.png")

if __name__ == "__main__":

    for feature_set in ['volume', 'clinical', 'radiomics', 'combined']:
        for model_type in ['svm', 'rf']:
            for histology in ['syn', 'all_syn', 'mfh', 'all_mfh', 'lipo', 'all_lipo', 'all']:
                print(f"Model type: {model_type}, Histology: {histology}, Feature set: {feature_set}")

                mlflow.set_experiment("survival")
                mlflow.start_run()
                main(model_type=model_type, histology=histology, feature_set=feature_set)
                mlflow.end_run()