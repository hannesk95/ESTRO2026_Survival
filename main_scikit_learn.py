import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

from xgboost import XGBRegressor
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM, MinlipSurvivalAnalysis
from sklearn.decomposition import PCA
from joblib import Memory
from utils import score_survival_model

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

EVENT_FIELD = "event"   
TIME_FIELD = "duration"   
MEMORY = Memory(location='/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/sklearn_cache', verbose=0)

def get_data(histology:str, feature_set:str):

    df = pd.read_csv("/home/johannes/Data/SSD_2.0TB/ESTRO2026_Survival/journal_extension_data.csv")

    if histology == "syn":
        patient_id_primary = sorted(df[df["histology"] == "SYN"].Pseudonym.tolist())
        patient_id_secondary = []
    
    elif histology == "all_syn":
        patient_id_primary = sorted(df[df["histology"] == "SYN"].Pseudonym.tolist())
        patient_id_secondary = sorted(df[df["histology"] != "SYN"].Pseudonym.tolist())
    
    elif histology == "mfh":
        patient_id_primary = sorted(df[df["histology"] == "MFH"].Pseudonym.tolist())
        patient_id_secondary = []
    
    elif histology == "all_mfh":
        patient_id_primary = sorted(df[df["histology"] == "MFH"].Pseudonym.tolist())
        patient_id_secondary = sorted(df[df["histology"] != "MFH"].Pseudonym.tolist())
    
    elif histology == "lipo":
        patient_id_primary = sorted(df[df["histology"] == "Liposarcoma"].Pseudonym.tolist())
        patient_id_secondary = []
    
    elif histology == "all_lipo":
        patient_id_primary = sorted(df[df["histology"] == "Liposarcoma"].Pseudonym.tolist())
        patient_id_secondary = sorted(df[df["histology"] != "Liposarcoma"].Pseudonym.tolist())    
    
    else:
        raise ValueError(f"Unsupported histology: {histology}")
    
    n_primary = len(patient_id_primary)
    n_secondary = len(patient_id_secondary)
    patient_ids = patient_id_primary + patient_id_secondary

    match feature_set:
        case "volume":
            
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])               
         
                features.append(volume)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features).reshape(-1, 1)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "clinical":
            
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                age = float(patient_data['Age'].values[0])    
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])           
         
                features.append([age, grade, t_stage, n_stage])       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "radiomics":

            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]

                # take all values starting with "radiomics_" but without "shape" in the name
                radiomics_features = patient_data.filter(regex='^radiomics_').filter(regex='^(?!.*shape)').values.flatten().tolist()         
         
                features.append(radiomics_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary
        
        case "ctfm":

            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]

                # take all values starting with "ctfm_"
                ct_features = patient_data.filter(regex='^ctfm_').values.flatten().tolist()         
         
                features.append(ct_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary
        
        case "omnislicer":

            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]

                # take all values starting with "omnislicer_"
                omnislicer_features = patient_data.filter(regex='^omnislicer_').values.flatten().tolist()         
         
                features.append(omnislicer_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_clinical":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])   
                age = float(patient_data['Age'].values[0])    
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])             
         
                features.append([volume, age, grade, t_stage, n_stage])       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_radiomics":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])

                # take all values starting with "radiomics_" but without "shape" in the name
                radiomics_features = patient_data.filter(regex='^radiomics_').filter(regex='^(?!.*shape)').values.flatten().tolist()         
         
                features.append([volume] + radiomics_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_ctfm":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])

                # take all values starting with "ctfm_"
                ct_features = patient_data.filter(regex='^ctfm_').values.flatten().tolist()         
         
                features.append([volume] + ct_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_omnislicer":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])

                # take all values starting with "omnislicer_"
                omnislicer_features = patient_data.filter(regex='^omnislicer_').values.flatten().tolist()         
         
                features.append([volume] + omnislicer_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "clinical_radiomics":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "radiomics_" but without "shape" in the name
                radiomics_features = patient_data.filter(regex='^radiomics_').filter(regex='^(?!.*shape)').values.flatten().tolist()         
         
                features.append([age, grade, t_stage, n_stage] + radiomics_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary
            
        case "clinical_ctfm":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "ctfm_"
                ct_features = patient_data.filter(regex='^ctfm_').values.flatten().tolist()         
         
                features.append([age, grade, t_stage, n_stage] + ct_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary
            
        case "clinical_omnislicer":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "omnislicer_"
                omnislicer_features = patient_data.filter(regex='^omnislicer_').values.flatten().tolist()         
         
                features.append([age, grade, t_stage, n_stage] + omnislicer_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_clinical_radiomics":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "radiomics_" but without "shape" in the name
                radiomics_features = patient_data.filter(regex='^radiomics_').filter(regex='^(?!.*shape)').values.flatten().tolist()         
         
                features.append([volume, age, grade, t_stage, n_stage] + radiomics_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_clinical_ctfm":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "ctfm_"
                ct_features = patient_data.filter(regex='^ctfm_').values.flatten().tolist()         
         
                features.append([volume, age, grade, t_stage, n_stage] + ct_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_clinical_omnislicer":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "omnislicer_"
                omnislicer_features = patient_data.filter(regex='^omnislicer_').values.flatten().tolist()         
         
                features.append([volume, age, grade, t_stage, n_stage] + omnislicer_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case "volume_clinical_radiomics_ctfm_omnislicer":
            features = []
            labels_event = []
            labels_time = []
            histologies = []  

            for patient_id in patient_ids:

                patient_data = df[df['Pseudonym'] == patient_id]
                volume = float(patient_data['radiomics_original_shape_VoxelVolume'].values[0])
                age = float(patient_data['Age'].values[0])
                grade = float(patient_data['Grading'].values[0])
                t_stage = float(patient_data['TNMT'].values[0])
                n_stage = float(patient_data['TNMN'].values[0])

                # take all values starting with "radiomics_" but without "shape" in the name
                radiomics_features = patient_data.filter(regex='^radiomics_').filter(regex='^(?!.*shape)').values.flatten().tolist()         
                
                # take all values starting with "ctfm_"
                ct_features = patient_data.filter(regex='^ctfm_').values.flatten().tolist()         
                
                # take all values starting with "omnislicer_"
                omnislicer_features = patient_data.filter(regex='^omnislicer_').values.flatten().tolist()         
         
                features.append([volume, age, grade, t_stage, n_stage] + radiomics_features + ct_features + omnislicer_features)       
                
                histology_subtype = patient_data['histology'].values[0]   
                histologies.append(histology_subtype)

                event = bool(patient_data['event'].values[0])
                labels_event.append(event)

                time = float(patient_data['time'].values[0])
                labels_time.append(time)

            X = np.array(features)
            y = np.array(list(zip(labels_event, labels_time)), dtype=[('event', bool), ('duration', np.float64)])
            events = np.array(labels_event)
            histologies = np.array(histologies)
            times = np.array(labels_time)

            return X, y, times, events, histologies, n_primary, n_secondary

        case _:
            raise ValueError(f"Unsupported feature set: {feature_set}")

def get_model_and_param_grid(model_type):

    # Number of trailing covariates that must be excluded from PCA.
    # get_data() constructs combined features as:
    # [imaging features, volume (optional), Age, Grading, TNMT, TNMN (optional)]
    n_passthrough_features = {
        'volume': 1, 
        'clinical': 4, 
        'radiomics': 0, 
        'ctfm': 0, 
        'omnislicer': 0,
        'volume_clinical': 5, 
        'volume_radiomics': 1, 
        'volume_ctfm': 1, 
        'volume_omnislicer': 1,
        'clinical_radiomics': 4, 
        'clinical_ctfm': 4, 
        'clinical_omnislicer': 4,
        'volume_clinical_radiomics': 5, 
        'volume_clinical_ctfm': 5, 
        'volume_clinical_omnislicer': 5,
        'volume_clinical_radiomics_ctfm_omnislicer': 5
    }

    # Preserve the original behavior for Cox elastic net: scaling, but no PCA.
    use_pca = feature_set not in ["volume", "clinical", "volume_clinical"] and model_type != "cox_elastic" and model_type != "cox_ph"

    if use_pca:
        if feature_set not in n_passthrough_features:
            raise ValueError(f"Unsupported feature set for PCA preprocessing: {feature_set}")

        n_passthrough = n_passthrough_features[feature_set]

        if n_passthrough == 0:
            preprocessor = PCA(random_state=RANDOM_STATE)
            pca_param_name = "preprocessor__n_components"
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "pca",
                        PCA(random_state=RANDOM_STATE),
                        slice(0, n_passthrough),
                    ),
                    (
                        "covariates",
                        "passthrough",
                        slice(n_passthrough, None),
                    ),
                ],
                remainder="drop",
            )
            pca_param_name = "preprocessor__pca__n_components"
    
    else:
        preprocessor = "passthrough"
        pca_param_name = None

    # ================================================================
    # Random Survival Forest
    # ================================================================

    if model_type == "rsf":

        clf = RandomSurvivalForest(random_state=RANDOM_STATE)
        param_grid = {}

    # ================================================================
    # Extra Survival Trees
    # ================================================================

    elif model_type == "extra_trees":

        clf = ExtraSurvivalTrees(random_state=RANDOM_STATE)
        param_grid = {}

    # ================================================================
    # Gradient Boosting Survival Analysis
    # ================================================================

    elif model_type == "gradient_boosting":

        clf = GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE)
        param_grid = {}

    # ================================================================
    # Cox Proportional Hazards
    # ================================================================

    elif model_type == "cox_ph":

        clf = CoxPHSurvivalAnalysis()
        param_grid = {}

    # ================================================================
    # Cox Elastic Net
    # ================================================================

    elif model_type == "cox_elastic":

        clf = CoxnetSurvivalAnalysis()
        param_grid = {}

    # ================================================================
    # Fast Survival SVM
    # ================================================================

    elif model_type == "svm":

        clf = FastSurvivalSVM(random_state=RANDOM_STATE)
        param_grid = {}

    # ================================================================
    # Minimal Lipschitz Survival SVM
    # ================================================================

    elif model_type == "lipschitz_svm":

        clf = MinlipSurvivalAnalysis()
        param_grid = {}

    # ================================================================
    # Unsupported model
    # ================================================================

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("preprocessor", preprocessor),
            ("clf", clf),
        ],
        memory=MEMORY,
    )

    if use_pca:
        param_grid[pca_param_name] = [0.80, 0.90, 0.95, 0.99]

    return pipeline, param_grid

def main(model_type:str, histology:str, feature_set:str):

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("histology", histology)
    mlflow.log_param("feature_set", feature_set)
    mlflow.log_param("n_outer_splits", N_OUTER_SPLITS)
    mlflow.log_param("n_inner_splits", N_INNER_SPLITS)
    mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
    mlflow.log_param("random_state", RANDOM_STATE)

    # ---------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------
    X, y, times, events, histologies, n_primary, n_secondary = get_data(histology=histology, feature_set=feature_set)
    
    X_primary = X[:n_primary]
    X_secondary = X[n_primary:]

    y_primary = y[:n_primary]
    y_secondary = y[n_primary:]

    events_primary = events[:n_primary]
    events_secondary = events[n_primary:]

    n_samples = len(y)
    mlflow.log_param("n_samples", n_samples)

    # ---------------------------------------------------------------------
    # 2. Define outer cross-validation
    # ---------------------------------------------------------------------

    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = []
    risk_scores = np.zeros(n_samples)

    # ---------------------------------------------------------------------
    # 3. stratified nested cross-validation
    # ---------------------------------------------------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_primary, events_primary), start=1):
        X_train, X_test = X_primary[train_idx, :], X_primary[test_idx, :]
        y_train, y_test = y_primary[train_idx], y_primary[test_idx]
        events_train = events_primary[train_idx]

        X_train = np.vstack([X_train, X_secondary])
        y_train = np.hstack([y_train, y_secondary])
        events_train = np.hstack([events_train, events_secondary])

        cv_inner = RepeatedKFold(n_splits=N_INNER_SPLITS, n_repeats=100, random_state=RANDOM_STATE)   

        pipeline, param_grid = get_model_and_param_grid(model_type=model_type)

        grid_search = GridSearchCV(pipeline, param_grid, scoring=score_survival_model, cv=cv_inner, refit=True, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        risk_scores[test_idx] = best_model.predict(X_test)

        preprocessor = best_model.named_steps["preprocessor"]
        if isinstance(preprocessor, PCA):
            n_pca_features = preprocessor.n_components_
            n_final_features = n_pca_features

        elif isinstance(preprocessor, ColumnTransformer):
            pca = preprocessor.named_transformers_["pca"]
            n_pca_features = pca.n_components_
            n_final_features = preprocessor.transform(X_train[:1]).shape[1]

        else:
            n_pca_features = None
            n_final_features = X_train.shape[1]

        if n_pca_features is not None:
            print("Number of PCA features:", n_pca_features)
            mlflow.log_param(
                f"n_pca_features_fold_{fold_idx + 1}",
                n_pca_features,
            )

        c_index = concordance_index_censored(y_test[EVENT_FIELD], y_test[TIME_FIELD], risk_scores[test_idx])[0]
        outer_scores.append(c_index)

        print(f"Fold {fold_idx}: C-index = {c_index:.3f} | best params = {grid_search.best_params_}")
        mlflow.log_metric(f"c_index_fold_{fold_idx}", c_index)
        mlflow.log_param(f"best_params_fold_{fold_idx}", grid_search.best_params_)

    
    c_index_mean = np.mean(outer_scores)
    c_index_std = np.std(outer_scores)

    print(f"\nOverall c-index: {c_index_mean:.3f} +/- {c_index_std:.3f}")

    mlflow.log_metric("c_index_mean", c_index_mean)
    mlflow.log_metric("c_index_std", c_index_std)

    # ---------------------------------------------------------------------
    # 4. Bootstrap confidence interval (95 resamples) of the C-index
    # ---------------------------------------------------------------------
    bootstrap_scores = []

    rng = np.random.RandomState(42)

    indices = np.arange(len(y))

    for _ in range(N_BOOTSTRAP):

        sample = rng.choice(indices, size=len(indices), replace=True)
        score = concordance_index_censored(y["event"][sample], y["duration"][sample], risk_scores[sample])[0]
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)
    mean_score = np.mean(bootstrap_scores)
    std_score = np.std(bootstrap_scores)

    print(f"Bootstrap C-index: {mean_score:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")

    mlflow.log_metric("bootstrap_c_index_mean", mean_score)
    mlflow.log_metric("bootstrap_c_index_std", std_score)
    mlflow.log_metric("bootstrap_c_index_lower", lower)
    mlflow.log_metric("bootstrap_c_index_upper", upper)

    try:
    
        # ---------------------------------------------------------------------
        # 5. Risk-group stratification + log-rank test
        # ---------------------------------------------------------------------
        median_risk = np.median(risk_scores)
        risk_group = np.where(risk_scores >= median_risk, "high", "low")

        chisq, pvalue = compare_survival(y, risk_group)
        print(f"\nLog-rank test: chi2 = {chisq:.3f}, p-value = {pvalue:.4g}")
        mlflow.log_metric("logrank_chisq", chisq)
        mlflow.log_metric("logrank_pvalue", pvalue)

        # ---------------------------------------------------------------------
        # 6. Kaplan-Meier plot for the two risk groups
        # ---------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        for group_label in np.unique(risk_group):
            mask = risk_group == group_label
            time_g, survival_prob_g, conf_int = kaplan_meier_estimator(y[EVENT_FIELD][mask], y[TIME_FIELD][mask], conf_type="log-log")
            plt.step(time_g, survival_prob_g, where="post", label=f"{group_label} risk (n={mask.sum()})")
            plt.fill_between(time_g, conf_int[0], conf_int[1], alpha=0.2, step="post")

        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.title(f"Kaplan-Meier Curves by Risk Group\nLog-rank p = {pvalue:.4g}")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("kaplan_meier_plot.png", dpi=300)
        plt.close()
        mlflow.log_artifact("kaplan_meier_plot.png")

        #---------------------------------------------------------------------
        # 7. c-index and log-rank result in significant results
        #---------------------------------------------------------------------
        if pvalue < 0.05 and lower > 0.5:
            mlflow.log_param("significant_results", True)
        else:
            mlflow.log_param("significant_results", False)

    except:
        pass         
    
if __name__ == "__main__":

    for feature_set in [
                        'volume', 'clinical', 'radiomics', 'ctfm', 'omnislicer',
                        'volume_clinical', 'volume_radiomics', 'volume_ctfm', 'volume_omnislicer',
                        'clinical_radiomics', 'clinical_ctfm', 'clinical_omnislicer',
                        'volume_clinical_radiomics', 'volume_clinical_ctfm', 'volume_clinical_omnislicer'
                        ]:                              
                                

        for model_type in ['rsf', 'extra_trees', 'gradient_boosting', 'cox_elastic', 'svm', 'lipschitz_svm']:
            for histology in ['syn', 'all_syn', 'mfh', 'all_mfh', 'lipo', 'all_lipo']:
                
                print("\n###########################################################################")
                print(f"Model type: {model_type}, Histology: {histology}, Feature set: {feature_set}")
                print("###########################################################################\n")

                mlflow.set_experiment("DEGRO_journal_extension_new")
                mlflow.start_run()
                main(model_type=model_type, histology=histology, feature_set=feature_set)
                mlflow.end_run()
