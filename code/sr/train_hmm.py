from operator import index

import joblib

import DataWrangling
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from preprocessing import Last_Observed_Imputer
from joblib import dump

def train_hmm(data : pd.DataFrame,feature_cols:list,n_components = 3, iterations = 200 ):

    df = data.sort_values(by=["person_id","measurement_datetime"])
    lengths = df.groupby("person_id").size().tolist()
    arr = df[feature_cols].to_numpy()

    hmm = GaussianHMM(n_components=n_components,covariance_type="diag",n_iter=iterations)
    hmm.fit(arr,lengths=lengths)

    return hmm,df

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


    # load and transform data
    print("loading and transforming data...")
    measurement_meds = pd.read_csv(f"{PROJECT_ROOT}/data/cleaned_training/measurement_meds.csv")
    measurement_lab = pd.read_csv(f"{PROJECT_ROOT}/data/cleaned_training/measurement_lab.csv")
    sepsis_labels = pd.read_csv(f"{PROJECT_ROOT}/data/training_data/SepsisLabel_train.csv")

    data = sepsis_labels.merge(measurement_meds,on=["person_id","measurement_datetime"],how="left").reset_index(drop=True)
    data = data.merge(measurement_lab,on=["person_id","measurement_datetime","visit_occurrence_id"],how="left").reset_index(drop=True)
    data = data.drop(["visit_occurrence_id","Ionised calcium measurement"],axis=1)

    # Preprocess and prepare data
    print("preprocessing data....")
    features = data.drop(["person_id", "measurement_datetime", "SepsisLabel"], axis=1).columns
    last_observed_imputer = Last_Observed_Imputer(columns=features)
    pipeline = Pipeline([
        ("median_imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformer = ColumnTransformer([
        ("num", pipeline, features),
    ], remainder="passthrough", verbose_feature_names_out=False)
    data_imputed_scaled = last_observed_imputer.fit_transform(data)
    data_imputed_scaled = pd.DataFrame(transformer.fit_transform(data_imputed_scaled),columns=transformer.get_feature_names_out())

    # 1st model, on full data
    print("Fitting model 1....")
    model_1,data_1 = train_hmm(data_imputed_scaled,feature_cols=features,n_components=4)

    # 2nd model, only sepsis patients
    print("Fitting model 2....")
    patients = DataWrangling.get_sepsis_labels(f"{PROJECT_ROOT}/data/training_data/SepsisLabel_train.csv")
    septic_patients = patients[patients["SepsisLabel"] == 1].reset_index(drop=True)

    data_only_septic = data_imputed_scaled[data_imputed_scaled["person_id"].isin(septic_patients["person_id"].tolist())].reset_index(drop=True)
    model_2,data_2 = train_hmm(data_only_septic,feature_cols=features,n_components=4)

    # 3rd model, balanced number of patients
    print("Fitting model 3....")
    non_septic_sample = patients[patients["SepsisLabel"] == 0].reset_index(drop=True).sample(n = 78, random_state = 42)
    balanced_patients = pd.concat([septic_patients,non_septic_sample],ignore_index=True)
    balanced_data = data_imputed_scaled[data_imputed_scaled["person_id"].isin(balanced_patients["person_id"].tolist())].reset_index(drop=True)

    model_3,data_3 = train_hmm(balanced_data,feature_cols=features,n_components=4)

    #saving models
    print("savings models....")
    joblib.dump(model_1,f"{PROJECT_ROOT}/models/hmms/hmm_all.pkl")
    data_1.to_csv(f"{PROJECT_ROOT}/models/hmms/hmm_all.csv",index= False)
    joblib.dump(model_2,f"{PROJECT_ROOT}/models/hmms/hmm_septic.pkl")
    data_2.to_csv(f"{PROJECT_ROOT}/models/hmms/hmm_septic.csv",index = False)
    joblib.dump(model_3,f"{PROJECT_ROOT}/models/hmms/hmm_balanced.pkl")
    data_3.to_csv(f"{PROJECT_ROOT}/models/hmms/hmm_balanced.csv",index = False)
    print("Done!")


