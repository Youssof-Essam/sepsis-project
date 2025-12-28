import DataWrangling
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    data = DataWrangling.load_data(f"{PROJECT_ROOT}/data/training_data")

    data_fixed = data["measurement_meds"].copy()
    data_fixed["Body temperature"] = data_fixed["Body temperature"].apply(DataWrangling.temperature_fixer)
    data_fixed["Heart rate"] = data_fixed["Heart rate"].apply(DataWrangling.heart_rate_fixer)
    data_fixed["Respiratory rate"] = data_fixed["Respiratory rate"].apply(DataWrangling.respiration_fixer)
    data_fixed["Systolic blood pressure"] = data_fixed["Systolic blood pressure"].apply(DataWrangling.sys_bp_fixer)
    data_fixed["Diastolic blood pressure"] = data_fixed["Diastolic blood pressure"].apply(DataWrangling.dias_bp_fixer)
    data_fixed["Measurement of oxygen saturation at periphery"] = data_fixed["Measurement of oxygen saturation at periphery"].apply(DataWrangling.oxy_sat_fixer)

    data_imputed = data_fixed.copy()
    data_imputed.drop("Oxygen/Gas total [Pure volume fraction] Inhaled gas", axis=1, inplace=True)

    for column in data_imputed.columns:
        data_imputed = DataWrangling.last_observed_imputer(data_imputed, column)

    data_imputed.to_csv(f"{PROJECT_ROOT}/data/cleaned_training/measurement_meds_train_cleaned.csv")