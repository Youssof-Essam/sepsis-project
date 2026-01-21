import DataWrangling
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    data = DataWrangling.load_data(f"{PROJECT_ROOT}/data/training_data")
    data_fixed = dict()
    data_fixed["measurement_meds"] = data["measurement_meds"].copy()
    data_fixed["measurement_meds"]["Body temperature"] = data_fixed["measurement_meds"]["Body temperature"].apply(DataWrangling.temperature_fixer)
    data_fixed["measurement_meds"]["Heart rate"] = data_fixed["measurement_meds"]["Heart rate"].apply(DataWrangling.heart_rate_fixer)
    data_fixed["measurement_meds"]["Respiratory rate"] = data_fixed["measurement_meds"]["Respiratory rate"].apply(DataWrangling.respiration_fixer)
    data_fixed["measurement_meds"]["Systolic blood pressure"] = data_fixed["measurement_meds"]["Systolic blood pressure"].apply(DataWrangling.sys_bp_fixer)
    data_fixed["measurement_meds"]["Diastolic blood pressure"] = data_fixed["measurement_meds"]["Diastolic blood pressure"].apply(DataWrangling.dias_bp_fixer)
    data_fixed["measurement_meds"]["Measurement of oxygen saturation at periphery"] = data_fixed["measurement_meds"]["Measurement of oxygen saturation at periphery"].apply(DataWrangling.oxy_sat_fixer)

    data_fixed["measurement_lab"] = data["measurement_lab"].copy()

    lab_extremes_sepsis_tolerant = {
        "Base excess in Venous blood by calculation": {"unit": "mmol/L", "min": -25, "max": 25},
        "Base excess in Arterial blood by calculation": {"unit": "mmol/L", "min": -25, "max": 25},
        "Phosphate [Moles/volume] in Serum or Plasma": {"unit": "mmol/L", "min": 0.4, "max": 6},
        "Potassium [Moles/volume] in Blood": {"unit": "mmol/L", "min": 2.8, "max": 9},
        "Bilirubin.total [Moles/volume] in Serum or Plasma": {"unit": "µmol/L", "min": 0, "max": 700},
        "Neutrophil Ab [Units/volume] in Serum": {"unit": "×10^9/L", "min": 0.2, "max": 100},
        "Bicarbonate [Moles/volume] in Arterial blood": {"unit": "mmol/L", "min": 7, "max": 45},
        "Hematocrit [Volume Fraction] of Blood": {"unit": "L/L", "min": 0.22, "max": 0.70},
        "Glucose [Moles/volume] in Serum or Plasma": {"unit": "mmol/L", "min": 2, "max": 45},
        "Calcium [Moles/volume] in Serum or Plasma": {"unit": "mmol/L", "min": 1.4, "max": 3.2},
        "Chloride [Moles/volume] in Blood": {"unit": "mmol/L", "min": 80, "max": 125},
        "Sodium [Moles/volume] in Serum or Plasma": {"unit": "mmol/L", "min": 115, "max": 165},
        "C reactive protein [Mass/volume] in Serum or Plasma": {"unit": "mg/L", "min": 0, "max": 400},
        "Carbon dioxide [Partial pressure] in Venous blood": {"unit": "mmHg", "min": 12, "max": 65},
        "Oxygen [Partial pressure] in Venous blood": {"unit": "mmHg", "min": 18, "max": 55},
        "Albumin [Mass/volume] in Serum or Plasma": {"unit": "g/L", "min": 12, "max": 60},
        "Bicarbonate [Moles/volume] in Venous blood": {"unit": "mmol/L", "min": 7, "max": 45},
        "Oxygen [Partial pressure] in Arterial blood": {"unit": "mmHg", "min": 45, "max": 160},
        "Carbon dioxide [Partial pressure] in Arterial blood": {"unit": "mmHg", "min": 18, "max": 65},
        "Interleukin 6 [Mass/volume] in Body fluid": {"unit": "pg/mL", "min": 0, "max": 6000},
        "Magnesium [Moles/volume] in Blood": {"unit": "mmol/L", "min": 0.5, "max": 3},
        "Prothrombin time (PT)": {"unit": "sec", "min": 9, "max": 50},
        "Procalcitonin [Mass/volume] in Serum or Plasma": {"unit": "ng/mL", "min": 0, "max": 150},
        "Lactate [Moles/volume] in Blood": {"unit": "mmol/L", "min": 0.5, "max": 20},
        "Creatinine [Mass/volume] in Blood": {"unit": "µmol/L", "min": 40, "max": 900},
        "Fibrinogen measurement": {"unit": "g/L", "min": 1, "max": 7},
        "Bilirubin measurement": {"unit": "µmol/L", "min": 0, "max": 700},
        "Partial thromboplastin time": {"unit": "sec", "min": 25, "max": 120},
        "activated": {"unit": "", "min": 0, "max": 1},  # Binary flag
        "Total white blood count": {"unit": "×10^9/L", "min": 1, "max": 60},
        "Platelet count": {"unit": "×10^9/L", "min": 20, "max": 1200},
        "White blood cell count": {"unit": "×10^9/L", "min": 1, "max": 60},
        "Blood venous pH": {"unit": "pH", "min": 6.85, "max": 7.65},
        "D-dimer level": {"unit": "µg/mL FEU", "min": 0, "max": 30},
        "Blood arterial pH": {"unit": "pH", "min": 6.85, "max": 10},
        "Hemoglobin [Moles/volume] in Blood": {"unit": "mmol/L", "min": 5, "max": 22},
        "Ionised calcium measurement": {"unit": "mmol/L", "min": 1.0, "max": 2.2}
    }

    for column in data_fixed["measurement_lab"].drop(["visit_occurrence_id", "person_id", "measurement_datetime", "index"], axis=1).columns:
        try:
            data_fixed["measurement_lab"][column] = data_fixed["measurement_lab"][column].apply(
                lambda x: DataWrangling.value_fixer(x, lab_extremes_sepsis_tolerant[column]["max"]))
        except:
            continue

    data_imputed = dict()

    for dataframe in data_fixed.keys():
        data_imputed[dataframe] = data_fixed[dataframe].copy()
        for column in data_fixed[dataframe].columns:
            data_imputed[dataframe] = DataWrangling.last_observed_imputer(data_imputed[dataframe], column)


    for dataframe in data_imputed.keys():
        data_imputed[dataframe].to_csv(f"{PROJECT_ROOT}/data/cleaned_training/{dataframe}.csv")
