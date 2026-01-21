import os
import pandas as pd

def load_data(path:str):
    files = os.listdir(path)
    data_frames = [pd.read_csv(f"{path}/{dataset}") for dataset in files ]

    data_dict =  dict(zip([file.replace("_train.csv","") for file in files], data_frames))

    def parse_date(df:pd.DataFrame,column:str):
        df[column] = pd.to_datetime(df[column])
        df = df.sort_values(by = ["person_id",column]).reset_index(drop = False)
        return df
    data_dict["devices"] = parse_date(data_dict["devices"],"device_datetime_hourly")
    data_dict["drugsexposure"] = parse_date(data_dict["drugsexposure"],"drug_datetime_hourly")
    data_dict["measurement_lab"] = parse_date(data_dict["measurement_lab"],"measurement_datetime")
    data_dict["measurement_meds"] = parse_date(data_dict["measurement_meds"],"measurement_datetime")
    data_dict["measurement_observation"] = parse_date(data_dict["measurement_observation"],"measurement_datetime")
    data_dict["observation"] = parse_date(data_dict["observation"],"observation_datetime")
    data_dict["proceduresoccurrences"] = parse_date(data_dict["proceduresoccurrences"],"procedure_datetime_hourly")
    data_dict["SepsisLabel"] = parse_date(data_dict["SepsisLabel"],"measurement_datetime")
    data_dict["person_demographics_episode"] = data_dict["person_demographics_episode"].sort_values(by = ["person_id"]).reset_index(drop = False)

    return data_dict

def last_observed_imputer(df:pd.DataFrame,column:str):
    person_ids = df["person_id"].unique()
    result = list()
    for person_id in person_ids:
        person_df = df[df["person_id"] == person_id].copy()
        person_df[column] = person_df[column].bfill()
        person_df[column] = person_df[column].ffill()

        result.append(person_df)

    return pd.concat(result)

def temperature_fixer(temp):
    new_temp = temp
    while new_temp >=100:
        new_temp /= 10
    if new_temp > 44:
        return None
    if new_temp < 25:
        return None
    return new_temp

def heart_rate_fixer(heart_rate):
    new_heart_rate = heart_rate
    while new_heart_rate >250:
        new_heart_rate /= 10
    if new_heart_rate <30:
        return None
    return new_heart_rate

def respiration_fixer(respiration):
    new_respiration = respiration
    while new_respiration >85:
        new_respiration /= 10
    if new_respiration < 5:
        return None
    return new_respiration
def sys_bp_fixer(sys_bp):
    new_sys_bp = sys_bp
    while new_sys_bp >200:
        new_sys_bp /= 10
    if new_sys_bp < 20:
        return None
    return new_sys_bp
def dias_bp_fixer(dias_bp):
    new_dias_bp = dias_bp
    while new_dias_bp >120:
        new_dias_bp /= 10
    if new_dias_bp < 5:
        return None
    return new_dias_bp
def oxy_sat_fixer(oxy_sat):
    new_oxy_sat = oxy_sat
    while new_oxy_sat >170:
        new_oxy_sat /= 10
    if new_oxy_sat < 20:
        return None
    return new_oxy_sat
def phosphate_fixer(phosphate):
    new_phosphate = phosphate
    while new_phosphate >3:
        new_phosphate /= 10
    return new_phosphate
def potassium_fixer(potassium):
    while potassium >6:
        potassium /= 10

def value_fixer(value,multiple_limit,abs_max = None,abs_min = None):
    new_value = value
    while new_value >multiple_limit:
        new_value /= 10

    if abs_max:
        if new_value > abs_max:
            return None
    if abs_min:
        if new_value < abs_min:
            return None
    return new_value