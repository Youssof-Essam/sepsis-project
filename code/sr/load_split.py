import DataWrangling
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    labels = DataWrangling.get_sepsis_labels(f"{PROJECT_ROOT}/data/raw_data/SepsisLabel_train.csv")
    train_labels , test_labels = DataWrangling.split_stratified(labels,0.2,42)

    data = DataWrangling.load_data(f"{PROJECT_ROOT}/data/raw_data")

    train_dfs = dict()
    test_dfs = dict()

    # getting training and test rows
    for dataframe in data.keys():
        train_dfs[dataframe] = data[dataframe][data[dataframe]["person_id"].isin(train_labels["person_id"].tolist())].copy()
        test_dfs[dataframe] = data[dataframe][data[dataframe]["person_id"].isin(test_labels["person_id"].tolist())].copy()

    # saving training data
    for dataframe in train_dfs.keys():
        train_dfs[dataframe].drop("index", axis = 1).to_csv(f"{PROJECT_ROOT}/data/training_data/{dataframe}_train.csv", index=False)

    # saving testing data
    for dataframe in test_dfs.keys():
        test_dfs[dataframe].drop("index", axis = 1).to_csv(f"{PROJECT_ROOT}/data/testing_data/{dataframe}_test.csv", index=False)

