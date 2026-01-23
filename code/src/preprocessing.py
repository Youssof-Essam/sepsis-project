from sklearn.base import BaseEstimator, TransformerMixin
from DataWrangling import last_observed_imputer
import pandas as pd

class Last_Observed_Imputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns:list):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        result  = X.copy()
        for col in self.columns:
            result= last_observed_imputer(result,col)
        return result