# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class Select(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_columns = None
        self.feature_support = None

    def fit(self, x, y=None):
        pass

    def transform(self,   x):

        return x[self.feature_columns[self.feature_support].tolist()]

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
