# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class BaseSelect(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, date_columns, nums_columns=None):

        self.keep_columns = keep_columns
        self.date_columns = date_columns
        self.nums_columns = nums_columns

        self.feature_columns = None
        self.feature_support = None

    def fit(self, x, y=None):
        pass

    def transform(self,   x):

        return x[self.keep_columns +
                 self.date_columns +
                 self.feature_columns[self.feature_support].tolist()]

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
