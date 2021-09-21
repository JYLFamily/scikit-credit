# coding: utf-8

import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class Select(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, date_columns):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.feature_columns_ = None
        self.feature_support_ = None

    def fit(self, x, y=None):
        pass

    def transform(self,   x):

        return x[self.keep_columns +
                 self.date_columns +
                 self.feature_columns_[self.feature_support_].tolist()]

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
