# coding: utf-8

import os
import gc
import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class BaseSelect(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, time_columns):
        self.keep_columns = keep_columns
        self.time_columns = time_columns

        self.feature_columns_ = None
        self.feature_support_ = None

    def fit(self, X,  y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return x[self.keep_columns +
                 self.time_columns +
                 self.feature_columns_[self.feature_support_].tolist()]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)
