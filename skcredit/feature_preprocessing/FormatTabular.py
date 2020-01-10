# coding: utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class FormatTabular(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, cat_columns, num_columns):
        self.keep_columns = keep_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns

        self.cat_value_ = dict()
        self.num_value_ = dict()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if self.cat_columns:
            x[self.cat_columns] = x[self.cat_columns].fillna("missing").astype(str)

            for col in self.cat_columns:
                stats = x[col].value_counts(normalize=True)
                if 1 - stats.max() >= 0.05:
                    self.cat_value_[col] = stats.to_dict()

        if self.num_columns:
            x[self.num_columns] = x[self.num_columns].fillna(- 9999.00)

            for col in self.num_columns:
                stats = x[col].value_counts(normalize=True)
                if 1 - stats.max() >= 0.05:
                    self.num_value_[col] = -9999.

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        x = x[self.keep_columns + list(self.cat_value_.keys()) + list(self.num_value_.keys())]

        if self.cat_value_.keys():
            x[list(self.cat_value_.keys())] = x[list(self.cat_value_.keys())].fillna("missing").astype(str)

            for col in self.cat_value_.keys():
                to_rep = ("missing" if "missing" in self.cat_value_[col].keys()
                          else list(self.cat_value_[col].keys())[0])
                x[col] = x[col].apply(
                    lambda element: element if element in self.cat_value_[col].keys() else to_rep)

        if self.num_value_.keys():
            for col in self.num_value_.keys():
                x[col] = x[col].fillna(self.num_value_[col])

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)