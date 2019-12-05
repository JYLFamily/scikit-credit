# coding: utf-8

import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class TidyTabula(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, cat_columns, num_columns):
        self.keep_columns = keep_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns

        self.cat_value_ = dict()
        self.num_value_ = -9999.

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if len(self.cat_columns) != 0:
            x[self.cat_columns] = x[self.cat_columns].fillna("missing").astype(str)

            for col in self.cat_columns:
                self.cat_value_[col] = x[col].value_counts().to_dict()

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if len(self.cat_columns) != 0:
            x[self.cat_columns] = x[self.cat_columns].fillna("missing").astype(str)

            for col in self.cat_columns:
                to_rep = ("missing" if "missing" in self.cat_value_[col].keys()
                          else list(self.cat_value_[col].keys())[0])
                x[col] = x[col].apply(
                    lambda element: element if element in self.cat_value_[col].keys() else to_rep)

        if len(self.num_columns) != 0:
            x[self.num_columns] = x[self.num_columns].fillna(self.num_value_)

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)