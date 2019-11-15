# coding: utf-8

import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class PreProcess(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, cat_columns, num_columns):
        self.__keep_columns = keep_columns
        self.__cat_columns = cat_columns
        self.__num_columns = num_columns

        self.__cat_value = dict()
        self.__num_value = -9999.

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if len(self.__cat_columns) != 0:
            x[self.__cat_columns] = x[self.__cat_columns].fillna("missing").astype(str)

            for col in self.__cat_columns:
                self.__cat_value[col] = x[col].value_counts().to_dict()

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if len(self.__cat_columns) != 0:
            x[self.__cat_columns] = x[self.__cat_columns].fillna("missing").astype(str)

            for col in self.__cat_columns:
                to_rep = ("missing" if "missing" in self.__cat_value[col].keys()
                          else list(self.__cat_value[col].keys())[0])
                x[col] = x[col].apply(
                    lambda element: element if element in self.__cat_value[col].keys() else to_rep)

        if len(self.__num_columns) != 0:
            x[self.__num_columns] = x[self.__num_columns].fillna(self.__num_value)

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)