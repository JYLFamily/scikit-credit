# coding: utf-8

import gc
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class SaveMemory(object):
    def __init__(self, keep_columns, cat_columns, num_columns):
        self.keep_columns = keep_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns

        self.num_dtype_ = dict()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        for col in self.num_columns:
            col_min = x[col].min()
            col_max = x[col].max()
            col_type = x[col].dtype

            if str(col_type)[:3] == "int":
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    self.num_dtype_[col] = np.int8
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    self.num_dtype_[col] = np.int16
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    self.num_dtype_[col] = np.int32
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    self.num_dtype_[col] = np.int64
            else:
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    self.num_dtype_[col] = np.float32
                elif col_min > np.finfo(np.float64).min and col_max < np.finfo(np.float64).max:
                    self.num_dtype_[col] = np.float64

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        for col in self.num_columns:
            x[col] = x[col].astype(self.num_dtype_[col])
        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)