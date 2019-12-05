# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from operator import add
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectVif(BaseEstimator, TransformerMixin):
    def __init__(self, *, keep_columns, rs_threshold=0.8):
        self.keep_columns = keep_columns
        self.rs_threshold = rs_threshold

        self.feature_columns_ = None
        self.feature_support_ = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array([col for col in x.columns if col not in self.keep_columns])
        self.feature_support_ = np.ones(len(self.feature_columns_), dtype=bool)

        for i in range(len(self.feature_columns_)):
            for j in range(add(i, 1), len(self.feature_columns_)):
                if self.feature_support_[j]:
                    model = LinearRegression()
                    model.fit(x.iloc[:, [i]], x.iloc[:, j])
                    rs = model.score(x.iloc[:, [i]], x.iloc[:, j])

                    if rs >= self.rs_threshold:
                        self.feature_support_[j] = False
                        logging.info(self.feature_columns_[j] + " remove !")

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return x[self.keep_columns + self.feature_columns_[self.feature_support_].tolist()]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)