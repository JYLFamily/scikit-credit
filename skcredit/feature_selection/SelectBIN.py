# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectBIN(BaseEstimator, TransformerMixin):
    def __init__(self, *, keep_columns):
        self.__keep_columns = keep_columns
        self.__feature_columns = None
        self.__feature_support = None

    def fit(self, X, y):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.__feature_columns = np.array([col for col in x.columns if col not in self.__keep_columns])
        self.__feature_support = np.ones(len(self.__feature_columns), dtype=bool)

        beta_0 = np.log(y.sum() / (y.shape[0] - y.sum()))
        beta_1 = 1
        for idx, col in enumerate(self.__feature_columns):
            logit_mod = sm.Logit(y, sm.add_constant(x[[col]], prepend=False))
            logit_res = logit_mod.fit(disp=0)  # disp=0 slience

            if (abs(logit_res.params["const"] - beta_0) > 0.00001 and
                    abs(logit_res.params[col] - beta_1) > 0.00001):
                self.__feature_support[idx] = False
                logging.info(col + " remove SELECTBIN !")

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return x[self.__keep_columns + self.__feature_columns[self.__feature_support].tolist()]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)