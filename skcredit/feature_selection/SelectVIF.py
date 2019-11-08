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


class SelectVIF(BaseEstimator, TransformerMixin):
    def __init__(self, *, keep_columns, vif_threshold):
        self.__keep_columns, self.__vif_threshold = keep_columns, vif_threshold
        self.__feature_columns = None
        self.__feature_support = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.__feature_columns = np.array([col for col in x.columns if col not in self.__keep_columns])
        self.__feature_support = np.ones(len(self.__feature_columns), dtype=bool)

        num_feature = len(self.__feature_columns)
        for p in range(num_feature):
            for q in range(add(p, 1), num_feature):
                if self.__feature_support[q]:
                    lm = LinearRegression()
                    lm.fit(x.iloc[:, [p]], x.iloc[:, q])
                    vif = 1 / (1 - lm.score(x.iloc[:, [p]], x.iloc[:, q]))

                    if vif >= self.__vif_threshold:
                        self.__feature_support[q] = False
                        logging.info(self.__feature_columns[q] + " remove SELECTVIF !")

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return x[self.__keep_columns + self.__feature_columns[self.__feature_support].tolist()]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)