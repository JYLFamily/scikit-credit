# coding: utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class CTabular(BaseEstimator, TransformerMixin):
    def __init__(self, rule):
        self.rule = rule

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        for k, v in self.rule.items():
            for i in v:
                x[k] = x[k].replace({int(i): np.nan})

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)


