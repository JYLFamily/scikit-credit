# coding:utf-8

import pandas as pd
import numpy  as np
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)


class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, target):
        self.columns = columns
        self.target  = target
        self.lookup = dict()

    def fit(self,    x, y):
        for column in self.columns:
            self.lookup[column] = y.groupby(x[column]).agg(lambda group:
                    round(np.log((0.0005 if (temp := group.eq(1).sum()) == 0 else temp) /
                                 (0.0005 if (temp := group.eq(0).sum()) == 0 else temp)), 5)).to_dict()

        return self

    def transform(self, x):
        x_transformed  = x.copy(deep=True)

        for column in self.columns:
            x_transformed.loc[~x_transformed[column].isin(self.lookup[column].keys()), column] = (
                max(self.lookup[column], key=self.lookup[column].get))
            x_transformed[column] = x[column].map(self.lookup[column])

        return x_transformed

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
