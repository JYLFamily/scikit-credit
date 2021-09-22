# coding: utf-8

import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class FormatTabular(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, date_columns, cat_columns, num_columns):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.cat_columns = cat_columns
        self.num_columns = num_columns

    def fit(self, x, y=None):
        pass

    def transform(self, x):
        z = pd.DataFrame().reindex_like(x)
        z[self.cat_columns] = x[self.cat_columns].fillna("missing").astype(str)
        z[self.num_columns] = x[self.num_columns].fillna(-999999.0)

        return z

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)


