# coding:utf-8

import warnings
import pandas as pd
import numpy  as np
from copy import deepcopy
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_columns = None
        self.num_columns = None

        # cat
        self.column_cat_lookup = dict()
        self.column_woe_lookup = dict()

    def fit(self,    x, y):
        self.cat_columns = x.select_dtypes(include="category").columns.tolist()
        self.num_columns = x.select_dtypes(exclude="category").columns.tolist()

        for column in self.cat_columns:
            self.column_cat_lookup[column] = CategoricalDtype(categories=np.unique(x[column]).tolist())

            self.column_woe_lookup[column] = y.groupby(x[column]).agg(lambda group:
                    round(np.log((0.0005 if (temp := group.eq(1).sum()) == 0 else temp) /
                                 (0.0005 if (temp := group.eq(0).sum()) == 0 else temp)), 5)).to_dict()

        return self

    def transform(self, x):
        x_transformed = deepcopy(x)

        for column in self.cat_columns:
            # not in train replace nan
            x_transformed[column] = x_transformed[column].astype(self.column_cat_lookup[column])
            x_transformed[column] = x_transformed[column].apply( self.column_woe_lookup[column])

        return x_transformed

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
