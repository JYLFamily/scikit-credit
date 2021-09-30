# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.SplitCat import replace_cat
from skcredit.feature_discretization.SplitNum import replace_num
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class Discrete(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, date_columns):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.cat_spliter = None
        self.num_spliter = None

        self.information_value_score = None
        self.information_value_table = None

    def fit(self, x,  y=None):
        pass

        return self

    def transform(self, x):
        z = pd.DataFrame().reindex_like(x[self.keep_columns +
                                          self.date_columns +
                                          list(self.num_spliter.keys()) +
                                          list(self.cat_spliter.keys())])
        z[self.keep_columns] = x[self.keep_columns]
        z[self.date_columns] = x[self.date_columns]

        if self.cat_spliter:
            z[  list(self.cat_spliter.keys())] = pd.DataFrame(dict(zip(
                list(self.cat_spliter.keys()),
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(replace_cat)(x[col], sc) for col, sc in self.cat_spliter.items()]))
            ))

        if self.num_spliter:
            z[  list(self.num_spliter.keys())] = pd.DataFrame(dict(zip(
                list(self.num_spliter.keys()),
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(replace_num)(x[col], sn) for col, sn in self.num_spliter.items()]))
            ))

        return z

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)

    def show_order(self):
        return self.information_value_score

    def show_table(self):
        return self.information_value_table
