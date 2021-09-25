# coding:utf-8

import numpy  as np
import pandas as pd
from collections import ChainMap
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.SplitCat import binning_cat, replace_cat
from skcredit.feature_discretization.SplitNum import binning_num, replace_num
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class Discrete(BaseEstimator, TransformerMixin):
    def __init__(self,  keep_columns, date_columns):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.cat_columns_ = list()
        self.num_columns_ = list()

        self.cat_spliter_ = dict()
        self.num_spliter_ = dict()

        self.information_value_score = None
        self.information_value_table = None

    def fit(self, x,  y=None):
        self.cat_columns_ = [col for col in x.select_dtypes(include="object").columns
            if col not in self.keep_columns and col not in self.date_columns]
        self.num_columns_ = [col for col in x.select_dtypes(exclude="object").columns
            if col not in self.keep_columns and col not in self.date_columns]

        if self.cat_columns_:
            self.cat_spliter_ = (dict(zip(
                self.cat_columns_,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(binning_cat)(x[column], y) for column in self.cat_columns_]))
            ))

        if self.num_columns_:
            self.num_spliter_ = (dict(zip(
                self.num_columns_,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(binning_num)(x[column], y) for column in self.num_columns_]))
            ))

        temp = ChainMap(
            {column: spliter.table for column, spliter in self.cat_spliter_.items()},
            {column: spliter.table for column, spliter in self.num_spliter_.items()}
        )
        temp = dict(sorted(temp.items(), key=lambda item: item[1]["IvS"].sum(), reverse=True))

        self.information_value_score = pd.DataFrame.from_dict(
            {column: table["IvS"].sum() for column, table in temp.items()}, orient="index", columns=["IvS"])
        self.information_value_table = pd.concat(temp.values())

        return self

    def transform(self, x):
        z = pd.DataFrame().reindex_like(x)
        z[self.keep_columns] = x[self.keep_columns]
        z[self.date_columns] = x[self.date_columns]

        if self.cat_columns_:
            z[self.cat_columns_] = pd.DataFrame(dict(zip(
                self.cat_columns_,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(replace_cat)(x[col], sc) for col, sc in self.cat_spliter_.items()]))
            ))

        if self.num_columns_:
            z[self.num_columns_] = pd.DataFrame(dict(zip(
                self.num_columns_,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(replace_num)(x[col], sn) for col, sn in self.num_spliter_.items()]))
            ))

        return z

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)

    def show_order(self):
        return self.information_value_score

    def show_table(self):
        return self.information_value_table
