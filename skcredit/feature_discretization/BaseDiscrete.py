# coding: utf-8

import os
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.DiscreteImple import replace_cat_woe, replace_num_woe
from skcredit.feature_discretization.DiscreteImple import replace_cat_woe_cross, replace_num_woe_cross

np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class BaseDiscrete(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns

        self.cat_table_ = dict()
        self.num_table_ = dict()
        self.cat_table_cross_ = dict()
        self.num_table_cross_ = dict()

        self.cat_value_ = OrderedDict()
        self.num_value_ = OrderedDict()
        self.cat_value_cross_ = OrderedDict()
        self.num_value_cross_ = OrderedDict()

        self.information_value_ = OrderedDict()
        self.information_value_cross_ = OrderedDict()

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        z = pd.DataFrame(
            data=np.zeros(shape=(
                x.shape[0], len(self.information_value_) + len(self.information_value_cross_))),
            columns=list(self.information_value_.keys()) + list(self.information_value_cross_.keys()))

        f = pd.Series(self.information_value_).append(
            pd.Series(self.information_value_cross_)).sort_values(ascending=False).index.tolist()

        # feature
        if self.cat_table_.keys():
            with Pool(mp.cpu_count() - 2) as pool:
                z[list(self.cat_table_.keys())] = pd.DataFrame(
                    dict(zip(self.cat_table_.keys(), pool.starmap(
                        replace_cat_woe,
                        [(x[[col]], table[col], table["WoE"]) for col, table in self.cat_table_.items()]))))

        if self.num_table_.keys():
            with Pool(mp.cpu_count() - 2) as pool:
                z[list(self.num_table_.keys())] = pd.DataFrame(
                    dict(zip(self.num_table_.keys(), pool.starmap(
                        replace_num_woe,
                        [(x[[col]], table[col], table["WoE"]) for col, table in self.num_table_.items()]))))

        # feature cross
        if self.cat_table_cross_.keys():
            with Pool(mp.cpu_count() - 2) as pool:
                z[list(self.cat_table_cross_.keys())] = pd.DataFrame(
                    dict(zip(self.cat_table_cross_.keys(), pool.starmap(
                        replace_cat_woe_cross,
                        [(x[col.split(" @ ")],
                          * col.split(" @ "),
                          table[col.split(" @ ")[0]],
                          table[col.split(" @ ")[1]],
                          table["WoE"])
                         for col, table in self.cat_table_cross_.items()]))))

        if self.num_table_cross_.keys():
            with Pool(mp.cpu_count() - 2) as pool:
                z[list(self.num_table_cross_.keys())] = pd.DataFrame(
                    dict(zip(self.num_table_cross_.keys(), pool.starmap(
                        replace_cat_woe_cross,
                        [(x[col.split(" @ ")],
                          * col.split(" @ "),
                          table[col.split(" @ ")[0]],
                          table[col.split(" @ ")[1]],
                          table["WoE"])
                         for col, table in self.num_table_cross_.items()]))))

        return pd.concat([x[self.tim_columns], z.reindex(columns=f)], axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)

    def save_order(self, path):
        order = pd.DataFrame.from_dict(self.information_value_, orient="index", columns=["IV"])
        order = order.reset_index().rename(columns={"index": "feature"})

        with pd.ExcelWriter(os.path.join(path, "order.xlsx")) as writer:
            order.to_excel(writer, index=False)

    def save_table(self, path):
        table = dict()
        table.update(self.num_table_)
        table.update(self.cat_table_)

        with pd.ExcelWriter(os.path.join(path, "table.xlsx")) as writer:
            for feature, table in table.items():
                table.to_excel(writer, sheet_name=feature[-30:], index=False)

    def save_order_cross(self, path):
        order = pd.DataFrame.from_dict(self.information_value_cross_, orient="index", columns=["IV"])
        order = order.reset_index().rename(columns={"index": "feature"})

        with pd.ExcelWriter(os.path.join(path, "order_cross.xlsx")) as writer:
            order.to_excel(writer, index=False)

    def save_table_cross(self, path):
        table = dict()
        table.update(self.num_table_cross_)
        table.update(self.cat_table_cross_)

        with pd.ExcelWriter(os.path.join(path, "table_cross.xlsx")) as writer:
            for feature, table in table.items():
                table.to_excel(writer, sheet_name=feature[-30:], index=False)
