# coding:utf-8

import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
from skcredit.feature_discretization import BaseDiscrete
from skcredit.feature_discretization.DiscreteImple import force_cat_table, force_num_table
from skcredit.feature_discretization.DiscreteImple import force_cat_table_cross, force_num_table_cross
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class DiscreteCust(BaseDiscrete):
    def __init__(self, tim_columns, group_dict, break_dict, group_dict_cross, break_dict_cross):
        super().__init__(tim_columns)

        self.group_dict = group_dict
        self.break_dict = break_dict
        self.group_dict_cross = group_dict_cross
        self.break_dict_cross = break_dict_cross

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        # feature
        with Pool(mp.cpu_count() - 2) as pool:
            if list(self.group_dict.keys()):
                self.cat_table_ = dict(zip(list(self.group_dict.keys()), pool.starmap(
                    force_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.group_dict[col]) for col in
                        list(self.group_dict.keys())])))
        self.cat_value_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.cat_value_ = OrderedDict(sorted(self.cat_value_.items(), key=lambda t: t[1], reverse=True))

        with Pool(mp.cpu_count() - 2) as pool:
            if list(self.break_dict.keys()):
                self.num_table_ = dict(zip(list(self.break_dict.keys()), pool.starmap(
                    force_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.break_dict[col]) for col in
                        list(self.break_dict.keys())])))
        self.num_value_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.num_value_ = OrderedDict(sorted(self.num_value_.items(), key=lambda t: t[1], reverse=True))

        self.information_value_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.information_value_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.information_value_ = OrderedDict(
            sorted(self.information_value_.items(), key=lambda t: t[1], reverse=True))

        # feature cross
        with Pool(mp.cpu_count() - 2) as pool:
            if list(self.group_dict_cross.keys()):
                self.cat_table_cross_ = dict(zip(list(self.group_dict_cross.keys()), pool.starmap(
                    force_cat_table_cross,
                    [(pd.concat([x[col.split(" @ ")], y.to_frame("target")], axis=1),
                      * col.split(" @ "),
                      * self.group_dict_cross[col].values())
                     for col in list(self.group_dict_cross.keys())])))
        self.cat_value_cross_.update({col: val["IV"].sum() for col, val in self.cat_table_cross_.items()})
        self.cat_value_cross_ = OrderedDict(sorted(self.cat_value_cross_.items(), key=lambda t: t[1], reverse=True))

        with Pool(mp.cpu_count() - 2) as pool:
            if list(self.break_dict_cross.keys()):
                self.num_table_cross_ = dict(zip(list(self.break_dict_cross.keys()), pool.starmap(
                    force_num_table_cross,
                    [(pd.concat([x[col.split(" @ ")], y.to_frame("target")], axis=1),
                      * col.split(" @ "),
                      * self.break_dict_cross[col].values())
                     for col in list(self.break_dict_cross.keys())])))
        self.num_value_cross_.update({col: val["IV"].sum() for col, val in self.num_table_cross_.items()})
        self.num_value_cross_ = OrderedDict(sorted(self.num_value_cross_.items(), key=lambda t: t[1], reverse=True))

        self.information_value_cross_.update(self.cat_value_cross_)
        self.information_value_cross_.update(self.num_value_cross_)
        self.information_value_cross_ = OrderedDict(
            sorted(self.information_value_cross_.items(), key=lambda t: t[1], reverse=True))

        return self
