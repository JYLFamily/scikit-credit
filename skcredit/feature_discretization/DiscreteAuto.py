# coding:utf-8

import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from itertools import combinations
from collections import OrderedDict
from skcredit.feature_discretization.BaseDiscrete import BaseDiscrete
from skcredit.feature_discretization.DiscreteImple import merge_cat_table, merge_num_table
from skcredit.feature_discretization.DiscreteImple import merge_cat_table_cross, merge_num_table_cross
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class DiscreteAuto(BaseDiscrete):
    def __init__(self,   tim_columns):
        super().__init__(tim_columns)
        self.cat_columns_ = None
        self.num_columns_ = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.cat_columns_ = [col for col in x.select_dtypes(include="object").columns if col not in self.tim_columns]
        self.num_columns_ = [col for col in x.select_dtypes(exclude="object").columns if col not in self.tim_columns]

        # feature
        with Pool(mp.cpu_count() - 1) as pool:
            if self.cat_columns_:
                self.cat_table_ = dict(zip(self.cat_columns_, pool.starmap(
                    merge_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col) for col in self.cat_columns_])))
        self.cat_table_ = {
            col: val for col, val in self.cat_table_.items() if val["IV"].sum() >= 0}
        self.cat_value_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.cat_value_ = OrderedDict(sorted(self.cat_value_.items(), key=lambda t: t[1], reverse=True))

        with Pool(mp.cpu_count() - 1) as pool:
            if self.num_columns_:
                self.num_table_ = dict(zip(self.num_columns_, pool.starmap(
                    merge_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col) for col in self.num_columns_])))
        self.num_table_ = {
            col: val for col, val in self.num_table_.items() if val["IV"].sum() >= 0}
        self.num_value_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.num_value_ = OrderedDict(sorted(self.num_value_.items(), key=lambda t: t[1], reverse=True))

        self.information_value_.update(self.cat_value_)
        self.information_value_.update(self.num_value_)
        self.information_value_ = OrderedDict(
            sorted(self.information_value_.items(), key=lambda t: t[1], reverse=True))

        # feature cross
        # with Pool(mp.cpu_count() - 1) as pool:
        #     if len(self.cat_value_.keys()) >= 2:
        #         self.cat_table_cross_ = dict(zip(
        #             ["{} @ {}".format(col_1, col_2) for col_1, col_2 in combinations(self.cat_value_.keys(), 2)],
        #             pool.starmap(
        #                 merge_cat_table_cross,
        #                 [(pd.concat([x[[col_1, col_2]], y.to_frame("target")], axis=1), col_1, col_2)
        #                  for col_1, col_2 in combinations(self.cat_value_.keys(), 2)]
        #             )
        #         ))
        # self.cat_table_cross_ = {
        #     col: val for col, val in self.cat_table_cross_.items()
        #     if val["IV"].sum() > self.cat_value_[col.split(" @ ")[0]] + self.cat_value_[col.split(" @ ")[1]]}
        # self.cat_table_cross_ = {
        #     col: val for col, val in self.cat_table_cross_.items()
        #     if val["IV"].sum() > 0.1}
        # self.cat_value_cross_.update({col: val["IV"].sum() for col, val in self.cat_table_cross_.items()})
        # self.cat_value_cross_ = OrderedDict(sorted(self.cat_value_cross_.items(), key=lambda t: t[1], reverse=True))
        #
        # with Pool(mp.cpu_count() - 1) as pool:
        #     if len(self.num_value_.keys()) >= 2:
        #         self.num_table_cross_ = dict(zip(
        #             ["{} @ {}".format(col_1, col_2) for col_1, col_2 in combinations(self.num_value_.keys(), 2)],
        #             pool.starmap(
        #                 merge_num_table_cross,
        #                 [(pd.concat([x[[col_1, col_2]], y.to_frame("target")], axis=1), col_1, col_2)
        #                     for col_1, col_2 in combinations(self.num_value_.keys(), 2)]
        #             )
        #         ))
        # self.num_table_cross_ = {
        #     col: val for col, val in self.num_table_cross_.items()
        #     if val["IV"].sum() > self.num_value_[col.split(" @ ")[0]] + self.num_value_[col.split(" @ ")[1]]}
        # self.num_table_cross_ = {
        #     col: val for col, val in self.num_table_cross_.items()
        #     if val["IV"].sum() > 0.1}
        # self.num_value_cross_.update({col: val["IV"].sum() for col, val in self.num_table_cross_.items()})
        # self.num_value_cross_ = OrderedDict(sorted(self.num_value_cross_.items(), key=lambda t: t[1], reverse=True))
        #
        # self.information_value_cross_.update(self.cat_value_cross_)
        # self.information_value_cross_.update(self.num_value_cross_)
        # self.information_value_cross_ = OrderedDict(
        #     sorted(self.information_value_cross_.items(), key=lambda t: t[1], reverse=True))

        return self
