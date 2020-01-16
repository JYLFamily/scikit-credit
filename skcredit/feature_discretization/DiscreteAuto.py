# coding:utf-8

import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
from skcredit.feature_discretization.BaseDiscrete import BaseDiscrete
from skcredit.feature_discretization.DiscreteImple import merge_cat_table, merge_num_table
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class DiscreteAuto(BaseDiscrete):
    def __init__(self, tim_columns):
        super().__init__(tim_columns)

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.cat_columns_ = [col for col in x.select_dtypes(include="object").columns if col not in self.tim_columns]
        self.num_columns_ = [col for col in x.select_dtypes(exclude="object").columns if col not in self.tim_columns]

        with Pool(mp.cpu_count() - 2) as pool:
            if self.cat_columns_:
                self.cat_table_ = dict(zip(self.cat_columns_, pool.starmap(
                    merge_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col) for col in self.cat_columns_])))
        self.cat_table_ = {
            col: val for col, val in self.cat_table_.items() if val["IV"].sum() >= 0.1}

        with Pool(mp.cpu_count() - 2) as pool:
            if self.num_columns_:
                self.num_table_ = dict(zip(self.num_columns_, pool.starmap(
                    merge_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col) for col in self.num_columns_])))
        self.num_table_ = {
            col: val for col, val in self.num_table_.items() if val["IV"].sum() >= 0.1}

        self.information_values_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.information_values_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.information_values_ = OrderedDict(
            sorted(self.information_values_.items(), key=lambda t: t[1], reverse=True))

        return self
