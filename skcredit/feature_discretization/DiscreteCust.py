# coding:utf-8

import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
from skcredit.feature_discretization import BaseDiscrete
from skcredit.feature_discretization.DiscreteImple import force_cat_table, force_num_table
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class DiscreteCust(BaseDiscrete):
    def __init__(self, tim_columns, group_dict, break_dict):
        super().__init__(tim_columns)

        self.group_dict = group_dict
        self.break_dict = break_dict

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.cat_columns_ = [col for col in x.select_dtypes(include="object").columns if col not in self.tim_columns]
        self.num_columns_ = [col for col in x.select_dtypes(exclude="object").columns if col not in self.tim_columns]

        with Pool(mp.cpu_count() - 2) as pool:
            if self.cat_columns_:
                self.cat_table_ = dict(zip(self.cat_columns_, pool.starmap(
                    force_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.group_dict[col]) for col in
                     self.cat_columns_])))
        self.cat_table_ = {col: val for col, val in self.cat_table_.items()}

        with Pool(mp.cpu_count() - 2) as pool:
            if self.num_columns_:
                self.num_table_ = dict(zip(self.num_columns_, pool.starmap(
                    force_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.break_dict[col]) for col in
                        self.num_columns_])))
        self.num_table_ = {col: val for col, val in self.num_table_.items()}

        self.information_value_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.information_value_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.information_value_ = OrderedDict(
            sorted(self.information_value_.items(), key=lambda t: t[1], reverse=True))

        return self
