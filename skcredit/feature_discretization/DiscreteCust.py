# coding:utf-8

import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import OrderedDict
from skcredit.feature_discretization import BaseDiscrete
from skcredit.feature_discretization.DiscreteImple import force_cat_table, force_num_table
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class DiscreteCust(BaseDiscrete):
    def __init__(self, keep_columns, cat_columns, num_columns, force_bin):
        super().__init__(keep_columns, cat_columns, num_columns)

        self.force_bin = force_bin

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        with Pool() as pool:
            if len(self.cat_columns) != 0:
                self.cat_table_ = dict(zip(self.cat_columns, pool.starmap(
                    force_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.force_bin[col]) for col in
                     self.cat_columns])))
        self.cat_table_ = {col: val for col, val in self.cat_table_.items()}

        with Pool() as pool:
            if len(self.num_columns) != 0:
                self.num_table_ = dict(zip(self.num_columns, pool.starmap(
                    force_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.force_bin[col]) for col in
                        self.num_columns])))
        self.num_table_ = {col: val for col, val in self.num_table_.items()}

        self.information_values_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.information_values_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.information_values_ = OrderedDict(
            sorted(self.information_values_.items(), key=lambda t: t[1], reverse=True))

        return self
