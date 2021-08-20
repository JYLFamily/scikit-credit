# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
from skcredit.tools import mi
from itertools   import combinations
from skcredit.tools import cmi as ci
from skcredit.feature_selection import BaseSelect
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectCMIM(BaseSelect):
    def __init__(self,   keep_columns, date_columns):
        super().__init__(keep_columns, date_columns)

    def fit(self, X,  y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array([col for col in x.columns
                                          if col not in self.keep_columns and col not in self.date_columns])
        self.feature_support_ = np.zeros(self.feature_columns_.shape[0], dtype=bool)

        f_t_mi = pd.Series([mi(x[column], y) for column in self.feature_columns_],
                        index=self.feature_columns_)
        f_f_mi = pd.DataFrame(np.zeros((self.feature_columns_.shape[0], self.feature_columns_.shape[0])),
                        columns=self.feature_columns_, index=self.feature_columns_)
        f_f_ci = pd.DataFrame(np.zeros((self.feature_columns_.shape[0], self.feature_columns_.shape[0])),
                        columns=self.feature_columns_, index=self.feature_columns_)

        for col_i, col_j in combinations(self.feature_columns_, 2):
            mi_temp = mi(x[col_i], x[col_j])
            f_f_mi.loc[col_i, col_j] = mi_temp
            f_f_mi.loc[col_j, col_i] = mi_temp

            ci_temp = ci(x[col_i], x[col_j], y)
            f_f_ci.loc[col_i, col_j] = ci_temp
            f_f_ci.loc[col_j, col_i] = ci_temp

        # self.feature_columns_[~self.feature_support_] no select feature
        # self.feature_columns_[self.feature_support_]  selected  feature

        self.feature_support_[f_t_mi.argmax()] = True

        for _ in range(25):
            score = (
                f_t_mi.loc[self.feature_columns_[~self.feature_support_]] -
                (f_f_ci.loc[self.feature_columns_[~self.feature_support_],
                            self.feature_columns_[self.feature_support_]] -
                 f_f_mi.loc[self.feature_columns_[~self.feature_support_],
                            self.feature_columns_[self.feature_support_]]).max(axis=1))

            self.feature_support_[np.where(self.feature_columns_ == score.idxmax())[0]] = True
        print(self.feature_columns_[self.feature_support_])
        return self
