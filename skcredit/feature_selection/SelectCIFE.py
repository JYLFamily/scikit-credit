# coding:utf-8

import numpy  as np
import pandas as pd
from skcredit.tools import mi
from joblib import Parallel, delayed
from itertools   import combinations
from skcredit.tools import cmi as ci
from skcredit.feature_selection import Select
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class SelectCIFE(Select):
    def __init__(self,   keep_columns, date_columns, nums_feature):
        super().__init__(keep_columns, date_columns)
        self.nums_feature = nums_feature

    def fit(self, x, y=None):

        self.feature_columns_ = np.array([col for col in x.columns
                                          if col not in self.keep_columns and col not in self.date_columns])
        self.feature_support_ = np.zeros(self.feature_columns_.shape[0], dtype=bool)

        f_t_mi = pd.Series([mi(x[column], y) for column in self.feature_columns_],
                        index=self.feature_columns_)
        f_f_mi = pd.DataFrame(np.zeros((self.feature_columns_.shape[0], self.feature_columns_.shape[0])),
                        columns=self.feature_columns_, index=self.feature_columns_)
        f_f_ci = pd.DataFrame(np.zeros((self.feature_columns_.shape[0], self.feature_columns_.shape[0])),
                        columns=self.feature_columns_, index=self.feature_columns_)

        mi_temp = Parallel(n_jobs=-1, verbose=20)(
            [delayed(mi)(x[col_i],  x[col_j]   ) for col_i, col_j in combinations(self.feature_columns_, 2)])

        ci_temp = Parallel(n_jobs=-1, verbose=20)(
            [delayed(ci)(x[col_i],  x[col_j], y) for col_i, col_j in combinations(self.feature_columns_, 2)])

        for idx, (col_i, col_j) in enumerate(combinations(self.feature_columns_, 2)):
            f_f_mi.loc[col_i, col_j] = mi_temp[idx]
            f_f_mi.loc[col_j, col_i] = mi_temp[idx]

            f_f_ci.loc[col_i, col_j] = ci_temp[idx]
            f_f_ci.loc[col_j, col_i] = ci_temp[idx]

        self.feature_support_[f_t_mi.argmax()] = True

        for _ in range(self.nums_feature):
            score = ((
                f_t_mi.loc[self.feature_columns_[~self.feature_support_]] +
                f_f_ci.loc[self.feature_columns_[~self.feature_support_],
                           self.feature_columns_[self.feature_support_]].mean(axis=1))  /
                f_f_mi.loc[self.feature_columns_[~self.feature_support_],
                           self.feature_columns_[self.feature_support_]].mean(axis=1))

            self.feature_support_[np.where(self.feature_columns_ == score.idxmax())[0]] = True

        return self
