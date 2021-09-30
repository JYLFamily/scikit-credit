# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.tools import mis
from skcredit.tools import cmi
from joblib import Parallel, delayed
from itertools   import combinations
from skcredit.feature_selection import Select
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SelectCIFE(Select):
    def __init__(self,   keep_columns, date_columns, nums_feature):
        super().__init__(keep_columns, date_columns)
        self.nums_feature = nums_feature

    def fit(self, x, y=None):
        self.feature_columns = np.array(
            [col for col in x.columns if col not in self.keep_columns and col not in self.date_columns])
        self.feature_support = np.zeros(self.feature_columns.shape[0], dtype=bool)

        f_t_mi = pd.Series([mis(x[column], y) for column in self.feature_columns],
                        index=self.feature_columns)
        f_f_mi = pd.DataFrame(np.zeros((self.feature_columns.shape[0], self.feature_columns.shape[0])),
                        columns=self.feature_columns, index=self.feature_columns)
        f_f_ci = pd.DataFrame(np.zeros((self.feature_columns.shape[0], self.feature_columns.shape[0])),
                        columns=self.feature_columns, index=self.feature_columns)

        mi_temp = Parallel(n_jobs=-1, verbose=20)(
            [delayed(mis)(x[col_i],  x[col_j]   ) for col_i, col_j in combinations(self.feature_columns, 2)])

        ci_temp = Parallel(n_jobs=-1, verbose=20)(
            [delayed(cmi)(x[col_i],  x[col_j], y) for col_i, col_j in combinations(self.feature_columns, 2)])

        for idx, (col_i, col_j) in enumerate(combinations(self.feature_columns, 2)):
            f_f_mi.loc[col_i, col_j] = mi_temp[idx]
            f_f_mi.loc[col_j, col_i] = mi_temp[idx]

            f_f_ci.loc[col_i, col_j] = ci_temp[idx]
            f_f_ci.loc[col_j, col_i] = ci_temp[idx]

        self.feature_support[f_t_mi.argmax()] = True

        for _ in range(self.nums_feature):
            score = ((
                f_t_mi.loc[self.feature_columns[~self.feature_support]] +
                f_f_ci.loc[self.feature_columns[~self.feature_support],
                           self.feature_columns[ self.feature_support]].mean(axis=1))  /
                f_f_mi.loc[self.feature_columns[~self.feature_support],
                           self.feature_columns[ self.feature_support]].mean(axis=1))

            self.feature_support[np.where(self.feature_columns == score.idxmax())[0]] = True

        return self
