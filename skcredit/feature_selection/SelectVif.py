# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
from skcredit.feature_selection import BaseSelect
from skcredit.snacks import symmetrical_uncertainty
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectVif(BaseSelect):
    def __init__(self,   keep_columns, date_columns):
        super().__init__(keep_columns, date_columns)

    def fit(self, X,  y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array([col for col in x.columns
                                          if col not in self.keep_columns and col not in self.date_columns])
        self.feature_support_ = np.zeros(len(self.feature_columns_), dtype=bool)

        su_target = pd.Series([symmetrical_uncertainty(x[column], y) for column in self.feature_columns_],
                              index=self.feature_columns_)
        su_column = x[self.feature_columns_].corr(symmetrical_uncertainty)

        for i in range(10):
            score = su_target.loc[self.feature_columns_[~self.feature_support_]].divide(
                    su_column.loc[self.feature_columns_[~self.feature_support_],
                                  self.feature_columns_[self.feature_support_]].mean(axis=1))
            self.feature_support_[score.argmax()] = True
            print(score)

        print(self.feature_columns_[self.feature_support_])

        return self
