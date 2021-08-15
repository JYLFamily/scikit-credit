# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
import statsmodels.api as sm
from skcredit.feature_selection.BaseSelect import BaseSelect
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectBin(BaseSelect):
    def __init__(self,   keep_columns, time_columns):
        super().__init__(keep_columns, time_columns)

    def fit(self, X,  y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array([col for col in x.columns
                                          if col not in self.keep_columns and col not in self.time_columns])
        self.feature_support_ = np.zeros(len(self.feature_columns_), dtype=bool)

        beta_0 = np.log(y.sum() / (y.shape[0] - y.sum()))
        beta_1 = 1

        for idx, col in enumerate(self.feature_columns_):
            logit_mod = sm.GLM(y, sm.add_constant(x[[col]], has_constant="add"), family=sm.families.Binomial())
            logit_res = logit_mod.fit()

            if (abs(logit_res.params["const"] - beta_0) <= 1e-8 and
                    abs(logit_res.params[col] - beta_1) <= 1e-8 and
                    logit_res.pvalues[col] != 0.):
                self.feature_support_[idx] = True
                logging.info("{:<10} retain !".format(col))

        return self

