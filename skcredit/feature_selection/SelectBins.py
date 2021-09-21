# coding:utf-8

import numpy  as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from skcredit.feature_selection.Select import Select
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class SelectBins(Select):
    def __init__(self,   keep_columns, date_columns):
        super().__init__(keep_columns, date_columns)

    def fit(self, x, y=None):
        self.feature_columns_ = np.array(
            [column for column in x.columns if column not in self.keep_columns and column not in self.date_columns])

        intercept   = np.log(y.sum() / (y.shape[0] - y.sum()))
        coefficient = 1

        def _fit(x, y):
            logit_mod = sm.GLM(y, sm.add_constant(x, has_constant="add"), family=sm.families.Binomial())
            logit_res = logit_mod.fit()

            return (abs(logit_res.params["const"] -   intercept) <= 1e-2 and
                    abs(logit_res.params[x.name ] - coefficient) <= 1e-2)

        self.feature_support_ = np.array(
            Parallel(n_jobs=-1,  verbose=20)([delayed(_fit)(x[column],  y)  for column  in  self.feature_columns_]))

        return self



