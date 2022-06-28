# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
import statsmodels.api         as sm
from joblib import Parallel, delayed
from skcredit.feature_selector.BaseSelect import BaseSelect
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SelectBINS(BaseSelect):
    def __init__(self,   keep_columns, date_columns, nthread, verbose):
        super(SelectBINS, self).__init__(
                         keep_columns, date_columns, nthread, verbose)

    def fit(self, x, y=None):
        super(SelectBINS, self).fit(x, y)

        intercept   = np.log(y.sum() / (y.shape[0] - y.sum()))
        coefficient = 1

        def _fit(x_in, y_in):
            logit_mod = sm.GLM(y_in, sm.add_constant(x_in,  has_constant="add"), family=sm.families.Binomial())
            logit_res = logit_mod.fit()

            return (abs(logit_res.params["const"  ] -   intercept) <= 1e-2 and
                    abs(logit_res.params[x_in.name] - coefficient) <= 1e-2)

        self.feature_support = np.array(
            Parallel(n_jobs=self.nthread, verbose=self.verbose)(
                [delayed(_fit)(x[column], y) for column in self.feature_columns]))

        return self



