# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectBin(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns

        self.feature_columns_ = None
        self.feature_support_ = None

    def fit(self, x, y=None):
        self.feature_columns_ = np.array([col for col in x.columns if col not in self.tim_columns])
        self.feature_support_ = np.ones(len(self.feature_columns_), dtype=bool)

        beta_0 = np.log(y.sum() / (y.shape[0] - y.sum()))
        beta_1 = 1

        for idx, col in enumerate(self.feature_columns_):
            logit_mod = sm.GLM(y, sm.add_constant(x[[col]], has_constant="add"), family=sm.families.Binomial())
            logit_res = logit_mod.fit()

            if (abs(logit_res.params["const"] - beta_0) > 1e-8 and
                    abs(logit_res.params[col] - beta_1) > 1e-8 and
                    logit_res.pvalues[col] != 0.):
                self.feature_support_[idx] = False
                logging.info("{:<10} remove !".format(col))

        return self

    def transform(self, x):
        return x[self.tim_columns + self.feature_columns_[self.feature_support_].tolist()]

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)