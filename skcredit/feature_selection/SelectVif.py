# coding:utf-8

import logging
import numpy  as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tools.sm_exceptions import PerfectSeparationError
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectVif(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns

        self.feature_columns_ = None
        self.feature_support_ = None

    def fit(self, x, y=None):
        self.feature_columns_ = np.array([col for col in x.columns if col not in self.tim_columns])
        self.feature_support_ = np.ones(len(self.feature_columns_), dtype=bool)

        for i in range(len(self.feature_columns_)):
            from operator import add
            for j in range(add(i, 1), len(self.feature_columns_)):
                if self.feature_support_[j]:
                    try:
                        ols_mod = sm.GLM(
                            x[self.feature_columns_].iloc[:,   j],
                            x[self.feature_columns_].iloc[:, [i]],
                            family=sm.families.Gaussian()
                        )
                        ols_res = ols_mod.fit()

                        rs = r2_score(
                            x[self.feature_columns_].iloc[:, j],
                            ols_res.predict(x[self.feature_columns_].iloc[:, [i]])
                        )

                        if rs >= 0.8:
                            self.feature_support_[j] = False
                            logging.info("{:<10} remove !".format(self.feature_columns_[j]))
                    except PerfectSeparationError:
                        self.feature_support_[j] = False
                        logging.info("{:<10} remove !".format(self.feature_columns_[j]))

        return self

    def transform(self, x):
        return x[self.tim_columns + self.feature_columns_[self.feature_support_].tolist()]

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)