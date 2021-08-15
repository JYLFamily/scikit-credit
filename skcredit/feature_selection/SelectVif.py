# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
import statsmodels.api as sm
from skcredit.feature_selection.BaseSelect import BaseSelect
from statsmodels.stats.outliers_influence import variance_inflation_factor
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class SelectVif(BaseSelect):
    def __init__(self,   keep_columns, time_columns):
        super().__init__(keep_columns, time_columns)

    def fit(self, X,  y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array([col for col in x.columns
                                          if col not in self.keep_columns and col not in self.time_columns])
        self.feature_support_ = np.zeros(len(self.feature_columns_), dtype=bool)

        sm.GLM(y, sm.add_constant(x[[col]], has_constant="add"), family=sm.families.Binomial()).fit()

        for i in min(25, len(self.feature_columns_)):
            for col in self.feature_columns_[self.feature_support_]:
                variance_inflation_factor(self.feature_columns_[self.feature_support_], col)


        return self