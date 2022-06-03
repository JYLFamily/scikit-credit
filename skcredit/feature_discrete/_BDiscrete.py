# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from joblib import Parallel,  delayed
from sklearn.pipeline import _transform_one
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class _BDiscrete(BaseEstimator,  TransformerMixin):
    def __init__(self, keep_columns, date_columns, feature_spliter=None):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.feature_columns = None
        self.feature_spliter = feature_spliter

        self.information_value_score = None
        self.information_value_table = None

    def fit(self, x,  y):
        pass

    def transform(self, x):
        z = pd.DataFrame()

        z[self.keep_columns] = x[self.keep_columns ]
        z[self.date_columns] = x[self.date_columns ]

        z = pd.concat([
            z,
            pd.concat(
                Parallel(n_jobs=-1, verbose=20)([
                    delayed(_transform_one  )(
                        smnd,
                        x[column.split(" @ ")],
                        None,
                        None
                    ) for column, smnd in self.feature_spliter.items()
                ]), axis="columns"
            )
        ], axis="columns")

        return z

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)

    def show_order(self):
        return self.information_value_score

    def show_table(self):
        return self.information_value_table

