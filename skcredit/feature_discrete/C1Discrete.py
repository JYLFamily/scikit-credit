# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from joblib import Parallel,  delayed
from sklearn.pipeline import _fit_one
from skcredit.feature_discrete import _BDiscrete
from skcredit.feature_discrete import SplitMixND
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class C1Discrete(_BDiscrete):
    def __init__(self,   keep_columns, date_columns):
        super().__init__(keep_columns, date_columns)

    def fit(self, x, y=None):
        self.feature_columns = np.array([col for col in x.columns
             if col not in self.keep_columns and col not in self.date_columns])

        self.feature_spliter = dict(zip(
            self.feature_columns,
            Parallel(verbose=20 , n_jobs=-1)([
                delayed(_fit_one)(
                    SplitMixND(),
                    x[[column] ],
                    y,
                    weight=None
                ) for column in self.feature_columns
            ])
            )
        )

        self.information_value_score = pd.DataFrame.from_dict(
            {column: smnd.build_table()["IvS"].tolist()[-1]
             for column, smnd in self.feature_spliter.items()}, orient="index", columns=["IvS"])

        self.information_value_table = pd.concat([
            smnd.build_table() for smnd in self.feature_spliter.values()])

        return self



