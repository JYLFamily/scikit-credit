# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.feature_discrete import _BDiscrete
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class CXDiscrete(_BDiscrete):
    def __init__(self,   keep_columns, date_columns, feature_spliter):
        super().__init__(keep_columns, date_columns, feature_spliter)

    def fit(self, x, y=None):
        self.feature_columns = np.array([col for col in x.columns
             if col not in self.keep_columns and col not in self.date_columns])

        self.information_value_score = pd.DataFrame.from_dict(
            {column: smnd.build_table()["IvS"].tolist()[-1]
             for column, smnd in self.feature_spliter.items()}, orient="index", columns=["IvS"])

        self.information_value_table = pd.concat([
            smnd.build_table() for smnd in self.feature_spliter.values()])

        return self
