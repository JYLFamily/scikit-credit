# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
from collections import ChainMap
from skcredit.feature_discretization import Discrete
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
warnings.simplefilter(action="ignore", category=FutureWarning)


class DiscreteCust(Discrete):
    def __init__(self,   keep_columns, date_columns, cat_spliter, num_spliter):
        super().__init__(keep_columns, date_columns)

        self.cat_spliter = cat_spliter
        self.num_spliter = num_spliter

    def fit(self, x,  y=None):
        temp = ChainMap(
            {column: spliter.table for column, spliter in self.cat_spliter.items()},
            {column: spliter.table for column, spliter in self.num_spliter.items()}
        )
        temp = dict(sorted(temp.items(), key=lambda item: item[1]["IvS"].sum(), reverse=True))

        self.information_value_score = pd.DataFrame.from_dict(
            {column: table["IvS"].sum() for column, table in temp.items()}, orient="index", columns=["IvS"])
        self.information_value_table = pd.concat(temp.values())

        return self
