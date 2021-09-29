# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
from collections import ChainMap
from joblib import Parallel, delayed
from skcredit.feature_discretization import Discrete
from skcredit.feature_discretization.SplitCat import binning_cat
from skcredit.feature_discretization.SplitNum import binning_num
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)
warnings.simplefilter(action="ignore", category=FutureWarning)


class DiscreteAuto(Discrete):
    def __init__(self,   keep_columns, date_columns, cat_columns, num_columns):
        super().__init__(keep_columns, date_columns)
        self.cat_columns = cat_columns
        self.num_columns = num_columns

    def fit(self, x,  y=None):
        if  self.cat_columns:
            self.cat_spliter = (dict(zip(
                self.cat_columns,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(binning_cat)(x[column], y) for column in self.cat_columns]))
            ))

        if  self.num_columns:
            self.num_spliter = (dict(zip(
                self.num_columns,
                Parallel(n_jobs=-1, verbose=20)(
                    [delayed(binning_num)(x[column], y) for column in self.num_columns]))
            ))

        temp = ChainMap(
            {column: spliter.table for column, spliter in self.cat_spliter.items()},
            {column: spliter.table for column, spliter in self.num_spliter.items()}
        )
        temp = dict(sorted(temp.items(), key=lambda item: item[1]["IvS"].sum(), reverse=True))

        self.information_value_score = pd.DataFrame.from_dict(
            {column: table["IvS"].sum() for column, table in temp.items()}, orient="index", columns=["IvS"])
        self.information_value_table = pd.concat(temp.values())

        return self
