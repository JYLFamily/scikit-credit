# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.feature_bucketer.SplitND import SplitND
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitNumND(SplitND):
    def __init__(self,
                 column,
                 target,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(
            column,
            target,
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

    def fit(self,   x,  y):
        super().fit(x,  y)
        self._fit(  x,  y)

        return self

    def transform(self, x):
        return self._transform(x)


def binning_num(x, y, column, target):
    snnd = SplitNumND(column, target)
    snnd.fit(x, y)

    return snnd


def replace_num(x, snnd):

    return snnd.transform(x)