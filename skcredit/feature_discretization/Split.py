# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from dataclasses   import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Info:
    split: float
    xy_l_non: pd.DataFrame
    xy_r_non: pd.DataFrame
    xy_l_cnt_negative_non: float
    xy_l_cnt_positive_non: float
    xy_r_cnt_negative_non: float
    xy_r_cnt_positive_non: float
    xy_l_woe_non: float
    xy_r_woe_non: float
    xy_l_ivs_non: float
    xy_r_ivs_non: float


class Split(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):

        self.min_bin_cnt_negative = min_bin_cnt_negative
        self.min_bin_cnt_positive = min_bin_cnt_positive
        self.min_information_value_split_gain = min_information_value_split_gain

        self.column = None
        self.target = None

        self.all_cnt_negative_non = None
        self.all_cnt_positive_non = None
        self.all_cnt_negative_mis = None
        self.all_cnt_positive_mis = None

        self.datas = list()
        self.table = None

    def fit(self, x, y):
        self.column = x.name
        self.target = y.name

        return self

    def _stats(self, sub_cnt_negative,  sub_cnt_positive):
        # missing
        if not sub_cnt_negative and not sub_cnt_positive:
            return 0, 0

        sub_cnt_negative = 0.0005 if not sub_cnt_negative else sub_cnt_negative
        sub_cnt_positive = 0.0005 if not sub_cnt_positive else sub_cnt_positive

        negative_rate = sub_cnt_negative / (self.all_cnt_negative_non + self.all_cnt_negative_mis)
        positive_rate = sub_cnt_positive / (self.all_cnt_positive_non + self.all_cnt_positive_mis)

        woe = np.log(positive_rate  /  negative_rate)
        ivs = (positive_rate - negative_rate)  *  woe

        return woe, ivs
