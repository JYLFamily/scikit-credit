# coding: utf-8

import warnings
import numpy   as np
import pandas  as pd
from numbers import Real
from dataclasses  import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Node:
    isleaf: bool
    column: list
    bucket: dict
    splits: dict
    sub_xy: pd.DataFrame
    sub_xy_cnt_negative: Real
    sub_xy_cnt_positive: Real
    sub_xy_woe: Real
    sub_xy_ivs: Real
    direct: dict = None
    direct_min_value: Real = float('-inf')
    direct_max_value: Real = float('+inf')
    l_child: object = None
    r_child: object = None


class SplitND(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.001):

        self.min_bin_cnt_negative = min_bin_cnt_negative
        self.min_bin_cnt_positive = min_bin_cnt_positive
        self.min_information_value_split_gain = min_information_value_split_gain

        self.column = None
        self.target = None

        self.all_cnt_negative = None
        self.all_cnt_positive = None

        self.roots = list()
        self.datas = list()
        self.table = None

    def fit(self, x, y):
        self.column = x.columns.tolist()
        self.target = y.name

        self.all_cnt_negative = y.tolist().count(0)
        self.all_cnt_positive = y.tolist().count(1)

        return self

    def _stats(self, sub_cnt_negative,  sub_cnt_positive):
        if not sub_cnt_negative and not sub_cnt_positive:
            return 0, 0

        sub_cnt_negative = 0.0005 if not sub_cnt_negative else sub_cnt_negative
        sub_cnt_positive = 0.0005 if not sub_cnt_positive else sub_cnt_positive

        negative_rate = sub_cnt_negative / self.all_cnt_negative
        positive_rate = sub_cnt_positive / self.all_cnt_positive

        woe = np.log(positive_rate  /  negative_rate)
        ivs = (positive_rate - negative_rate)  *  woe

        return woe, ivs
