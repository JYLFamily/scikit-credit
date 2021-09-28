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
                 monotone_constraints,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):

        self.monotone_constraints = monotone_constraints
        self.min_bin_cnt_negative = min_bin_cnt_negative
        self.min_bin_cnt_positive = min_bin_cnt_positive
        self.min_information_value_split_gain = min_information_value_split_gain

        self.column = None
        self.target = None

        self.all_cnt_negative_non = None
        self.all_cnt_positive_non = None
        self.all_cnt_negative_mis = None
        self.all_cnt_positive_mis = None

        self.table = list()

    def fit(self, x, y):
        self.column = x.name
        self.target = y.name

        return self

    def _split(self, xy_non, ivs, min_value, max_value):
        largest_ivs_gain = 0.0

        best_split = None
        best_xy_l_non = None
        best_xy_r_non = None
        best_xy_l_cnt_negative_non = None
        best_xy_l_cnt_positive_non = None
        best_xy_r_cnt_negative_non = None
        best_xy_r_cnt_positive_non = None
        best_xy_l_woe_non = None
        best_xy_r_woe_non = None
        best_xy_l_ivs_non = None
        best_xy_r_ivs_non = None

        for temp_split in np.unique(xy_non[self.column]):
            temp_xy_l_non = xy_non.loc[xy_non[self.column] <= temp_split, :]
            temp_xy_r_non = xy_non.loc[xy_non[self.column] >  temp_split, :]

            temp_xy_l_cnt_negative_non = temp_xy_l_non[self.target].tolist().count(0)
            temp_xy_l_cnt_positive_non = temp_xy_l_non[self.target].tolist().count(1)
            temp_xy_r_cnt_negative_non = temp_xy_r_non[self.target].tolist().count(0)
            temp_xy_r_cnt_positive_non = temp_xy_r_non[self.target].tolist().count(1)

            if (temp_xy_l_cnt_negative_non >= self.min_bin_cnt_positive and
                    temp_xy_l_cnt_positive_non >= self.min_bin_cnt_negative and
                    temp_xy_r_cnt_negative_non >= self.min_bin_cnt_positive and
                    temp_xy_r_cnt_positive_non >= self.min_bin_cnt_negative):

                temp_xy_l_woe_non, temp_xy_l_ivs_non = self._stats(
                    temp_xy_l_cnt_negative_non, temp_xy_l_cnt_positive_non)
                temp_xy_r_woe_non, temp_xy_r_ivs_non = self._stats(
                    temp_xy_r_cnt_negative_non, temp_xy_r_cnt_positive_non)

                if temp_xy_l_ivs_non + temp_xy_r_ivs_non - ivs > max(
                        self.min_information_value_split_gain, largest_ivs_gain):

                    if (min_value <= temp_xy_l_woe_non <= max_value and
                            min_value <= temp_xy_r_woe_non <= max_value and
                            (self.monotone_constraints == "increasing" and temp_xy_l_woe_non <= temp_xy_r_woe_non) or
                            (self.monotone_constraints == "decreasing" and temp_xy_l_woe_non >= temp_xy_r_woe_non)):

                        largest_ivs_gain = temp_xy_l_ivs_non + temp_xy_r_ivs_non - ivs

                        best_split = temp_split
                        best_xy_l_non = temp_xy_l_non
                        best_xy_r_non = temp_xy_r_non
                        best_xy_l_cnt_negative_non = temp_xy_l_cnt_negative_non
                        best_xy_l_cnt_positive_non = temp_xy_l_cnt_positive_non
                        best_xy_r_cnt_negative_non = temp_xy_r_cnt_negative_non
                        best_xy_r_cnt_positive_non = temp_xy_r_cnt_positive_non
                        best_xy_l_woe_non = temp_xy_l_woe_non
                        best_xy_r_woe_non = temp_xy_r_woe_non
                        best_xy_l_ivs_non = temp_xy_l_ivs_non
                        best_xy_r_ivs_non = temp_xy_r_ivs_non

        return Info(best_split, best_xy_l_non, best_xy_r_non,
                    best_xy_l_cnt_negative_non, best_xy_l_cnt_positive_non,
                    best_xy_r_cnt_negative_non, best_xy_r_cnt_positive_non,
                    best_xy_l_woe_non, best_xy_r_woe_non,
                    best_xy_l_ivs_non, best_xy_r_ivs_non)

    def _stats(self, sub_cnt_negative,  sub_cnt_positive):
        if not sub_cnt_negative and not sub_cnt_positive:
            return 0, 0

        sub_cnt_negative = 0.5 if not sub_cnt_negative else sub_cnt_negative
        sub_cnt_positive = 0.5 if not sub_cnt_positive else sub_cnt_positive

        negative_rate = sub_cnt_negative / (self.all_cnt_negative_non + self.all_cnt_negative_mis)
        positive_rate = sub_cnt_positive / (self.all_cnt_positive_non + self.all_cnt_positive_mis)

        woe = np.log(positive_rate  /  negative_rate)
        ivs = (positive_rate - negative_rate)  *  woe

        return woe, ivs
