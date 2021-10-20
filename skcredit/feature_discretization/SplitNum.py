# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from portion import singleton
from portion import open   as oo
from portion import closed as oc
from scipy.stats import  spearmanr
from collections import namedtuple
from skcredit.tools import NINF,  PINF,  NAN
from skcredit.feature_discretization import Split,  Info
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)

Prebin = namedtuple("Prebin", ["bucket", "splits"])


def get_num_prebin(x, y):
    if (x.empty and y.empty) or np.all(x == x[0]) or np.all(y == y[0]):
        return Prebin(oo(NINF, PINF), [])

    return Prebin(oo(NINF, PINF),
                  np.unique(np.quantile(x, q=np.linspace(0, 1, 129), interpolation="higher")).tolist())


class SplitNum(Split):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.0001):
        super().__init__(
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

        self.monotone_constraints = None

    def fit(self,   x, y):
        super().fit(x, y)

        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[~xy[self.column].isna(), :].reset_index(drop=True)
        xy_mis = xy.loc[ xy[self.column].isna(), :].reset_index(drop=True)

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)

        prebin = get_num_prebin(xy_non[self.column], xy_non[self.target])
        if prebin.splits:
            self.monotone_constraints = ("increasing" if spearmanr(xy_non[self.column], xy_non[self.target])[0] > 0 else
                                         "decreasing")
        # non-missing
        self._calc_table_non(
            xy_non,
            prebin,
            self.all_cnt_negative_non,
            self.all_cnt_positive_non,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non),
            float('-inf'), float('+inf'))

        #     missing
        self._calc_table_mis(
            singleton(NAN),
            self.all_cnt_negative_mis,
            self.all_cnt_positive_mis,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non))

        # table
        self.table = pd.DataFrame.from_records(self.datas)

        return self

    def _calc_table_non(self, xy_non, prebin, cnt_negative, cnt_positive, woe, ivs, min_value, max_value):
        info = self._split(   xy_non, prebin, ivs, min_value, max_value)

        if info.split is None:
            self.datas.append({
                "Column":   self.column,
                "Bucket": prebin.bucket,
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            })
            return

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        if self.monotone_constraints == "increasing":
            self._calc_table_non(
                info.xy_l_non,
                Prebin(prebin.bucket & oc(NINF, info.split), [val for val in prebin.splits if val <= info.split]),
                info.xy_l_cnt_negative_non,  info.xy_l_cnt_positive_non,  info.xy_l_woe_non,  info.xy_l_ivs_non,
                min_value, midd)
            self._calc_table_non(
                info.xy_r_non,
                Prebin(prebin.bucket & oo(info.split, PINF), [val for val in prebin.splits if val >  info.split]),
                info.xy_r_cnt_negative_non,  info.xy_r_cnt_positive_non,  info.xy_r_woe_non,  info.xy_r_ivs_non,
                midd, max_value)

        if self.monotone_constraints == "decreasing":
            self._calc_table_non(
                info.xy_l_non,
                Prebin(prebin.bucket & oc(NINF, info.split), [val for val in prebin.splits if val <= info.split]),
                info.xy_l_cnt_negative_non,  info.xy_l_cnt_positive_non,  info.xy_l_woe_non,  info.xy_l_ivs_non,
                midd, max_value)
            self._calc_table_non(
                info.xy_r_non,
                Prebin(prebin.bucket & oo(info.split, PINF), [val for val in prebin.splits if val >  info.split]),
                info.xy_r_cnt_negative_non,  info.xy_r_cnt_positive_non,  info.xy_r_woe_non,  info.xy_r_ivs_non,
                min_value, midd)

    def _calc_table_mis(self, bucket, cnt_negative, cnt_positive, woe, ivs):
        self.datas.append({
                "Column": self.column,
                "Bucket": bucket,
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            })

    def _split(self, xy_non, prebin, ivs, min_value, max_value):
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

        for temp_split in prebin.splits:
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

    def transform( self, x):
        x_transformed = x.apply(lambda element: self._transform(element))

        return x_transformed

    def _transform(self, x):
        for bucket, woe in zip(self.table["Bucket"],  self.table["WoE"]):
            if x in bucket:
                return woe

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)


def binning_num(x,  y):
    sn = SplitNum()
    sn.fit(x, y)
    return sn


def replace_num(x, sn):
    return sn.transform(x)


if __name__ == "__main__":
    application_train = pd.read_csv("C:\\Users\\P1352\\Desktop\\application_train.csv")
    sn = SplitNum()
    sn.fit(application_train["EXT_SOURCE_3"], application_train["TARGET"])
    print(sn.table["Bucket"][0].upper)
