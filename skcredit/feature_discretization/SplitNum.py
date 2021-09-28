# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from portion.const import inf
from portion import open   as oo
from portion import closed as cc
from portion import openclosed as oc
from skcredit.feature_discretization import Split
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitNum(Split):
    def __init__(self,
                 monotone_constraints,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(
            monotone_constraints,
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

    def fit(self,   x, y):
        super().fit(x, y)

        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[xy[self.column] != -999999.0, :].reset_index(drop=True)
        xy_mis = xy.loc[xy[self.column] == -999999.0, :].reset_index(drop=True)

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)



        # non missing
        self._calc_table_non(xy_non, oo(-inf, inf),
            self.all_cnt_negative_non, self.all_cnt_positive_non,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non), float('-inf'), float('+inf'))

        # missing
        self._calc_table_mis(cc(-999999,  -999999),
            self.all_cnt_negative_mis, self.all_cnt_negative_mis,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non))

        # non missing & missing
        self.table = pd.concat([self.table_non, self.table_mis])

        return self

    def _calc_table_non(self, xy_non, bucket, cnt_negative, cnt_positive, woe, ivs, min_value, max_value):
        info = self._split(   xy_non, ivs, min_value, max_value)

        if info.split is None:
            self.table_non = self.table_non.append(pd.DataFrame.from_dict({
                "Column": self.column,
                "Bucket": bucket,
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            }, orient="index"))
            return

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        if self.monotone_constraints == "increasing":
            self._calc_table_non(
                info.xy_l_non, bucket & oc(-inf, info.split),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                min_value, midd)
            self._calc_table_non(
                info.xy_r_non, bucket & oo(info.split,  inf),
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                midd, max_value)

        if self.monotone_constraints == "decreasing":
            self._calc_table_non(
                info.xy_l_non, bucket & oc(-inf, info.split),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                midd, max_value)
            self._calc_table_non(
                info.xy_r_non, bucket & oo(info.split,  inf),
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                min_value, midd)

    def _calc_table_mis(self, bucket, cnt_negative, cnt_positive, woe, ivs):
        self.table_mis = self.table_mis.append(pd.DataFrame.from_dict({
                "Column": self.column,
                "Bucket": bucket,
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            }, orient="index"))

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