# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from portion import Interval
from portion.const import  inf
from portion import open   as oo
from portion import closed as cc
from scipy.stats import spearmanr
from dataclasses import dataclass
from portion import openclosed as oc
from skcredit.feature_discretization import Split
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Node:
    column: str
    bucket: Interval
    cnt_negative: float
    cnt_positive: float
    woe: float
    ivs: float
    l_child: dataclass = None
    r_child: dataclass = None


class SplitNum(Split):
    def __init__(self,   column, target, monotone_constraints, min_bin_cnt_negative=75, min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(column, target, monotone_constraints, min_bin_cnt_negative,    min_bin_cnt_positive,
                 min_information_value_split_gain)

    def fit( self, x, y):
        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[xy[self.column] != -999999, :].reset_index(drop=True)
        xy_mis = xy.loc[xy[self.column] == -999999, :].reset_index(drop=True)

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)

        self.dtree = self._fit(xy_non, oo(-inf, inf),
            self.all_cnt_negative_non, self.all_cnt_positive_non, 0., 0.,
            float('-inf'), float('inf'))

        column = list()
        bucket = list()
        cnt_negative = list()
        cnt_positive = list()
        woe = list()
        ivs = list()

        def leaf_rule(node):
            if node.l_child == node.r_child:
                column.append(node.column)
                bucket.append(node.bucket)
                cnt_negative.append(node.cnt_negative)
                cnt_positive.append(node.cnt_positive)
                woe.append(node.woe)
                ivs.append(node.ivs)
                return

            leaf_rule(node.l_child)
            leaf_rule(node.r_child)

        leaf_rule(self.dtree)

        column.append(self.column)
        bucket.append(cc(-999999, -999999))
        cnt_negative.append(self.all_cnt_negative_mis)
        cnt_positive.append(self.all_cnt_positive_mis)

        if self.all_cnt_negative_mis == 0 and self.all_cnt_positive_mis == 0:
            woe.append(0)
            ivs.append(0)
        elif self.all_cnt_negative_mis == 0:
            woe.append(self._stats(0.5, self.all_cnt_positive_mis)[0])
            ivs.append(self._stats(0.5, self.all_cnt_positive_mis)[1])
        elif self.all_cnt_positive_mis == 0:
            woe.append(self._stats(self.all_cnt_negative_mis, 0.5)[0])
            ivs.append(self._stats(self.all_cnt_negative_mis, 0.5)[1])

        self.table = pd.concat([
            pd.Series(column).to_frame("Column"),
            pd.Series(bucket).to_frame("Bucket"),
            pd.Series(cnt_negative).to_frame("CntNegative"),
            pd.Series(cnt_positive).to_frame("Cntpositive"),
            pd.Series(woe).to_frame("WoE"),
            pd.Series(ivs).to_frame("IVS"),
        ], axis=1)

        return self

    def _fit(self,       xy_non, bucket, cnt_negative, cnt_positive, woe, ivs, min_value, max_value):
        node = Node(self.column, bucket, cnt_negative, cnt_positive, woe, ivs)
        info = self._split(xy_non,  ivs,    min_value,    max_value)

        if info.split_point is None:
            return node

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        if self.monotone_constraints == "increasing":
            node.l_child = self._fit(
                info.xy_l_non, bucket & oc(-inf, info.split_point),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                min_value, midd)
            node.r_child = self._fit(
                info.xy_r_non, bucket & oo(info.split_point,  inf),
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                midd, max_value)

        if self.monotone_constraints == "decreasing":
            node.l_child = self._fit(
                info.xy_l_non, bucket & oc(-inf, info.split_point),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                midd, max_value)
            node.r_child = self._fit(
                info.xy_r_non, bucket & oo(info.split_point,  inf),
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                min_value, midd)

        return node

    def transform(self, x):
        x_transformed = x.apply(lambda element: self._transform(element))

        return x_transformed

    def _transform(self, x):
        for bucket, woe in zip(self.table["Bucket"],  self.table["WoE"]):
            if x in bucket:
                return woe

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)


def binning_num(x,  y, column, target):
    sn = SplitNum(column, target,
                  "increasing" if spearmanr(x, y, nan_policy='omit')[0] > 0 else "decreasing")
    sn.fit(x, y)
    return sn


def replace_num(x, sn):
    return sn.transform(x)