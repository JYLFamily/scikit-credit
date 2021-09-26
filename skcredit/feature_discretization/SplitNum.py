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
from astropy.stats import bayesian_blocks
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
    def __init__(self,   min_bin_cnt_negative=75, min_bin_cnt_positive=75, min_information_value_split_gain=0.015):
        super().__init__(min_bin_cnt_negative,    min_bin_cnt_positive,    min_information_value_split_gain)

    def fit(self,   x, y):
        super().fit(x, y)

        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[xy[self.column] != -999999.0, :].reset_index(drop=True)
        xy_mis = xy.loc[xy[self.column] == -999999.0, :].reset_index(drop=True)

        self.monotone_constraints = "increasing" if spearmanr(xy_non[self.column], xy_non[self.target]) > 0 else "decreasing"

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)

        bucket = oo(-inf, inf)
        xy_non[self.column] = (xy_non[self.column] if xy_non[self.column].nunique() < 100
                               else bayesian_blocks(xy_non[self.column]))

        self.dtree = self._fit(xy_non, bucket,
            self.all_cnt_negative_non, self.all_cnt_positive_non,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non),
            float('-inf'), float('inf'))

        column_list = list()
        bucket_list = list()
        cnt_negative_list = list()
        cnt_positive_list = list()
        woe_list = list()
        ivs_list = list()

        def leaf_rule(node):
            if node.l_child == node.r_child:
                column_list.append(node.column)
                bucket_list.append(node.bucket)
                cnt_negative_list.append(node.cnt_negative)
                cnt_positive_list.append(node.cnt_positive)
                woe_list.append(node.woe)
                ivs_list.append(node.ivs)
                return

            leaf_rule(node.l_child)
            leaf_rule(node.r_child)

        leaf_rule(self.dtree)

        column_list.append(self.column)
        bucket_list.append(cc(-999999, -999999))
        cnt_negative_list.append(self.all_cnt_negative_mis)
        cnt_positive_list.append(self.all_cnt_positive_mis)
        woe_list.append(self._stats(self.all_cnt_negative_mis, self.all_cnt_positive_mis)[0])
        ivs_list.append(self._stats(self.all_cnt_negative_mis, self.all_cnt_positive_mis)[1])

        self.table = pd.concat([
            pd.Series(column_list).to_frame("Column"),
            pd.Series(bucket_list).to_frame("Bucket"),
            pd.Series(cnt_negative_list).to_frame("CntNegative"),
            pd.Series(cnt_positive_list).to_frame("CntPositive"),
            pd.Series(woe_list).to_frame("WoE"),
            pd.Series(ivs_list).to_frame("IvS"),
        ], axis=1)

        return self

    def _fit(self,       xy_non, bucket, cnt_negative, cnt_positive, woe, ivs, min_value, max_value):
        node = Node(self.column, bucket, cnt_negative, cnt_positive, woe, ivs)
        info = self._split(xy_non,  ivs,    min_value,    max_value)

        if info.split is None:
            return node

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        if self.monotone_constraints == "increasing":
            node.l_child = self._fit(
                info.xy_l_non, bucket & oc(-inf, info.split),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                min_value, midd)
            node.r_child = self._fit(
                info.xy_r_non, bucket & oo(info.split,  inf),
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                midd, max_value)

        if self.monotone_constraints == "decreasing":
            node.l_child = self._fit(
                info.xy_l_non, bucket & oc(-inf, info.split),
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                midd, max_value)
            node.r_child = self._fit(
                info.xy_r_non, bucket & oo(info.split,  inf),
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


def binning_num(x,  y):
    sn = SplitNum()
    sn.fit(x, y)
    return sn


def replace_num(x, sn):
    return sn.transform(x)