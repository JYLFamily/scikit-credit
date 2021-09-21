# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from dataclasses  import dataclass
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
    bucket: set
    cnt_negative: float
    cnt_positive: float
    woe: float
    ivs: float
    l_child: dataclass = None
    r_child: dataclass = None


class SplitCat(Split):
    def __init__(self,   column, target, monotone_constraints, min_bin_cnt_negative=75, min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(column, target, monotone_constraints, min_bin_cnt_negative,    min_bin_cnt_positive,
                 min_information_value_split_gain)

    def fit( self, x, y):
        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[xy[self.column] != "missing", :].reset_index(drop=True)
        xy_mis = xy.loc[xy[self.column] == "missing", :].reset_index(drop=True)

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)

        pivot_table = xy_non.pivot_table(index=self.column, columns=self.target, aggfunc="size", fill_value=0)
        sub_cnt_negative = pivot_table[0].where(pivot_table[0] != 0, 0.5)
        sub_cnt_positive = pivot_table[1].where(pivot_table[1] != 0, 0.5)
        bucket = - np.log((sub_cnt_negative / (self.all_cnt_negative_non + self.all_cnt_negative_mis)) /
                          (sub_cnt_positive / (self.all_cnt_positive_non + self.all_cnt_positive_mis)))

        xy_non[self.column].replace(bucket.to_dict(), inplace=True)

        self.dtree = self._fit(xy_non, bucket.to_dict(),
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
        bucket.append({"missing"})
        cnt_negative.append(self.all_cnt_negative_mis)
        cnt_positive.append(self.all_cnt_positive_mis)

        if self.all_cnt_negative_mis == 0 and self.all_cnt_positive_non == 0:
            woe.append(0)
            ivs.append(0)
        elif self.all_cnt_negative_mis == 0:
            woe.append(self._stats(0.5, self.all_cnt_positive_non)[0])
            ivs.append(self._stats(0.5, self.all_cnt_positive_non)[1])
        elif self.all_cnt_positive_mis == 0:
            woe.append(self._stats(self.all_cnt_positive_mis, 0.5)[0])
            ivs.append(self._stats(self.all_cnt_positive_mis, 0.5)[1])

        self.table = pd.concat([
            pd.Series(column).to_frame("Column"),
            pd.Series(bucket).to_frame("Bucket"),
            pd.Series(cnt_negative).to_frame("CntNegative"),
            pd.Series(cnt_positive).to_frame("Cntpositive"),
            pd.Series(woe).to_frame("WoE"),
            pd.Series(ivs).to_frame("IVS"),
        ], axis=1)

        return self

    def _fit(self,       xy_non, bucket, cnt_negative, cnt_positive, woe,     ivs, min_value, max_value):
        node = Node(self.column,   set(bucket.keys()), cnt_negative, cnt_positive, woe,  ivs)
        info = self._split(xy_non, ivs,     min_value,    max_value)

        if info.split_point is None:
            return node

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        if self.monotone_constraints == "increasing":
            node.l_child = self._fit(
                info.xy_l_non, {key: val for key, val in bucket.items() if val <= info.split_point},
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                min_value, midd)
            node.r_child = self._fit(
                info.xy_r_non, {key: val for key, val in bucket.items() if val >  info.split_point},
                info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
                midd, max_value)

        if self.monotone_constraints == "decreasing":
            node.l_child = self._fit(
                info.xy_l_non, {key: val for key, val in bucket.items() if val <= info.split_point},
                info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
                midd, max_value)
            node.r_child = self._fit(
                info.xy_r_non, {key: val for key, val in bucket.items() if val >  info.split_point},
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
        else:
            # test 出现了 train 中没有出现的类别使用 train 中最大的（风险最高的） 类别 WoE 替换
            return self.table["WoE"].max()

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)


def binning_cat(x, y, column, target):
    sc = SplitCat(column, target)
    sc.fit(x, y)
    return sc


def replace_cat(x, sc):
    return sc.transform(x)
