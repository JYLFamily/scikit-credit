# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from functools import  reduce
from itertools import product
from portion import singleton
from portion import open    as oo
from portion import closed  as oc
from scipy.stats import spearmanr
from skcredit.tools import NINF,  PINF,  NAN
from skcredit.feature_discretization.SplitND import SplitND, Node
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_bucket(x, y):
    if (x.empty and y.empty) or np.all(x == x[0]) or np.all(y == y[0]) or pd.isna(x).all() or pd.isna(y).all():
        return singleton(NAN)

    return oo(NINF, PINF)


def get_splits(x, y):
    if (x.empty and y.empty) or np.all(x == x[0]) or np.all(y == y[0]) or pd.isna(x).all() or pd.isna(y).all():
        return []

    return np.unique(np.quantile(x, q=np.linspace(0, 1, 129), interpolation="higher")).tolist()


def get_direct(x, y):
    if (x.empty and y.empty) or np.all(x == x[0]) or np.all(y == y[0]) or pd.isna(x).all() or pd.isna(y).all():
        return "increasing"

    return "increasing" if spearmanr(x, y)[0] > 0 else "decreasing"


class SplitNum(SplitND):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

    def fit(self,   x, y):
        super().fit(x, y)

        xy = pd.concat([x, y.to_frame(self.target)], axis=1)

        for masks in product(* [[pd.notna(xy[column]), pd.isna(xy[column])] for column in self.column]):
            masks = np.logical_and(* masks)  # reduce

            sub_xy = xy.loc[masks, :].reset_index(drop=True)
            sub_cnt_negative = sub_xy[self.target].tolist().count(0)
            sub_cnt_positive = sub_xy[self.target].tolist().count(1)
            sub_woe, sub_ivs = self._stats(sub_cnt_negative, sub_cnt_positive)

            node = Node(
                isleaf=True,
                column=self.column,
                bucket={column: get_bucket(sub_xy[column], sub_xy[self.target]) for column in self.column},
                splits={column: get_splits(sub_xy[column], sub_xy[self.target]) for column in self.column},
                sub_xy=sub_xy,
                sub_xy_cnt_negative=sub_cnt_negative,
                sub_xy_cnt_positive=sub_cnt_negative,
                sub_xy_woe=sub_woe,
                sub_xy_ivs=sub_ivs,
                direct={column: get_direct(sub_xy[column], sub_xy[self.target]) for column in self.column},
            )
            self.roots.append(node)
            self._calc_table( node)

        # table
        self.table = pd.DataFrame.from_records(self.datas)

        return self

    def _calc_table(self,  node):
        node = self._split(node)

        if node.isleaf:
            self.datas.append({
                "Column": f"INTERACT({node.column})",
                "Bucket": list(node.bucket.values()),
                "CntPositive":  node.sub_xy_cnt_positive,
                "CntNegative":  node.sub_xy_cnt_negative,
                "WoE": node.sub_xy_woe,
                "IvS": node.sub_xy_ivs
            })
            return

        self._calc_table(node.l_child)
        self._calc_table(node.r_child)

    def _split(self, node):
        largest_ivs_gain = 0.0

        for    temp_column in node.column:
            for temp_split in node.splits[temp_column]:
                temp_sub_xy_l = node.sub_xy.loc[node.sub_xy[temp_column] <= temp_split, :]
                temp_sub_xy_r = node.sub_xy.loc[node.sub_xy[temp_column] >  temp_split, :]

                temp_sub_xy_l_cnt_negative = temp_sub_xy_l[self.target].tolist().count(0)
                temp_sub_xy_l_cnt_positive = temp_sub_xy_l[self.target].tolist().count(1)
                temp_sub_xy_r_cnt_negative = temp_sub_xy_r[self.target].tolist().count(0)
                temp_sub_xy_r_cnt_positive = temp_sub_xy_r[self.target].tolist().count(1)

                if (temp_sub_xy_l_cnt_negative >= self.min_bin_cnt_negative and
                    temp_sub_xy_l_cnt_positive >= self.min_bin_cnt_positive and
                    temp_sub_xy_r_cnt_negative >= self.min_bin_cnt_negative and
                    temp_sub_xy_r_cnt_positive >= self.min_bin_cnt_positive):

                    temp_sub_xy_l_woe, temp_sub_xy_l_ivs = self._stats(
                        temp_sub_xy_l_cnt_negative, temp_sub_xy_l_cnt_positive)
                    temp_sub_xy_r_woe, temp_sub_xy_r_ivs = self._stats(
                        temp_sub_xy_r_cnt_negative, temp_sub_xy_r_cnt_positive)

                    if temp_sub_xy_l_ivs + temp_sub_xy_r_ivs - node.sub_xy_ivs > max(
                            self.min_information_value_split_gain, largest_ivs_gain):

                        if (node.direct_min_value <= temp_sub_xy_l_woe <= node.direct_max_value and
                            node.direct_min_value <= temp_sub_xy_r_woe <= node.direct_max_value and
                           (node.direct[temp_column] == "increasing" and temp_sub_xy_l_woe < temp_sub_xy_r_woe) or
                           (node.direct[temp_column] == "decreasing" and temp_sub_xy_l_woe > temp_sub_xy_r_woe)):

                            largest_ivs_gain = temp_sub_xy_l_ivs + temp_sub_xy_r_ivs - node.sub_xy_ivs
                            midd = (temp_sub_xy_l_woe + temp_sub_xy_r_woe) / 2

                            node.isleaf  = False
                            node.l_child = Node(
                                isleaf=True,
                                column=node.column,
                                bucket={column: node.bucket[column] & oc(NINF, temp_split)
                                    if column == temp_column else node.bucket[column] for column in self.column},
                                splits={column: [split for split in node.splits[column] if split <= temp_split]
                                    if column == temp_column else node.splits[column] for column in self.column},
                                sub_xy=temp_sub_xy_l,
                                sub_xy_cnt_negative=temp_sub_xy_l_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_l_cnt_positive,
                                sub_xy_woe=temp_sub_xy_l_woe,
                                sub_xy_ivs=temp_sub_xy_l_ivs,
                                direct=node.direct,
                                direct_min_value=(
                                    node.direct_min_value if node.direct[temp_column] == "increasing" else midd),
                                direct_max_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_max_value),
                            )
                            node.r_child = Node(
                                isleaf=True,
                                column=node.column,
                                bucket={column: node.bucket[column] & oo(temp_split, PINF)
                                    if column == temp_column else node.bucket[column] for column in self.column},
                                splits={column: [split for split in node.splits[column] if split >  temp_split]
                                    if column == temp_column else node.splits[column] for column in self.column},
                                sub_xy=temp_sub_xy_r,
                                sub_xy_cnt_negative=temp_sub_xy_r_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_r_cnt_positive,
                                sub_xy_woe=temp_sub_xy_r_woe,
                                sub_xy_ivs=temp_sub_xy_r_ivs,
                                direct=node.direct,
                                direct_min_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_min_value),
                                direct_max_value=(
                                    node.direct_max_value if node.direct[temp_column] == "increasing" else midd),
                            )

        return node

    # def transform( self, x):
    #     x_transformed = x.apply(lambda element: self._transform(element))
    #
    #     return x_transformed
    #
    # def _transform(self, x):
    #     for bucket, woe in zip(self.table["Bucket"],  self.table["WoE"]):
    #         if x in bucket:
    #             return woe
    #
    # def fit_transform(self, x, y=None, **fit_params):
    #     self.fit(x, y)
    #
    #     return self.transform(x)


def binning_num(x,  y):
    sn = SplitNum()
    sn.fit(x, y)
    return sn


def replace_num(x, sn):
    return sn.transform(x)


if __name__ == "__main__":
    application_train = pd.read_csv("C:\\Users\\15795\\Desktop\\application_train.csv", usecols=["EXT_SOURCE_1", "EXT_SOURCE_3", "TARGET"])
    sn = SplitNum()
    sn.fit(application_train[["EXT_SOURCE_1", "EXT_SOURCE_3"]], application_train["TARGET"])
    print(sn.table.to_markdown())