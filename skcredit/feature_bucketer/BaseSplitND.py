# coding: utf-8

import warnings
import numpy   as np
import pandas  as pd
from numbers import Real
from portion import singleton
from portion import open    as  oo
from portion import closed  as  oc
from dataclasses  import dataclass
from itertools import product, chain
from skcredit.tools import NAN, NINF, PINF
from sklearn.base   import BaseEstimator,    TransformerMixin
from skcredit.tools import l_bound_operator, r_bound_operator, get_splits, get_direct, calc_stats
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
    sub_xy: pd.DataFrame
    sub_xy_cnt_negative: Real
    sub_xy_cnt_positive: Real
    sub_xy_woe: Real
    sub_xy_ivs: Real
    splits: dict
    direct: dict
    direct_min_value: Real = float('-inf')
    direct_max_value: Real = float('+inf')
    l_child: object = None
    r_child: object = None


class BaseSplitND(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.0015):

        self.min_bin_cnt_negative = min_bin_cnt_negative
        self.min_bin_cnt_positive = min_bin_cnt_positive
        self.min_information_value_split_gain = min_information_value_split_gain

        self.columns = None
        self.target  = None

        self.all_cnt_negative = None
        self.all_cnt_positive = None

        self._datas = list()
        self._table = None
        self._image = None

    def fit( self, x, y):
        self.columns = x.columns
        self.target  = y.name

        self.all_cnt_negative = y.tolist().count(0)
        self.all_cnt_positive = y.tolist().count(1)

        return self

    def _fit(self, x, y):
        # x 2D
        xy = pd.concat([x.reindex(columns=self.columns), y.to_frame(self.target)], axis=1)

        for masks in product(* [[0, 1] for _ in self.columns]):
            bucket = dict(zip(self.columns, [singleton(NAN) if mask else oo(NINF, PINF) for mask in masks]))
            sub_xy = xy[np.logical_and.reduce([xy[column].isna() if mask else xy[column].notna()
                  for column, mask in zip(self.columns, masks)], axis=0)].reset_index(drop=True)

            sub_cnt_negative = sub_xy[self.target].tolist().count(0)
            sub_cnt_positive = sub_xy[self.target].tolist().count(1)
            sub_woe, sub_ivs = calc_stats(
                sub_cnt_negative,
                sub_cnt_positive,
                self.all_cnt_negative,
                self.all_cnt_positive
            )

            root_node = Node(
                isleaf=True,
                column=self.columns,
                bucket=bucket,
                sub_xy=sub_xy,
                sub_xy_cnt_negative=sub_cnt_negative,
                sub_xy_cnt_positive=sub_cnt_positive,
                sub_xy_woe=sub_woe,
                sub_xy_ivs=sub_ivs,
                splits={column: get_splits(sub_xy[column], sub_xy[self.target]) for column in self.columns},
                direct={column: get_direct(sub_xy[column], sub_xy[self.target]) for column in self.columns},
            )

            self._datas.append([])
            self._build(root_node)

        # table
        self._table = pd.DataFrame.from_records(chain.from_iterable(self._datas))

        return self

    def transform( self, x):
        pass

    def _transform(self, x):
        x_transformed = pd.DataFrame(index=x.index, columns=[f"FEATURE({', '.join(self.columns)})"])

        for columns, buckets, woe in zip(self._table["Column"], self._table["Bucket"], self._table["WoE"]):
            masks = [l_bound_operator[bucket.left ](x[column], bucket.lower) &
                     r_bound_operator[bucket.right](x[column], bucket.upper)
                     for column, bucket in zip(columns, buckets)]

            x_transformed[np.logical_and.reduce(masks, axis=0)] = woe

        return x_transformed

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)

    def _build(self, node):
        node = self._split(node)

        if node.isleaf:
            self._datas[-1].append({
                "Column": f"FEATURE({', '.join(self.columns)})",
                "Bucket": list(node.bucket.values()   ),
                "CntPositive": node.sub_xy_cnt_positive,
                "CntNegative": node.sub_xy_cnt_negative,
                "CntPositive(%)": node.sub_xy_cnt_positive / self.all_cnt_positive,
                "CntNegative(%)": node.sub_xy_cnt_negative / self.all_cnt_negative,
                "WoE": node.sub_xy_woe,
                "IvS": node.sub_xy_ivs
            })

            return

        self._build(node.l_child)
        self._build(node.r_child)

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

                    temp_sub_xy_l_woe, temp_sub_xy_l_ivs = calc_stats(
                        temp_sub_xy_l_cnt_negative,
                        temp_sub_xy_l_cnt_positive,
                        self.all_cnt_negative,
                        self.all_cnt_positive)
                    temp_sub_xy_r_woe, temp_sub_xy_r_ivs = calc_stats(
                        temp_sub_xy_r_cnt_negative,
                        temp_sub_xy_r_cnt_positive,
                        self.all_cnt_negative,
                        self.all_cnt_positive)

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
                                    if column == temp_column else node.bucket[column] for column in self.columns},
                                sub_xy=temp_sub_xy_l,
                                sub_xy_cnt_negative=temp_sub_xy_l_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_l_cnt_positive,
                                sub_xy_woe=temp_sub_xy_l_woe,
                                sub_xy_ivs=temp_sub_xy_l_ivs,
                                direct=node.direct,
                                splits={column: [split for split in node.splits[column] if split <= temp_split]
                                    if column == temp_column else node.splits[column] for column in self.columns},
                                direct_min_value=(
                                    node.direct_min_value if node.direct[temp_column] == "increasing" else midd),
                                direct_max_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_max_value),
                            )
                            node.r_child = Node(
                                isleaf=True,
                                column=node.column,
                                bucket={column: node.bucket[column] & oo(temp_split, PINF)
                                    if column == temp_column else node.bucket[column] for column in self.columns},
                                sub_xy=temp_sub_xy_r,
                                sub_xy_cnt_negative=temp_sub_xy_r_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_r_cnt_positive,
                                sub_xy_woe=temp_sub_xy_r_woe,
                                sub_xy_ivs=temp_sub_xy_r_ivs,
                                splits={column: [split for split in node.splits[column] if split >  temp_split]
                                    if column == temp_column else node.splits[column] for column in self.columns},
                                direct=node.direct,
                                direct_min_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_min_value),
                                direct_max_value=(
                                    node.direct_max_value if node.direct[temp_column] == "increasing" else midd),
                            )

        return node

    def build_table(self):
        pass

    def build_image(self):
        pass

