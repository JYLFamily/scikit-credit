# coding: utf-8

import warnings
import numpy   as np
import pandas  as pd
from numbers import Real
from skcredit.tools import  *
from portion import singleton
from itertools import product
import plotly.graph_objects as go
from portion import open   as  oo
from dataclasses import dataclass
from portion import openclosed as oc
from sklearn.base   import BaseEstimator,  TransformerMixin
from skcredit.feature_bucketer.WoEEncoder import WoEEncoder
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Node:
    isleaf: bool
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


class SplitMixND(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):

        self.min_bin_cnt_negative = min_bin_cnt_negative
        self.min_bin_cnt_positive = min_bin_cnt_positive
        self.min_information_value_split_gain = min_information_value_split_gain

        self.cat_columns = None
        self.num_columns = None
        self.all_columns = None
        self.cat_encoder = None

        self.target  = None

        self.all_cnt_negative = None
        self.all_cnt_positive = None

        self._datas = list()

        self._table = None
        self._image = None

    def fit( self, x, y):
        self.cat_columns = x.select_dtypes(include="category").columns.tolist()
        self.num_columns = x.select_dtypes(exclude="category").columns.tolist()
        self.all_columns = x.columns.tolist()

        self.target = y.name

        self.all_cnt_negative = y.tolist().count(0)
        self.all_cnt_positive = y.tolist().count(1)

        self.cat_encoder =  WoEEncoder().fit(x,  y)
        self._fit(self.cat_encoder.transform(x), y)

        return self

    def _fit(self, x, y):
        xy = pd.concat([x.reindex(columns=self.all_columns), y.to_frame(self.target)], axis=1)

        for masks in product(* [[0, 1] for _ in self.all_columns]):
            bucket = dict(zip(self.all_columns,  [singleton(NAN) if mask else oo(NINF, PINF) for mask in masks]))
            sub_xy = xy[np.logical_and.reduce(
                [xy[column].isna() if  mask else
                ~xy[column].isna() for column, mask in zip(self.all_columns, masks)], axis=0)]

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
                bucket=bucket,
                sub_xy=sub_xy,
                sub_xy_cnt_negative=sub_cnt_negative,
                sub_xy_cnt_positive=sub_cnt_positive,
                sub_xy_woe=sub_woe,
                sub_xy_ivs=sub_ivs,
                splits={column: get_splits(sub_xy[column], sub_xy[self.target]) for column in self.all_columns},
                direct={column: get_direct(sub_xy[column], sub_xy[self.target]) for column in self.all_columns},
            )

            self._datas.append([])
            self._build(root_node)

        return self

    def transform( self, x):

        return self._transform(self.cat_encoder.transform(x))

    def _transform(self, x):
        x_transformed = pd.DataFrame(index=x.index, columns=[f"WOE({', '.join(self.all_columns)})"])

        for masks, datas in zip(product(* [[0, 1] for _ in self.all_columns]),  self._datas):
            sub_x = x[np.logical_and.reduce(
                [x[column].isna() if  mask else
                ~x[column].isna() for column, mask in zip(self.all_columns, masks)], axis=0)]
            for data in datas:
                x_transformed.loc[sub_x[np.logical_and.reduce(
                    [l_bound_operator[bucket.left](sub_x[column], bucket.lower) &
                    r_bound_operator[bucket.right](sub_x[column], bucket.upper)
                    for column, bucket in zip(data["Column"], data["Bucket"])], axis=0)].index, :] = data["WoE"]

        return x_transformed

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)

    def _build(self,  node):
        node = self._split(node)

        if node.isleaf:
            self._datas[ -1].append({
                "Bucket": node.bucket,
                "CntPositive": node.sub_xy_cnt_positive,
                "CntNegative": node.sub_xy_cnt_negative,
                "PctPositive": node.sub_xy_cnt_positive / self.all_cnt_positive,
                "PctNegative": node.sub_xy_cnt_negative / self.all_cnt_negative,
                "WoE": node.sub_xy_woe,
                "IvS": node.sub_xy_ivs
            })

            return

        self._build(node.l_child)
        self._build(node.r_child)

    def _split(self,  node):
        largest_ivs_gain = 0.0

        for temp_column, temp_splits in node.splits.items():
            for temp_split in temp_splits:
                temp_sub_xy_l = node.sub_xy[node.sub_xy[temp_column] <= temp_split]
                temp_sub_xy_r = node.sub_xy[node.sub_xy[temp_column] >  temp_split]

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
                                bucket={column: node.bucket[column] & oc(NINF, temp_split)
                                    if column == temp_column else node.bucket[column] for column in node.bucket.keys()},
                                sub_xy=temp_sub_xy_l,
                                sub_xy_cnt_negative=temp_sub_xy_l_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_l_cnt_positive,
                                sub_xy_woe=temp_sub_xy_l_woe,
                                sub_xy_ivs=temp_sub_xy_l_ivs,
                                direct=node.direct,
                                splits={column: [split for split in node.splits[column] if split <= temp_split]
                                    if column == temp_column else node.splits[column] for column in node.splits.keys()},
                                direct_min_value=(
                                    node.direct_min_value if node.direct[temp_column] == "increasing" else midd),
                                direct_max_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_max_value),
                            )
                            node.r_child = Node(
                                isleaf=True,
                                bucket={column: node.bucket[column] & oo(temp_split, PINF)
                                    if column == temp_column else node.bucket[column] for column in node.bucket.keys()},
                                sub_xy=temp_sub_xy_r,
                                sub_xy_cnt_negative=temp_sub_xy_r_cnt_negative,
                                sub_xy_cnt_positive=temp_sub_xy_r_cnt_positive,
                                sub_xy_woe=temp_sub_xy_r_woe,
                                sub_xy_ivs=temp_sub_xy_r_ivs,
                                splits={column: [split for split in node.splits[column] if split >  temp_split]
                                    if column == temp_column else node.splits[column] for column in node.splits.keys()},
                                direct=node.direct,
                                direct_min_value=(
                                    midd if node.direct[temp_column] == "increasing" else node.direct_min_value),
                                direct_max_value=(
                                    node.direct_max_value if node.direct[temp_column] == "increasing" else midd),
                            )

        return node

    def build_table(self ):
        if self._table is not None:
            return self._table

        table_dict = prepare_table(
            self._datas,
            self.cat_columns,
            self.all_columns,
            self.cat_encoder,
        )

        self._table = go.Figure()

        for idx, (label, table) in enumerate(table_dict.items()):
            self._table.add_trace(
                go.Table(
                    header=dict(
                        values=[f"<b>{column}</b>" for column in table.columns],
                        fill_color="rgb(128, 128, 128)",
                        line_color="rgb(128, 128, 128)",
                        font=dict(family="Courier New", color="white", size=14),
                    ),
                    cells=dict(
                        values=[column.tolist() for _, column in table.items()],
                        fill_color="rgb(255, 255, 255)",
                        line_color="rgb(128, 128, 128)",
                        font=dict(family="Courier New", color="black", size=12),
                    ),
                    name=label,
                ),
            )

        self._table.update_layout(
            title={
                "text":    f"<b>TABLE({' @ '.join(self.all_columns)})</b>",
                "x": 0.500,
                "y": 0.975,
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 16,
                },
            },
            margin={
                "t": 60,
                "b": 5,
                "l": 5,
                "r": 5,
            }
        )

        return self._table

    def build_image(self ):
        if self._image is not None:
            return self._image

        image_dict = prepare_image(
            self._datas,
            self.cat_columns,
            self.all_columns,
            self.cat_encoder,
        )

        self._image = go.Figure()

        for label, image in image_dict.items():
            self._image.add_trace(
                go.Bar(
                    x=image["Bucket"],
                    y=image["WoE"   ],
                    name=label,
                    hovertemplate="Bucket: %{x}<br>WoE: %{y}",
                )
            )

        self._image.update_layout(
            title={
                "text":  f"<b>WoE Bar({' @ '.join(self.all_columns)})</b>",
                "x": 0.500,
                "y": 0.975,
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 16,
                },
            },
            margin={
                "t": 60,
                "b": 5,
                "l": 5,
                "r": 5,
            },
            legend={
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 14,
                },
            },
            hoverlabel={
                "bgcolor"    : "rgb(255, 255, 255)",
                "bordercolor": "rgb(128, 128, 128)",
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 14,
                },
            }
        )

        self._image.update_xaxes(
            title={
                "text": "Buckets",
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 14,
                },
            },
            tickfont={
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 12
            }
        )
        self._image.update_yaxes(
            title={
                "text": "WoE",
                "font": {
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 14,
                },
            },
            tickfont={
                    "family": "Courier New", "color": "rgb(128, 128, 128)",
                    "size": 12
            }
        )

        return self._image