# coding:utf-8

import gc
import logging
import numpy  as np
import pandas as pd
from functools import reduce
from category_encoders import WOEEncoder
from skcredit.feature_discretization.DiscreteSplit import dtree_split
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def calc_non_table(x, col, lst, spliter):
    cnt_positive = pd.Series(spliter.predict(x.loc[x["target"] == 1, [col]],
          pred_leaf=True, is_reshape=False)).value_counts(sort=False).to_frame("CntPositive")
    cnt_negative = pd.Series(spliter.predict(x.loc[x["target"] == 0, [col]],
          pred_leaf=True, is_reshape=False)).value_counts(sort=False).to_frame("CntNegative")
    cnt_rec = cnt_positive["CntPositive"].add(cnt_negative["CntNegative"]).to_frame("CntRec")

    table = reduce(
        lambda df_1, df_2: pd.merge(df_1, df_2, left_index=True, right_index=True, how="left"),
        [lst, cnt_rec, cnt_positive, cnt_negative])

    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})

    return table


def calc_mis_table(x, col):
    cnt_positive = x.loc[x["target"] == 1, [col]].value_counts(sort=False).to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, [col]].value_counts(sort=False).to_frame("CntNegative")
    cnt_rec = cnt_positive["CntPositive"].add(cnt_negative["CntNegative"]).to_frame("CntRec")

    table = reduce(
        lambda df_1, df_2: pd.merge(df_1, df_2, left_index=True, right_index=True, how="left"),
        [cnt_rec, cnt_positive, cnt_negative])

    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})
    table = table.reset_index().set_index([pd.Index([-1])])

    return table


def calc_cat_table(x_non, x_mis, col):
    # cat to num
    encoder = WOEEncoder(cols=[col], random_state=7)
    encoder.fit(x_non, x_non["target"])
    x_non = encoder.transform(X= x_non)

    # key cat val label
    # encoder.ordinal_encoder.mapping[0]["mapping"])
    # key label val woe
    # encoder.mapping[col]

    spliter, break_list = dtree_split(x_non,    col)

    # num to cat
    group_list = pd.DataFrame()
    group_list = group_list.reindex_like(break_list  ).fillna('')

    for key, val in encoder.ordinal_encoder.mapping[0]["mapping"].items():
        for idx in break_list.index:
            if encoder.mapping[col][val] in break_list.loc[idx, col]:
                if  group_list.loc[idx, col] == '':
                    group_list.loc[idx, col] =  key
                else:
                    group_list.loc[idx, col] += ", {}".format(key)

    if x_mis.empty is True:
        non_table = calc_non_table(x_non, col, group_list, spliter)
        table = non_table
    else:
        non_table = calc_non_table(x_non, col, group_list, spliter)
        mis_table = calc_mis_table(x_mis, col)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

    # return {"table": table, "encoder": encoder, "spliter": spliter}
    return table


def calc_num_table(x_non, x_mis, col):
    """
    :param x_non:
    :param x_mis:
    :param col:
    :return:
    """
    spliter, break_list = dtree_split(x_non, col)

    if x_mis.empty is True:
        non_table = calc_non_table(x_non, col, break_list, spliter)
        table = non_table
    else:
        non_table = calc_non_table(x_non, col, break_list, spliter)
        mis_table = calc_mis_table(x_mis, col)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

    # return {"table": table, "spliter": spliter}
    return table


def calc_non_table_cross(X, col_1, col_2, lst_1, lst_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :param lst_1:
    :param lst_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        for idx, (val_1, val_2) in enumerate(zip(lst_1, lst_2)):
            if element[0] in val_1 and element[1] in val_2:
                return idx

    cnt_positive = np.unique(
        np.apply_along_axis(func, axis=1, arr=x.loc[x["target"] == 1, [col_1, col_2]].to_numpy()),
        return_counts=True
    )
    cnt_negative = np.unique(
        np.apply_along_axis(func, axis=1, arr=x.loc[x["target"] == 0, [col_1, col_2]].to_numpy()),
        return_counts=True
    )
    cnt_positive = pd.Series(cnt_positive[1], index=cnt_positive[0])
    cnt_negative = pd.Series(cnt_negative[1], index=cnt_negative[0])
    cnt_rec = cnt_positive.add(cnt_negative, fill_value=0)

    table = pd.concat(
        [
            lst_1.to_frame(col_1),
            lst_2.to_frame(col_2),
            cnt_rec.to_frame("CntRec"),
            cnt_positive.to_frame("CntPositive"),
            cnt_negative.to_frame("CntNegative"),
        ],
        axis=1
    )
    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})

    return table


def calc_mis_table_cross(X, col_1, col_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    cnt_positive = x.loc[x["target"] == 1, [col_1, col_2]].groupby(
        [col_1, col_2]).size().to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, [col_1, col_2]].groupby(
        [col_1, col_2]).size().to_frame("CntNegative")
    cnt_rec = cnt_positive["CntPositive"].add(cnt_negative["CntNegative"], fill_value=0).to_frame("CntRec")

    table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})
    table = table.reset_index()

    return table


def calc_cat_table_cross(X, col_1, col_2, group_list_1, group_list_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :param group_list_1:
    :param group_list_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[(x[col_1] != "missing") & (x[col_2] != "missing")].copy(deep=True)
    x_mis = x.loc[(x[col_1] == "missing") | (x[col_2] == "missing")].copy(deep=True)

    x_mis[col_1], x_mis[col_2] = "missing", "missing"

    if x_mis.empty is True:
        non_table = calc_non_table_cross(x_non, col_1, col_2, group_list_1, group_list_2)
        table = non_table
    else:
        non_table = calc_non_table_cross(x_non, col_1, col_2, group_list_1, group_list_2)
        mis_table = calc_mis_table_cross(x_mis, col_1, col_2)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

    return table


def calc_num_table_cross(X, col_1, col_2, break_list_1, break_list_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :param break_list_1:
    :param break_list_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[(x[col_1] != -9999) & (x[col_2] != -9999)].copy(deep=True)
    x_mis = x.loc[(x[col_1] == -9999) | (x[col_2] == -9999)].copy(deep=True)

    x_mis[col_1], x_mis[col_2] = -9999, -9999

    if x_mis.empty is True:
        non_table = calc_non_table_cross(x_non, col_1, col_2, break_list_1, break_list_2)
        table = non_table
    else:
        non_table = calc_non_table_cross(x_non, col_1, col_2, break_list_1, break_list_2)
        mis_table = calc_mis_table_cross(x_mis, col_1, col_2)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

    return table
