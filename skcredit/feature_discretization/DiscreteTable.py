# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def calc_non_table(X, col, lst):
    """
    :param X:
    :param col:
    :param lst:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        for idx, val in lst.items():
            if element in val:
                return idx

    cnt_positive = np.unique(
        np.vectorize(func)(x.loc[x["target"] == 1, col].to_numpy()),
        return_counts=True
    )
    cnt_negative = np.unique(
        np.vectorize(func)(x.loc[x["target"] == 0, col].to_numpy()),
        return_counts=True
    )
    cnt_positive = pd.Series(cnt_positive[1], index=cnt_positive[0])
    cnt_negative = pd.Series(cnt_negative[1], index=cnt_negative[0])
    cnt_rec = cnt_positive.add(cnt_negative, fill_value=0)

    table = pd.concat(
        [
            lst.to_frame(col),
            cnt_rec.to_frame("CntRec"),
            cnt_positive.to_frame("CntPositive"),
            cnt_negative.to_frame("CntNegative"),
        ],
        axis=1
    )
    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})

    return table


def calc_mis_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    cnt_positive = x.loc[x["target"] == 1, [col]].groupby(col).size().to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, [col]].groupby(col).size().to_frame("CntNegative")
    cnt_rec = cnt_positive["CntPositive"].add(cnt_negative["CntNegative"], fill_value=0).to_frame("CntRec")

    table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})
    table = table.reset_index()

    return table


def calc_cat_table(X, col, group_list):
    """
    :param X:
    :param col:
    :param group_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != "missing"].copy(deep=True)
    x_mis = x.loc[x[col] == "missing"].copy(deep=True)

    if x_mis.empty is True:
        non_table = calc_non_table(x_non, col, group_list)
        table = non_table
    else:
        non_table = calc_non_table(x_non, col, group_list)
        mis_table = calc_mis_table(x_mis, col)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

    return table


def calc_num_table(X, col, break_list):
    """
    :param X:
    :param col:
    :param break_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999].copy(deep=True)
    x_mis = x.loc[x[col] == -9999].copy(deep=True)

    if x_mis.empty is True:
        non_table = calc_non_table(x_non, col, break_list)
        table = non_table
    else:
        non_table = calc_non_table(x_non, col, break_list)
        mis_table = calc_mis_table(x_mis, col)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntRec"]
    table["NegativeRate"] = table["CntNegative"] / table["CntRec"]

    table["PositiveCumRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeCumRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveCumRate"] / table["NegativeCumRate"])
    table["IV"] = (table["PositiveCumRate"] - table["NegativeCumRate"]) * table["WoE"]

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
