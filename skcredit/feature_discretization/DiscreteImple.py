# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from sympy import Interval
from operator import le, contains
from skcredit.feature_discretization.DiscreteSplit import dtree_split, dtree_split_cross
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
    a = 1 if 5 < 6 else 10
    cnt_positive = x.loc[x["target"] == 1, col].groupby(
        lambda index: [idx for idx, val in lst.items()
                       if (le(x.at[index, col], val.right)
                           if isinstance(val, Interval) else contains(val, x.at[index, col]))][0]
    ).size().to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, col].groupby(
        lambda index: [idx for idx, val in lst.items()
                       if (le(x.at[index, col], val.right)
                           if isinstance(val, Interval) else contains(val, x.at[index, col]))][0]
    ).size().to_frame("CntNegative")
    cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

    # x["levels"] = x.apply(lambda row: [idx for idx, val in lst.items()
    #                                    if row[col] in val][0], axis=1)
    # cnt_positive = x.loc[x["target"] == 1, [col, "target", "levels"]].groupby(
    #     "levels").size().to_frame("CntPositive").reset_index(drop=True)
    # cnt_negative = x.loc[x["target"] == 0, [col, "target", "levels"]].groupby(
    #     "levels").size().to_frame("CntNegative").reset_index(drop=True)
    # cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

    table = pd.concat([lst.to_frame(col), cnt_rec, cnt_positive, cnt_negative], axis=1)
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
    cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

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

    table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
    table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

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

    table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
    table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

    return table


def merge_cat_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # cat to num
    x_non = x.loc[x[col] != "missing"].copy(deep=True)
    x_non = x_non.reset_index(drop=True)

    weights = (1 / (1 + np.exp(-(x_non.groupby(col).size() - 1))))
    mapping = (1 - weights) * x_non["target"].mean() + weights * x_non.groupby(col)["target"].mean()
    mapping = mapping.to_dict()

    x_non[col] = x_non[col].replace(mapping)

    # break list to group list
    break_list = dtree_split(x_non, col)
    group_list = [[] for _ in break_list]

    for k, v in mapping.items():
        for idx, brk in break_list.items():
            if v in brk:
                group_list[idx].append(k)

    group_list = pd.Series([", ".join(l) for l in group_list])

    # calc cat table
    table = calc_cat_table(x, col, group_list)

    logging.info("{} split complete !".format(col))

    return table


def merge_num_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # num
    x_non = x.loc[x[col] != -9999].copy(deep=True)
    x_non = x_non.reset_index(drop=True)

    # break list
    break_list = dtree_split(x_non, col)

    # calc num table
    table = calc_num_table(x, col, break_list)

    logging.info("{} split complete !".format(col))

    return table


def force_cat_table(X, col, group_list):
    """
    :param X:
    :param col:
    :param group_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    table = calc_cat_table(x, col, group_list)

    logging.info("{} complete !".format(col))

    return table


def force_num_table(X, col, break_list):
    """
    :param X:
    :param col:
    :param break_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    table = calc_num_table(x, col, break_list)

    logging.info("{} complete !".format(col))

    return table


def replace_cat_woe(X, col, group_list, woe):
    """
    :param X:
    :param col:
    :param group_list:
    :param woe:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        if element == "missing":
            return woe[-1]
        else:
            for i, l in enumerate(group_list):
                if element in l:
                    return woe[i]

    x = np.vectorize(func)(x.to_numpy().reshape(-1, ))

    logging.info("{} transform complete !".format(col))

    return x


def replace_num_woe(X, col, break_list, woe):
    """
    :param X:
    :param col:
    :param break_list:
    :param woe:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        if element == "missing":
            return woe[-1]
        else:
            for i, l in enumerate(break_list):
                if element <= l.right:
                    return woe[i]

    x = np.vectorize(func)(x.to_numpy().reshape(-1, ))

    logging.info("{} transform complete !".format(col))

    return x


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

    cnt_positive = x.loc[x["target"] == 1, [col_1, col_2]].groupby(
        lambda index: [idx for idx, (val_1, val_2) in enumerate(zip(lst_1, lst_2))
                       if
                       (
                           le(x.at[index, col_1], val_1.right) and
                           le(x.at[index, col_2], val_2.right)
                       if isinstance(val_1, Interval) and isinstance(val_2, Interval)
                       else
                           x.at[index, col_1] in val_1 and
                           x.at[index, col_2] in val_2)][0]
    ).size().to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, [col_1, col_2]].groupby(
        lambda index: [idx for idx, (val_1, val_2) in enumerate(zip(lst_1, lst_2))
                       if
                       (
                           le(x.at[index, col_1], val_1.right) and
                           le(x.at[index, col_2], val_2.right)
                           if isinstance(val_1, Interval) and isinstance(val_2, Interval)
                           else
                           x.at[index, col_1] in val_1 and
                           x.at[index, col_2] in val_2)][0]
    ).size().to_frame("CntNegative")
    cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

    # x["levels"] = x.apply(lambda row: [idx for idx, (val_1, val_2) in enumerate(zip(lst_1, lst_2))
    #                                    if row[col_1] in val_1 and row[col_2] in val_2][0], axis=1)
    # cnt_positive = x.loc[x["target"] == 1, [col_1, col_2, "target", "levels"]].groupby(
    #     "levels").size().to_frame("CntPositive").reset_index(drop=True)
    # cnt_negative = x.loc[x["target"] == 0, [col_1, col_2, "target", "levels"]].groupby(
    #     "levels").size().to_frame("CntNegative").reset_index(drop=True)
    # cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

    table = pd.concat([lst_1.to_frame(col_1), lst_2.to_frame(col_2), cnt_rec, cnt_positive, cnt_negative], axis=1)
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
    cnt_rec = (cnt_positive["CntPositive"] + cnt_negative["CntNegative"]).to_frame("CntRec")

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

    table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
    table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

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

    table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
    table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

    return table


def merge_cat_table_cross(X, col_1, col_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # cat to num
    x_non = x.loc[(x[col_1] != "missing") & (x[col_2] != "missing")].copy(deep=True)
    x_non = x_non.reset_index(drop=True)

    weights_1 = (1 / (1 + np.exp(-(x_non.groupby(col_1).size() - 1))))
    mapping_1 = (1 - weights_1) * x_non["target"].mean() + weights_1 * x_non.groupby(col_1)["target"].mean()
    mapping_1 = mapping_1.to_dict()

    weights_2 = (1 / (1 + np.exp(-(x_non.groupby(col_2).size() - 1))))
    mapping_2 = (1 - weights_2) * x_non["target"].mean() + weights_2 * x_non.groupby(col_2)["target"].mean()
    mapping_2 = mapping_2.to_dict()

    x_non[col_1] = x_non[col_1].replace(mapping_1)
    x_non[col_2] = x_non[col_2].replace(mapping_2)

    # break list to group list
    break_list_1, break_list_2 = dtree_split_cross(x_non, col_1, col_2)
    group_list_1, group_list_2 = [[] for _ in break_list_1], [[] for _ in break_list_2]

    for k, v in mapping_1.items():
        for idx, brk in break_list_1.items():
            if v in brk:
                group_list_1[idx].append(k)

    for k, v in mapping_2.items():
        for idx, brk in break_list_2.items():
            if v in brk:
                group_list_2[idx].append(k)

    group_list_1 = pd.Series([", ".join(l) for l in group_list_1])
    group_list_2 = pd.Series([", ".join(l) for l in group_list_2])

    # calc cat table
    table = calc_cat_table_cross(x, col_1, col_2, group_list_1, group_list_2)

    logging.info("{} @ {} split complete !".format(col_1, col_2))

    return table


def merge_num_table_cross(X, col_1, col_2):
    """
    :param X:
    :param col_1:
    :param col_2:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # num
    x_non = x.loc[(x[col_1] != -9999) & (x[col_2] != -9999)].copy(deep=True)
    x_non = x_non.reset_index(drop=True)

    # break list
    break_list_1, break_list_2 = dtree_split_cross(x_non, col_1, col_2)

    # calc num table
    table = calc_num_table_cross(x, col_1, col_2, break_list_1, break_list_2)

    logging.info("{} @ {} split complete !".format(col_1, col_2))

    return table


def force_cat_table_cross(X, col_1, col_2, group_list_1, group_list_2):
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

    table = calc_cat_table_cross(x, col_1, col_2, group_list_1, group_list_2)

    logging.info("{} @ {} complete !".format(col_1, col_2))

    return table


def force_num_table_cross(X, col_1, col_2, break_list_1, break_list_2):
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

    table = calc_num_table_cross(x, col_1, col_2, break_list_1, break_list_2)

    logging.info("{} @ {} complete !".format(col_1, col_2))

    return table


def replace_cat_woe_cross(X, col_1, col_2, group_list_1, group_list_2, woe):
    """
    :param X:
    :param col_1
    :param col_2
    :param group_list_1:
    :param group_list_2:
    :param woe:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        if element[0] == "missing" or element[1] == "missing":
            return woe[-1]
        else:
            for i, (l_1, l_2) in enumerate(zip(group_list_1, group_list_2)):
                if element[0] in l_1 and element[1] in l_2:
                    return woe[i]

    x = np.apply_along_axis(func, axis=1, arr=x.to_numpy())

    logging.info("{} @ {} transform complete !".format(col_1, col_2))

    return x


def replace_num_woe_cross(X, col_1, col_2, break_list_1, break_list_2, woe):
    """
    :param X:
    :param col_1
    :param col_2
    :param break_list_1:
    :param break_list_2:
    :param woe:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    def func(element):
        if element[0] == -9999.0 or element[1] == -9999.0:
            return woe[-1]
        else:
            for i, (l_1, l_2) in enumerate(zip(break_list_1, break_list_2)):
                if element[0] <= l_1.right and element[1] <= l_2.right:
                    return woe[i]

    x = np.apply_along_axis(func, axis=1, arr=x.to_numpy())

    logging.info("{} @ {} transform complete !".format(col_1, col_2))

    return x
