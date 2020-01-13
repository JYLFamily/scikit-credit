# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from skcredit.feature_discretization.DiscreteMerge import chisq_merge
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def calc_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    cnt_rec = x.groupby(col)["target"].agg(len).to_frame("CntRec")
    cnt_positive = x.loc[x["target"] == 1, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntNegative")

    table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
    table = table.fillna({"CntPositive": 0.5, "CntNegative": 0.5})
    table = table.reset_index()

    return table


def calc_cat_table(X, col, group_list):
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != "missing"].copy(deep=True)
    x_mis = x.loc[x[col] == "missing"].copy(deep=True)

    x_non[col] = x_non[col].apply(
        lambda element: [", ".join(g) for g in group_list if element in g][0])

    if x_mis.empty is True:
        non_table = calc_table(x_non, col)
        table = non_table
    else:
        non_table = calc_table(x_non, col)
        mis_table = calc_table(x_mis, col)

        table = pd.concat([non_table, mis_table.reindex(columns=non_table.columns)])

    table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
    table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

    table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
    table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

    return table


def calc_num_table(X, col, break_list):
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999].copy(deep=True)
    x_mis = x.loc[x[col] == -9999].copy(deep=True)

    x_non[col] = x_non[col].apply(
        lambda element: [l for l in break_list if element in l][0])

    if x_mis.empty is True:
        non_table = calc_table(x_non, col)
        table = non_table
    else:
        non_table = calc_table(x_non, col)
        mis_table = calc_table(x_mis, col)

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
    break_list = chisq_merge(x_non,  col)
    group_list = [[] for _ in break_list]

    for k, v in mapping.items():
        for idx, brk in enumerate(break_list):
            if v in brk:
                group_list[idx].append(k)

    # calc cat table
    table = calc_cat_table(x, col, group_list)

    logging.info(col + " complete !")

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
    break_list = chisq_merge(x_non, col)

    # calc num table
    table = calc_num_table(x, col, break_list)

    logging.info(col + " complete !")

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

    logging.info(col + " complete !")

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

    logging.info(col + " complete !")

    return table


def replace_num_woe(element, break_list, woe):
    if element == -9999.0:
        return woe[-1]
    else:
        for i, l in enumerate(break_list):
            if element in l:
                return woe[i]


def replace_cat_woe(element, group_list, woe):
    if element == "missing":
        return woe[-1]
    else:
        for i, l in enumerate(group_list):
            if element in l:
                return woe[i]