# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from skcredit.feature_discretization.DiscreteChiMerge import group, chi_merge
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def calc_part_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    lower_bin = x.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
    upper_bin = x.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)

    cnt_rec = x.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec").reset_index(drop=True)
    cnt_positive = x.loc[x["target"] == 1, [col + "_bin", "target"]].groupby(
        col + "_bin")["target"].agg(len).to_frame("CntPositive").reset_index(drop=True)
    cnt_negative = x.loc[x["target"] == 0, [col + "_bin", "target"]].groupby(
        col + "_bin")["target"].agg(len).to_frame("CntNegative").reset_index(drop=True)

    part_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_positive, cnt_negative], axis=1)

    return part_table


def calc_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)

    columns = [
        "Lower", "Upper", "CntRec", "CntPositive", "CntNegative", "PositiveRate", "NegativeRate", "WoE", "IV"]

    if x_mis.empty is True:
        table = calc_part_table(x_non, col)

        table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
        table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

        table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
        table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

        table = table.reindex(columns=columns).reset_index(drop=True)
        table.loc[0, "Lower"], table.loc[table.shape[0] - 1, "Upper"] = -np.inf, np.inf
    else:
        non_table = calc_part_table(x_non, col)
        mis_table = calc_part_table(x_mis, col)
        table = pd.concat([non_table.reindex(columns=columns), mis_table.reindex(columns=columns)])

        table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
        table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

        table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
        table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

        table = table.reindex(columns=columns).reset_index(drop=True)
        table.loc[0, "Lower"], table.loc[table.shape[0] - 2, "Upper"] = -np.inf, np.inf

    return table


def merge_num_table(X, col, merge_bin):
    """
    :param X:
    :param col:
    :param merge_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)
    table = pd.DataFrame

    # init group list
    group_list = group(x_non, col, merge_bin=merge_bin)

    # merge bin
    for max_bins in np.arange(10, 1, -1):
        x_non[col + "_bin"], group_list = chi_merge(
            x_non, col, max_bins=max_bins, merge_bin=merge_bin, group_list=group_list)
        x_mis[col + "_bin"] = -9999
        x = pd.concat([x_non, x_mis[x_non.columns]])

        table = calc_table(x, col)
        if (table.loc[table["Upper"] != -9999, "WoE"].is_monotonic_increasing or
                table.loc[table["Upper"] != -9999, "WoE"].is_monotonic_decreasing):
            break

    logging.info(col + " complete !")

    return table


def merge_cat_table(X, col, merge_bin):
    """
    :param X:
    :param col:
    :param merge_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # cat to num
    x_non = x.loc[x[col] != "missing", :].copy(deep=True)
    x_mis = x.loc[x[col] == "missing", :].copy(deep=True)

    weights = 1 / (1 + np.exp(-(x_non.groupby(col).size() - 1)))
    mapping = (1 - weights) * x_non["target"].mean() + weights * x_non.groupby(col)["target"].mean()
    mapping = mapping.to_dict()

    x_non[col] = x_non[col].replace(mapping)
    x_mis[col] = x_mis[col].replace({"missing": -9999.0})

    # merge num table
    table = merge_num_table(pd.concat([x_non, x_mis]), col, merge_bin)

    # clean num table
    level = [[] for _ in range(len(table))]

    non_table = table.loc[table["Upper"] != -9999, :].copy(deep=True)
    mis_table = table.loc[table["Upper"] == -9999, :].copy(deep=True)

    for key, val in mapping.items():
        for index, upper in enumerate(non_table["Upper"]):
            if val <= upper:
                level[index].append(key)
                break

    if not mis_table.empty:
        level[-1].append("missing")

    table = pd.concat([
        pd.Series([", ".join(element) for element in level]).to_frame(col),
        pd.concat([
            non_table.drop(["Lower", "Upper"], axis=1),
            mis_table.drop(["Lower", "Upper"], axis=1)
        ])
    ], axis=1)

    return table


def replace_num_woe(x, upper, woe):
    num = len(upper)

    if x == -9999.0:
        return woe[-1]
    elif x <= upper[0]:
        return woe[0]
    else:
        for i in range(num - 1):
            if upper[i] < x <= upper[i + 1]:
                return woe[i + 1]


def replace_cat_woe(x, categories, woe):
    categories_woe = dict()

    for i, j in zip(categories, woe):
        for k in i.split(", "):
            categories_woe[k] = j

    for i, j in categories_woe.items():
        if x == i:
            return j
