# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from skcredit.feature_discretization.DiscreteMerge import group, chi_merge
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def calc_part_table(X, col, col_type):
    """
    :param X:
    :param col:
    :param col_type:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    part_table = pd.DataFrame

    if col_type == "cat":
        cnt_rec = x.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec")
        cnt_positive = x.loc[x["target"] == 1, [col + "_bin", "target"]].groupby(
            col + "_bin")["target"].agg(len).to_frame("CntPositive")
        cnt_negative = x.loc[x["target"] == 0, [col + "_bin", "target"]].groupby(
            col + "_bin")["target"].agg(len).to_frame("CntNegative")

        part_table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
        part_table = part_table.reset_index().rename(columns={col + "_bin": col})

    elif col_type == "num":
        lower_bin = x.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
        upper_bin = x.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)

        cnt_rec = x.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec").reset_index(drop=True)
        cnt_positive = x.loc[x["target"] == 1, [col + "_bin", "target"]].groupby(
            col + "_bin")["target"].agg(len).to_frame("CntPositive").reset_index(drop=True)
        cnt_negative = x.loc[x["target"] == 0, [col + "_bin", "target"]].groupby(
            col + "_bin")["target"].agg(len).to_frame("CntNegative").reset_index(drop=True)

        part_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_positive, cnt_negative], axis=1)

    return part_table


def calc_table(X, col, col_type):
    """
    :param X:
    :param col:
    :param col_type:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    table = pd.DataFrame

    if col_type == "cat":
        x_non = x.loc[x[col + "_bin"] != "missing", :].copy(deep=True)
        x_mis = x.loc[x[col + "_bin"] == "missing", :].copy(deep=True)

        cat_columns = [
            col, "CntRec", "CntPositive", "CntNegative", "PositiveRate", "NegativeRate", "WoE", "IV"]

        if x_mis.empty is True:
            table = calc_part_table(x_non, col, "cat")

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=cat_columns)
        else:
            non_table = calc_part_table(x_non, col, "cat")
            mis_table = calc_part_table(x_mis, col, "cat")
            table = pd.concat([non_table.reindex(columns=cat_columns), mis_table.reindex(columns=cat_columns)])

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=cat_columns)

    elif col_type == "num":
        x_non = x.loc[x[col] != -9999, :].copy(deep=True)
        x_mis = x.loc[x[col] == -9999, :].copy(deep=True)

        num_columns = [
            "Lower", "Upper", "CntRec", "CntPositive", "CntNegative", "PositiveRate", "NegativeRate", "WoE", "IV"]

        if x_mis.empty is True:
            table = calc_part_table(x_non, col, "num")

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=num_columns).reset_index(drop=True)
            table.loc[0, "Lower"], table.loc[table.shape[0] - 1, "Upper"] = -np.inf, np.inf
        else:
            non_table = calc_part_table(x_non, col, "num")
            mis_table = calc_part_table(x_mis, col, "num")
            table = pd.concat([non_table.reindex(columns=num_columns), mis_table.reindex(columns=num_columns)])

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=num_columns).reset_index(drop=True)
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

        table = calc_table(x, col, "num")

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
    x = pd.concat([x_non, x_mis[x_non.columns]])

    # merge num table
    table = merge_num_table(x, col, merge_bin)

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


def force_num_table(X, col, force_bin):
    """
    :param X:
    :param col:
    :param force_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)

    x_non[col + "_bin"] = pd.cut(x_non[col], bins=force_bin, right=False)
    x_mis[col + "_bin"] = -9999
    x = pd.concat([x_non, x_mis[x_non.columns]])

    table = calc_table(x, col, "num")

    logging.info(col + " complete !")

    return table


def force_cat_table(X, col, force_bin):
    """
    :param X:
    :param col:
    :param force_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != "missing", :].copy(deep=True)
    x_mis = x.loc[x[col] == "missing", :].copy(deep=True)

    x_non[col + "_bin"] = x_non[col].apply(lambda i: [", ".join(f) for f in force_bin if i in f][0])
    x_mis[col + "_bin"] = "missing"
    x = pd.concat([x_non, x_mis[x_non.columns]])

    table = calc_table(x, col, "cat")

    logging.info(col + " complete !")

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
