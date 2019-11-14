# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from skcredit.feature_discrete.DiscreteWeightedTreeSplit import tree_split
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

    if col_type == "numeric":
        lower_bin = x.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
        upper_bin = x.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)

        cnt_rec = x.groupby(col + "_bin")["sample_weight"].agg(sum).to_frame("CntRec").reset_index(drop=True)
        cnt_positive = x.loc[x["target"] == 1, [col + "_bin", "sample_weight"]].groupby(
            col + "_bin")["sample_weight"].agg(sum).to_frame("CntPositive").reset_index(drop=True)
        cnt_negative = x.loc[x["target"] == 0, [col + "_bin", "sample_weight"]].groupby(
            col + "_bin")["sample_weight"].agg(sum).to_frame("CntNegative").reset_index(drop=True)

        part_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_positive, cnt_negative], axis=1)
    else:
        cnt_rec = x.groupby(col)["sample_weight"].agg(sum).to_frame("CntRec")
        cnt_positive = x.loc[x["target"] == 1, [col, "sample_weight"]].groupby(
            col)["sample_weight"].agg(sum).to_frame("CntPositive")
        cnt_negative = x.loc[x["target"] == 0, [col, "sample_weight"]].groupby(
            col)["sample_weight"].agg(sum).to_frame("CntNegative")

        cnt_positive = cnt_positive.reindex(index=cnt_rec.index)
        cnt_negative = cnt_negative.reindex(index=cnt_rec.index)

        part_table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)

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

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)

    num_columns = [
        "Lower", "Upper", "CntRec", "CntPositive", "CntNegative", "PositiveRate", "NegativeRate", "WoE", "IV"]
    cat_columns = [
        "CntRec", "CntPositive", "CntNegative", "PositiveRate", "NegativeRate", "WoE", "IV"]

    if col_type == "numeric":
        if x_mis.empty is True:
            table = calc_part_table(x_non, col, col_type)

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=num_columns).reset_index(drop=True)
            table.loc[0, "Lower"], table.loc[table.shape[0] - 1, "Upper"] = -np.inf, np.inf
        else:
            non_table = calc_part_table(x_non, col, col_type)
            mis_table = calc_part_table(x_mis, col, col_type)
            table = pd.concat([non_table.reindex(columns=num_columns), mis_table.reindex(columns=num_columns)])

            table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
            table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

            table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
            table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

            table = table.reindex(columns=num_columns).reset_index(drop=True)
            table.loc[0, "Lower"], table.loc[table.shape[0] - 2, "Upper"] = -np.inf, np.inf
    else:
        table = calc_part_table(x, col, col_type)

        table["CntPositive"] = table["CntPositive"].replace({0: 0.5})
        table["CntNegative"] = table["CntNegative"].replace({0: 0.5})

        table["PositiveRate"] = table["CntPositive"] / table["CntPositive"].sum()
        table["NegativeRate"] = table["CntNegative"] / table["CntNegative"].sum()

        table["WoE"] = np.log(table["PositiveRate"] / table["NegativeRate"])
        table["IV"] = (table["PositiveRate"] - table["NegativeRate"]) * table["WoE"]

        table = table.reindex(columns=cat_columns).reset_index(drop=False)

    return table


def merge_num_table(X, col, split_bin):
    """
    :param X:
    :param col:
    :param split_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)
    table = pd.DataFrame

    # split bin
    for split_bin in np.arange(split_bin, 1, 0.015):
        x_non[col + "_bin"] = tree_split(x_non, col, split_bin=split_bin)
        x_mis[col + "_bin"] = -9999
        x = pd.concat([x_non, x_mis[x_non.columns]])

        table = calc_table(x, col, "numeric")
        if (table.loc[table["Upper"] != -9999, "WoE"].is_monotonic_increasing or
                table.loc[table["Upper"] != -9999, "WoE"].is_monotonic_decreasing):
            break

    logging.info(col + " complete !")

    return table


def merge_cat_table(X, col, merge_bin):
    """    :param X:
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

    # table = calc_table(x, col, "categorical")

    # mis_table = table.loc[table[col] == "missing", :].copy(deep=True)
    # non_table = table.loc[table[col] != "missing", :].sort_values(
    #     by="WoE", ascending=True).reset_index(
    #     drop=True).copy(deep=True)
    #
    # merge_flag = non_table["WoE"].diff().min()
    # while merge_flag <= merge_gap:
    #     idx = list(non_table["WoE"].diff()).index(merge_flag)
    #
    #     x = x.replace({non_table.loc[idx - 1, col]: (non_table.loc[idx - 1, col] + ", " + non_table.loc[idx, col])})
    #     x = x.replace({non_table.loc[idx, col]: (non_table.loc[idx - 1, col] + ", " + non_table.loc[idx, col])})
    #
    #     table = calc_table(x, col, "categorical")
    #     mis_table = table.loc[table[col] == "missing", :].copy(deep=True)
    #     non_table = table.loc[table[col] != "missing", :].sort_values(
    #         by="WoE", ascending=True).reset_index(
    #         drop=True).copy(deep=True)
    #     merge_flag = non_table["WoE"].diff().min()
    # table = pd.concat([mis_table, non_table.reindex(columns=mis_table.columns)]).reset_index(drop=True)

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
