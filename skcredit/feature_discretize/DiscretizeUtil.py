# coding:utf-8

import gc
import numpy as np
import pandas as pd
from skcredit.feature_discretize.DiscretizeChiMerge import chi_merge
from skcredit.feature_discretize.DiscretizeTreeSplit import tree_split
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


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

    cat_columns = ["CntRec", "CntGood", "CntBad", "GoodRate", "BadRate", "WoE", "IV"]
    num_columns = ["Lower", "Upper", "CntRec", "CntGood", "CntBad", "GoodRate", "BadRate", "WoE", "IV"]
    if col_type == "numeric":
        lower_bin = x_non.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
        upper_bin = x_non.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)
        cnt_rec = x_non.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec").reset_index(drop=True)
        cnt_bad = x_non.groupby(col + "_bin")["target"].agg(sum).to_frame("CntBad").reset_index(drop=True)
        non_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_bad], axis=1)
        non_table["CntGood"] = non_table["CntRec"] - non_table["CntBad"]

        non_table["BadRate"] = non_table["CntBad"] / non_table["CntBad"].sum()
        non_table["GoodRate"] = non_table["CntGood"] / non_table["CntGood"].sum()

        non_table["WoE"] = np.log(non_table["GoodRate"] / non_table["BadRate"])
        non_table["IV"] = (non_table["GoodRate"] - non_table["BadRate"]) * non_table["WoE"]
        non_table = non_table[num_columns].sort_values(by="Lower", ascending=True).reset_index(drop=True)
        non_table.loc[0, "Lower"], non_table.loc[non_table.shape[0] - 1, "Upper"] = -np.inf, np.inf

        if x_mis.empty is False:
            lower_bin = x_mis.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
            upper_bin = x_mis.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)
            cnt_rec = x_mis.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec").reset_index(drop=True)
            cnt_bad = x_mis.groupby(col + "_bin")["target"].agg(sum).to_frame("CntBad").reset_index(drop=True)
            mis_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_bad], axis=1)
            mis_table["CntGood"] = mis_table["CntRec"] - mis_table["CntBad"]

            mis_table["CntBad"] = mis_table["CntBad"].replace({0: 0.5})
            mis_table["CntGood"] = mis_table["CntGood"].replace({0: 0.5})

            mis_table["BadRate"] = mis_table["CntBad"] / (non_table["CntBad"].sum() + mis_table["CntBad"].sum())
            mis_table["GoodRate"] = mis_table["CntGood"] / (non_table["CntGood"].sum() + mis_table["CntGood"].sum())

            mis_table["WoE"] = np.log(mis_table["GoodRate"] / mis_table["BadRate"])
            mis_table["IV"] = (mis_table["GoodRate"] - mis_table["BadRate"]) * mis_table["WoE"]
            table = pd.concat([non_table, mis_table[non_table.columns]])
        else:
            table = non_table

        del lower_bin, upper_bin, cnt_rec, cnt_bad
        gc.collect()
    else:
        cnt_rec = x.groupby(col)["target"].agg(len).to_frame("CntRec")
        cnt_bad = x.groupby(col)["target"].agg(sum).to_frame("CntBad")
        table = pd.concat([cnt_rec, cnt_bad], axis=1)
        table["CntGood"] = table["CntRec"] - table["CntBad"]

        table["CntBad"] = table["CntBad"].replace({0: 0.5})
        table["CntGood"] = table["CntGood"].replace({0: 0.5})

        table["BadRate"] = table["CntBad"] / table["CntBad"].sum()
        table["GoodRate"] = table["CntGood"] / table["CntGood"].sum()

        table["WoE"] = np.log(table["GoodRate"] / table["BadRate"])
        table["IV"] = (table["GoodRate"] - table["BadRate"]) * table["WoE"]
        table = table[cat_columns].reset_index(drop=False)

        del cnt_rec, cnt_bad
        gc.collect()

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

    x_non = x.loc[x[col] != -9999, :].copy(deep=True)
    x_mis = x.loc[x[col] == -9999, :].copy(deep=True)

    group_list = None
    chi_table, tree_table = [None for _ in range(2)]

    # chi merge
    for max_bins in np.arange(10, 1, -1):
        if max_bins == 10:
            x_non[col + "_bin"], group_list = chi_merge(x_non, col, max_bins=max_bins, min_samples_bins=0.05)
            x_mis[col + "_bin"] = -9999
        else:
            x_non[col + "_bin"], group_list = chi_merge(
                x_non, col, max_bins=max_bins, min_samples_bins=0.05, group_list=group_list)
            x_mis[col + "_bin"] = -9999
        x = pd.concat([x_non, x_mis[x_non.columns]])

        chi_table = calc_table(x, col, "numeric")
        if (chi_table.loc[chi_table["Upper"] != -9999, "WoE"].is_monotonic_increasing or
                chi_table.loc[chi_table["Upper"] != -9999, "WoE"].is_monotonic_decreasing):
            break
        else:
            continue

    # tree split
    for min_samples_bins in np.arange(0.1, 0.55, 0.05):
        if min_samples_bins == 0.1:
            x_non[col + "_bin"], group_list = tree_split(x_non, col, min_samples_bins=min_samples_bins)
            x_mis[col + "_bin"] = -9999
        else:
            x_non[col + "_bin"], group_list = tree_split(
                x_non, col, min_samples_bins=min_samples_bins, group_list=group_list)
            x_mis[col + "_bin"] = -9999
        x = pd.concat([x_non, x_mis[x_non.columns]])

        tree_table = calc_table(x, col, "numeric")
        if (tree_table.loc[tree_table["Upper"] != -9999, "WoE"].is_monotonic_increasing or
                tree_table.loc[tree_table["Upper"] != -9999, "WoE"].is_monotonic_decreasing):
            break
        else:
            continue

    return chi_table if chi_table["IV"].sum() > tree_table["IV"].sum() else tree_table


def merge_cat_table(X, col):
    """
    :param X:
    :param col:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    table = calc_table(x, col, "categorical")
    gc.collect()

    mis_table = table.loc[table[col] == "missing", :].copy(deep=True)
    non_table = (table.loc[table[col] != "missing", :]
                 .sort_values(by="WoE", ascending=True)
                 .reset_index(drop=True)
                 .copy(deep=True))

    merge_flag = non_table["WoE"].diff().min()
    while merge_flag <= 0.2:
        idx = list(non_table["WoE"].diff()).index(merge_flag)

        x = x.replace({non_table.loc[idx - 1, col]: (non_table.loc[idx - 1, col] + ", " + non_table.loc[idx, col])})
        x = x.replace({non_table.loc[idx, col]: (non_table.loc[idx - 1, col] + ", " + non_table.loc[idx, col])})

        table = calc_table(x, col, "categorical")
        mis_table = table.loc[table[col] == "missing", :].copy(deep=True)
        non_table = (table.loc[table[col] != "missing", :]
                     .sort_values(by="WoE", ascending=True)
                     .reset_index(drop=True)
                     .copy(deep=True))
        merge_flag = non_table["WoE"].diff().min()
    table = pd.concat([non_table, mis_table]).reset_index(drop=True)

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
