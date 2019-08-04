# coding:utf-8

import gc
import numpy as np
import pandas as pd
from skcredit.feature_discretize.DiscretizeMethod import chi_merge, tree_split
np.random.seed(7)
pd.set_option("max_row", None)
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

    columns = ["Lower", "Upper", "CntRec", "CntGood", "CntBad", "GoodRate", "BadRate", "WoE", "IV"]
    if col_type == "numeric":
        lower_bin = x_non.groupby(col + "_bin")[col].min().to_frame("Lower").reset_index(drop=True)
        upper_bin = x_non.groupby(col + "_bin")[col].max().to_frame("Upper").reset_index(drop=True)
        cnt_rec = x_non.groupby(col + "_bin")["target"].agg(len).to_frame("CntRec").reset_index(drop=True)
        cnt_bad = x_non.groupby(col + "_bin")["target"].agg(sum).to_frame("CntBad").reset_index(drop=True)
        non_table = pd.concat([lower_bin, upper_bin, cnt_rec, cnt_bad], axis=1)
        non_table["CntGood"] = non_table["CntRec"] - non_table["CntBad"]

        non_table["CntBad"] = non_table["CntBad"].replace({0: 0.5})
        non_table["CntGood"] = non_table["CntGood"].replace({0: 0.5})

        non_table["BadRate"] = non_table["CntBad"] / non_table["CntBad"].sum()
        non_table["GoodRate"] = non_table["CntGood"] / non_table["CntGood"].sum()

        non_table["WoE"] = np.log(non_table["GoodRate"] / non_table["BadRate"])
        non_table["IV"] = (non_table["GoodRate"] - non_table["BadRate"]) * non_table["WoE"]
        non_table = non_table[columns].sort_values(by="Lower", ascending=True).reset_index(drop=True)

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

            table = pd.concat([non_table, mis_table], axis=0, sort=False)
            table.loc[0, "Lower"], table.loc[table.shape[0] - 2, "Upper"] = -np.inf, np.inf
            print(table)
        else:
            table = non_table
            table.loc[0, "Lower"], table.loc[table.shape[0] - 1, "Upper"] = -np.inf, np.inf

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
        table = table[columns].reset_index(drop=True)

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

    group_list = None  # 保留 chi merge 中间结果
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
        x = pd.concat([x_non, x_mis], axis=0, sort=False)

        chi_table = calc_table(x, col, "numeric")
        if chi_table["WoE"][:-1].is_monotonic_increasing or chi_table["WoE"][:-1].is_monotonic_decreasing:
            break
        else:
            continue

    # tree split
    for min_samples_bins in np.arange(0.05, 0.55, 0.025):
        x_non[col + "_bin"] = tree_split(x_non, col, min_samples_bins=min_samples_bins)
        x_mis[col + "_bin"] = -9999
        x = pd.concat([x_non, x_mis], axis=0, sort=False)

        tree_table = calc_table(x, col, "numeric")
        if tree_table["WoE"][:-1].is_monotonic_increasing or tree_table["WoE"][:-1].is_monotonic_decreasing:
            break
        else:
            continue
    print(chi_table)
    print(tree_table)
    table = chi_table if chi_table["IV"].sum() > tree_table["IV"].sum() else tree_table

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


if __name__ == "__main__":
    train = pd.read_csv("C:\\Users\\15795\\Desktop\\train.csv", encoding="GBK")
    # merge_num_table(
    #     train[["user_gray.contacts_number_statistic.pct_black_ratio", "target"]],
    #     "user_gray.contacts_number_statistic.pct_black_ratio"
    # )
    merge_num_table(
        train[["user_gray.contacts_rfm.call_cnt_be_all", "target"]].fillna(-9999),
        "user_gray.contacts_rfm.call_cnt_be_all"
    )
