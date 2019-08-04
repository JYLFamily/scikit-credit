# coding:utf-8

import gc
import numpy as np
import pandas as pd
from operator import *
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def chi(X):
    x = X.copy(deep=True)
    del X
    gc.collect()

    return np.power((x["expect"] - x["actual"]), 2).sum()


def chi_merge(X, col, max_bins, min_samples_bins, **kwargs):
    x = X.copy(deep=True)
    del X
    gc.collect()

    # total 样本数, actual positive 样本数, actual_ratio positive 后验 target 比例, expect_ratio 先验 target 比例
    regroup = pd.concat([
        x.groupby(col)["target"].count().to_frame("total"),
        x.groupby(col)["target"].sum().to_frame("actual")
    ], axis=1).reset_index()
    regroup["actual_ratio"] = regroup["actual"] / regroup["total"]
    regroup["expect_ratio"] = regroup["actual"].sum() / regroup["total"].sum()
    regroup["expect"] = regroup["total"] * regroup["expect_ratio"]

    if "group_list" in kwargs.keys():
        group_list = kwargs["group_list"]
    else:
        group_list = [[value] for value in sorted(x[col].unique().tolist())]

    # 分箱
    while len(group_list) > max_bins:
        chisq_list = []
        for idx in np.arange(sub(len(group_list), 1)):
            group = group_list[idx] + group_list[add(idx, 1)]
            chisq_list.append(chi(regroup.loc[regroup[col].isin(group)]))

        idx = chisq_list.index(min(chisq_list))
        group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
        group_list.remove(group_list[add(idx, 1)])

    # ratio == 0. or ratio == 1. 检查
    ratio_list = [regroup.loc[regroup[col].isin(group), "actual_ratio"].mean() for group in group_list]
    while min(ratio_list) == 0.:  # 存在全部为 negative 样本的 group
        idx = ratio_list.index(0.)
        if idx == 0.:  # 第一箱
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        elif idx == len(ratio_list) - 1:  # 最后箱
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])
        else:  # 中间箱
            group_after = group_list[idx] + group_list[add(idx, 1)]   # 后一箱
            chisq_after = chi(regroup.loc[regroup[col].isin(group_after)])

            group_before = group_list[idx] + group_list[sub(idx, 1)]  # 前一箱
            chisq_before = chi(regroup.loc[regroup[col].isin(group_before)])

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        ratio_list = [regroup.loc[regroup[col].isin(group), "actual_ratio"].mean() for group in group_list]
    while max(ratio_list) == 1.:  # 全部 positive 样本
        idx = ratio_list.index(1.)
        if idx == 0.:  # 第一箱
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        elif idx == len(ratio_list) - 1:  # 最后箱
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])
        else:  # 中间箱
            group_after = group_list[idx] + group_list[add(idx, 1)]   # 后一箱
            chisq_after = chi(regroup.loc[regroup[col].isin(group_after)])

            group_before = group_list[idx] + group_list[sub(idx, 1)]  # 前一箱
            chisq_before = chi(regroup.loc[regroup[col].isin(group_before)])

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        ratio_list = [regroup.loc[regroup[col].isin(group), "actual_ratio"].mean() for group in group_list]

    # min_samples 检查
    count_list = [regroup.loc[regroup[col].isin(group), "total"].sum() for group in group_list]
    while truediv(min(count_list), sum(count_list)) < min_samples_bins:
        idx = count_list.index(min(count_list))
        if idx == 0.:  # 第一箱
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        elif idx == len(count_list) - 1:  # 最后箱
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])
        else:  # 中间箱
            group_after = group_list[idx] + group_list[add(idx, 1)]   # 后一箱
            chisq_after = chi(regroup.loc[regroup[col].isin(group_after)])

            group_before = group_list[idx] + group_list[sub(idx, 1)]  # 前一箱
            chisq_before = chi(regroup.loc[regroup[col].isin(group_before)])

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        count_list = [regroup.loc[regroup[col].isin(group), "total"].sum() for group in group_list]

    return x[col].apply(lambda i: [max(g) for g in group_list if i in g][0]).tolist(), group_list


def tree_split(X, col, min_samples_bins=0.05):
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_bins)
    clf.fit(x[[col]], x["target"])

    return clf.predict_proba(x[[col]])[:, 1].tolist()


if __name__ == "__main__":
    train = pd.read_csv("D:\\Work\\Data\\WeCash\\train.csv", encoding="GBK")
    print(chi_merge(
        train[["user_gray.contacts_number_statistic.pct_black_ratio", "target"]],
        "user_gray.contacts_number_statistic.pct_black_ratio",
        max_bins=5,
        min_samples_bins=0.05
    ))