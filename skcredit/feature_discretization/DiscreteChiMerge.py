# coding:utf-8

import gc
import numpy as np
import pandas as pd
from operator import *
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def chisq(X):
    x = X.copy(deep=True)
    del X
    gc.collect()

    r = [x.iloc[0, :].sum(), x.iloc[1, :].sum()]
    c = [x.iloc[:, 0].sum(), x.iloc[:, 1].sum()]
    n = x.to_numpy().sum()
    chi2 = 0.0

    for row in np.arange(2):
        for col in np.arange(2):
            expect = r[row] * c[col] / n
            actual = x.loc[row, col]
            expect = 0.5 if expect < 0.5 else expect
            chi2 += pow((actual - expect), 2) / expect

    return chi2


def group(X, col, merge_bin):
    x = X.copy(deep=True)
    del X
    gc.collect()

    min_element, max_element = x[col].min(), x[col].max()
    print(min_element, max_element)

    clf = DecisionTreeClassifier(min_samples_leaf=merge_bin, random_state=7)
    clf.fit(x[[col]], x["target"])
    # from sklearn.tree import _tree
    # _tree.TREE_UNDEFINED == -2
    group_list = [[element] for element in np.sort(clf.tree_.threshold) if element != -2]  # sort
    group_list = [[min_element]] + group_list + [[max_element]]

    return group_list


def check(X, col, idx, group_list):
    x = X.copy(deep=True)
    del X
    gc.collect()

    if idx == 0:  # 第一箱
        group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
        group_list.remove(group_list[add(idx, 1)])
    elif idx == len(group_list) - 1:  # 最后箱
        group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
        group_list.remove(group_list[sub(idx, 1)])
    else:  # 中间箱
        observed_after = pd.DataFrame(np.vstack((  # 后一箱
            x.loc[x[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                axis=0, keepdims=True),
            x.loc[x[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                axis=0, keepdims=True)
        )))
        chisq_after = chisq(observed_after)

        observed_before = pd.DataFrame(np.vstack((  # 前一箱
            x.loc[x[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                axis=0, keepdims=True),
            x.loc[x[col].isin(group_list[sub(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                axis=0, keepdims=True)
        )))
        chisq_before = chisq(observed_before)

        if chisq_after < chisq_before:
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        else:
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])

    return group_list


def chi_merge(X, col, max_bins, merge_bin, **kwargs):
    x = X.copy(deep=True)
    del X
    gc.collect()

    if "group_list" in kwargs.keys():
        group_list = kwargs["group_list"]
    else:
        if x[col].nunique() <= 50:
            group_list = [[value] for value in sorted(x[col].unique())]
        else:
            # 使用分位数上限制代替原始值
            _, group_list = pd.qcut(x[col], q=50, retbins=True, duplicates="drop")
            group_list = [[value] for value in group_list]
    x[col] = x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0])  # 使用分位数上限制代替原始值

    # total 样本数, positive 样本数, negative 样本数
    regroup = pd.concat([
        x.groupby(col)["target"].count().to_frame("total"),
        x.groupby(col)["target"].sum().to_frame("positive")
    ], axis=1).reset_index()
    regroup["negative"] = regroup["total"] - regroup["positive"]

    # 分箱
    while len(group_list) > max_bins:
        chisq_list = []
        for idx in np.arange(sub(len(group_list), 1)):
            # 列联表
            observed = pd.DataFrame(np.vstack((
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=0, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=0, keepdims=True)
            )))
            chisq_list.append(chisq(observed))

        idx = chisq_list.index(min(chisq_list))
        group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
        group_list.remove(group_list[add(idx, 1)])

    # 存在 negative == 0 的 group
    count_list = [regroup.loc[regroup[col].isin(g), "negative"].count() for g in group_list]
    while min(count_list) == 0:
        idx = count_list.index(0)
        group_list = check(regroup, col, idx, group_list)
        count_list = [regroup.loc[regroup[col].isin(g), "negative"].count() for g in group_list]

    # 存在 positive == 0 的 group
    count_list = [regroup.loc[regroup[col].isin(g), "positive"].count() for g in group_list]
    while min(count_list) == 0:
        idx = count_list.index(0)
        group_list = check(regroup, col, idx, group_list)
        count_list = [regroup.loc[regroup[col].isin(g), "positive"].count() for g in group_list]

    # merge bin 检查
    count_list = [regroup.loc[regroup[col].isin(g), "total"].sum() for g in group_list]
    while truediv(min(count_list), sum(count_list)) < merge_bin:
        idx = count_list.index(min(count_list))
        group_list = check(regroup, col, idx, group_list)
        count_list = [regroup.loc[regroup[col].isin(g), "total"].sum() for g in group_list]

    return x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0]).tolist(), group_list
