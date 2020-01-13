# coding:utf-8

import gc
import numpy as np
import pandas as pd
from operator import *
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def table_chisq(X, col, idx, break_list):
    """
    :param X: regroup
    :param col:
    :param idx:
    :param break_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    table = np.vstack((
        x.loc[x[col].isin(break_list[add(idx, 0)]), ["CntPositive", "CntNegative"]].to_numpy().sum(
            axis=0, keepdims=True),
        x.loc[x[col].isin(break_list[add(idx, 1)]), ["CntPositive", "CntNegative"]].to_numpy().sum(
            axis=0, keepdims=True)
    ))

    return chi2_contingency(table)[0]


def check_break(X, col, idx, break_list):
    """
    :param X: regroup
    :param col:
    :param idx:
    :param break_list:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    # 第一箱
    if idx == 0:
        break_list[idx] = break_list[idx] + break_list[add(idx, 1)]
        break_list.remove(break_list[add(idx, 1)])
    # 最后箱
    elif idx == len(break_list) - 1:
        break_list[idx] = break_list[sub(idx, 1)] + break_list[idx]
        break_list.remove(break_list[sub(idx, 1)])
    # 中间箱
    else:
        # 后一箱
        chisq_aft = table_chisq(x, col, idx, break_list)

        # 前一箱
        chisq_bef = table_chisq(x, col, sub(idx, 1), break_list)

        if chisq_aft < chisq_bef:
            break_list[idx] = break_list[idx] + break_list[add(idx, 1)]
            break_list.remove(break_list[add(idx, 1)])
        else:
            break_list[idx] = break_list[idx] + break_list[sub(idx, 1)]
            break_list.remove(break_list[sub(idx, 1)])

    return break_list


def chisq_merge(X, col):
    x = X.copy(deep=True)
    del X
    gc.collect()

    # setup
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=0.05, random_state=7)
    clf.fit(x[[col]], x["target"])

    break_list = x.groupby(pd.Series(clf.apply(x[[col]])))[col].max().tolist()
    break_list = [[l] for l in sorted(break_list)]

    x[col] = x[col].apply(
        lambda element: [max(l) for l in break_list if element <= max(l)][0])

    regroup = pd.concat([
        x.groupby(col)["target"].agg(len).to_frame("CntRec"),
        x.loc[x["target"] == 1, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntPositive"),
        x.loc[x["target"] == 0, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntNegative")
    ], axis=1)
    regroup = regroup.fillna({"CntPositive": 0.5, "CntNegative": 0.5})
    regroup = regroup.reset_index()

    # merge
    chisq_list = [table_chisq(regroup, col, idx, break_list) for idx in range(sub(len(break_list), 1))]
    while len(break_list) > 2 and min(chisq_list) <= 10:
        idx = chisq_list.index(min(chisq_list))
        break_list[idx] = break_list[idx] + break_list[add(idx, 1)]
        break_list.remove(break_list[add(idx, 1)])
        chisq_list = [table_chisq(regroup, col, idx, break_list) for idx in range(sub(len(break_list), 1))]

    count_list = [regroup.loc[regroup[col].isin(l), "CntPositive"].sum() for l in break_list]
    while len(break_list) > 2 and min(count_list) <= 25:
        idx = count_list.index(min(count_list))
        break_list = check_break(regroup, col, idx, break_list)
        count_list = [regroup.loc[regroup[col].isin(l), "CntPositive"].sum() for l in break_list]

    count_list = [regroup.loc[regroup[col].isin(l), "CntNegative"].sum() for l in break_list]
    while len(break_list) > 2 and min(count_list) <= 25:
        idx = count_list.index(min(count_list))
        break_list = check_break(regroup, col, idx, break_list)
        count_list = [regroup.loc[regroup[col].isin(l), "CntNegative"].sum() for l in break_list]

    return pd.IntervalIndex.from_breaks([- np.inf] + [max(l) for l in break_list][:-1] + [np.inf])