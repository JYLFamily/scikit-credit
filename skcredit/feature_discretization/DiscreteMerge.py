# coding:utf-8

import gc
import numpy as np
import pandas as pd
from operator import *
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def calc_chisq(X, col, idx, break_list):
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
        x.loc[x[col].isin(break_list[idx]), ["CntPositive", "CntNegative"]].to_numpy().sum(
            axis=0, keepdims=True),
        x.loc[x[col].isin(break_list[add(idx, 1)]), ["CntPositive", "CntNegative"]].to_numpy().sum(
            axis=0, keepdims=True)
    ))

    r = table.sum(axis=1)  # [x.iloc[0, :].sum(), x.iloc[1, :].sum()]
    c = table.sum(axis=0)  # [x.iloc[:, 0].sum(), x.iloc[:, 1].sum()]
    n = table.sum()
    result = 0.0

    for row in range(2):
        for col in range(2):
            expect = r[row] * c[col] / n
            actual = table[row, col]
            expect = 0.5 if expect < 0.5 else expect
            result += pow((actual - expect), 2) / expect

    return result


def exam_break(X, col, idx, break_list):
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
        break_list[idx] = break_list[idx] + break_list[sub(idx, 1)]
        break_list.remove(break_list[sub(idx, 1)])
    # 中间箱
    else:
        # 后一箱
        chisq_after = calc_chisq(x, col, idx, break_list)

        # 前一箱
        chisq_before = calc_chisq(x, col, idx, break_list)

        if chisq_after < chisq_before:
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

    # 初始分箱
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=0.05, random_state=7)
    clf.fit(x[[col]], x["target"])

    # from sklearn.tree import _tree
    # _tree.TREE_UNDEFINED == -2
    break_list = [[element] for element in np.sort(clf.tree_.threshold) if element != -2]  # sort
    break_list = [[- np.inf]] + break_list + [[x[col].max()]]
    x[col] = x[col].apply(
        lambda element: [max(l) for l in break_list if element <= max(l)][0])

    regroup = pd.concat([
        x.groupby(col)["target"].agg(len).to_frame("CntRec"),
        x.loc[x["target"] == 1, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntPositive"),
        x.loc[x["target"] == 0, [col, "target"]].groupby(col)["target"].agg(len).to_frame("CntNegative")
    ], axis=1).reset_index()

    # 分箱过程 [0.900, df=1 2.705543] [0.950, df=1 3.841459] [0.990, df=1 6.634897] [0.999, df=1 10.82757]
    chisq_list = [calc_chisq(regroup, col, idx, break_list) for idx in range(sub(len(break_list), 1))]
    while min(chisq_list) < 10.82757:
        idx = chisq_list.index(min(chisq_list))
        break_list[idx] = break_list[idx] + break_list[add(idx, 1)]
        break_list.remove(break_list[add(idx, 1)])
        chisq_list = [calc_chisq(regroup, col, idx, break_list) for idx in range(sub(len(break_list), 1))]

    # 分箱检查 存在 positive <= 25 的 break
    count_list = [regroup.loc[regroup[col].isin(l), "CntPositive"].sum() for l in break_list]
    while min(count_list) <= 25:
        idx = count_list.index(min(count_list))
        break_list = exam_break(regroup, col, idx, break_list)
        count_list = [regroup.loc[regroup[col].isin(l), "CntPositive"].sum() for l in break_list]

    # 分箱检查 存在 negative <= 25 的 break
    count_list = [regroup.loc[regroup[col].isin(l), "CntNegative"].sum() for l in break_list]
    while min(count_list) <= 25:
        idx = count_list.index(min(count_list))
        break_list = exam_break(regroup, col, idx, break_list)
        count_list = [regroup.loc[regroup[col].isin(l), "CntNegative"].sum() for l in break_list]

    return pd.IntervalIndex.from_breaks([- np.inf] + [max(l) for l in break_list][:-1] + [np.inf])