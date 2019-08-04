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


def chisq(X):
    x = X.copy(deep=True)
    del X
    gc.collect()

    chi2, _, _, _ = chi2_contingency(x)
    return chi2


def chi_merge(X, col, max_bins, min_samples_bins, **kwargs):
    x = X.copy(deep=True)
    del X
    gc.collect()

    if "group_list" in kwargs.keys():
        group_list = kwargs["group_list"]
    else:
        if x[col].nunique() <= 100:
            group_list = [[value] for value in sorted(x[col].unique().tolist())]
        else:
            # 使用分位数上限制代替原始值
            group_list = [[value] for value in np.quantile(x[col], np.linspace(0.01, 1, 100)).tolist()]
            x[col] = x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0]).tolist()

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
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_list.append(chisq(observed))

        idx = chisq_list.index(min(chisq_list))
        group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
        group_list.remove(group_list[add(idx, 1)])

    # 存在 negative == 0 的 group
    count_list = [regroup.loc[regroup[col].isin(group), "negative"].count() for group in group_list]
    while min(count_list) == 0:
        idx = count_list.index(0)
        if idx == 0:  # 第一箱
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        elif idx == len(count_list) - 1:  # 最后箱
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])
        else:  # 中间箱
            observed_after = pd.DataFrame(np.vstack((   # 后一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_after = chisq(observed_after)

            observed_before = pd.DataFrame(np.vstack((  # 前一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[sub(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_before = chisq(observed_before)

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        count_list = [regroup.loc[regroup[col].isin(group), "negative"].count() for group in group_list]

    # 存在 positive == 0 的 group
    count_list = [regroup.loc[regroup[col].isin(group), "positive"].count() for group in group_list]
    while min(count_list) == 0:
        idx = count_list.index(0)
        if idx == 0.:  # 第一箱
            group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
            group_list.remove(group_list[add(idx, 1)])
        elif idx == len(count_list) - 1:  # 最后箱
            group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
            group_list.remove(group_list[sub(idx, 1)])
        else:  # 中间箱
            observed_after = pd.DataFrame(np.vstack((   # 后一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_after = chisq(observed_after)

            observed_before = pd.DataFrame(np.vstack((  # 前一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[sub(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_before = chisq(observed_before)

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        count_list = [regroup.loc[regroup[col].isin(group), "positive"].count() for group in group_list]

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
            observed_after = pd.DataFrame(np.vstack((   # 后一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[add(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_after = chisq(observed_after)

            observed_before = pd.DataFrame(np.vstack((  # 前一箱
                regroup.loc[regroup[col].isin(group_list[idx]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True),
                regroup.loc[regroup[col].isin(group_list[sub(idx, 1)]), ["positive", "negative"]].to_numpy().sum(
                    axis=1, keepdims=True)
            )))
            chisq_before = chisq(observed_before)

            if chisq_after < chisq_before:
                group_list[idx] = group_list[idx] + group_list[add(idx, 1)]
                group_list.remove(group_list[add(idx, 1)])
            else:
                group_list[idx] = group_list[idx] + group_list[sub(idx, 1)]
                group_list.remove(group_list[sub(idx, 1)])
        count_list = [regroup.loc[regroup[col].isin(group), "total"].sum() for group in group_list]

    return x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0]).tolist(), group_list


def tree_split(X, col, min_samples_bins=0.05):
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_bins)
    clf.fit(x[[col]], x["target"])

    return clf.predict_proba(x[[col]])[:, 1].tolist()


if __name__ == "__main__":
    pass