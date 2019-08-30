# coding:utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from skcredit.util.entropy import information_gain
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def tree_split(X, col, min_samples_bins, **kwargs):
    x = X.copy(deep=True)
    del X
    gc.collect()

    if "group_list" in kwargs.keys():
        group_list = kwargs["group_list"]
    else:
        if x[col].nunique() <= 50:
            group_list = [[value] for value in sorted(x[col].unique())]
        else:
            _, group_list = pd.qcut(x[col], q=50, retbins=True, duplicates="drop")
            group_list = [[value] for value in group_list]
    x[col] = x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0])

    clf = DecisionTreeSplit(col=col, min_samples_bins=min_samples_bins)
    clf.fit(x[[col]], x["target"])

    return clf.predict_proba(x[[col]]), group_list


class DecisionNode(object):
    def __init__(self, threshold=None, leaf_value=None, left_branch=None, right_branch=None):
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.left_branch = left_branch
        self.right_branch = right_branch


class DecisionTreeSplit(BaseEstimator, ClassifierMixin):
    def __init__(self, *, col, min_samples_bins):
        self.__col = col
        self.__n_samples = None
        self.__root_node = None
        self.__min_samples_bins = min_samples_bins

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.__n_samples = x.shape[0]
        self.__root_node = self.__fit(x, y)

        return self

    def __fit(self, X, y):
        x = X.copy(deep=True)
        del X
        gc.collect()

        best_sets = None
        best_threshold = None
        largest_impurity = 0.

        xy = pd.concat([x, y.to_frame("target")], axis=1)
        unique_values = sorted(x.squeeze().unique())

        for threshold in unique_values:
            xy["threshold"] = xy[self.__col].apply(
                lambda element: "left" if element <= threshold else "right")

            # 叶子节点样本数大于等训练集全部样本数 * 0.05 且叶子同时存在 positive 与 negative 样本
            if (np.sum(xy["threshold"] == "left") > self.__n_samples * self.__min_samples_bins and
                    np.sum(xy["threshold"] == "right") > self.__n_samples * self.__min_samples_bins and
                    0 < xy.loc[xy["threshold"] == "left", "target"].mean() < 1 and
                    0 < xy.loc[xy["threshold"] == "right", "target"].mean() < 1):

                impurity = information_gain(xy, "threshold", "target")
                if impurity > largest_impurity:
                    best_sets = {
                        "left_feature": xy.loc[xy["threshold"] == "left", [self.__col]],
                        "left_target": xy.loc[xy["threshold"] == "left", "target"],
                        "right_feature": xy.loc[xy["threshold"] == "right", [self.__col]],
                        "right_target": xy.loc[xy["threshold"] == "right", "target"]
                    }
                    best_threshold = threshold
                    largest_impurity = impurity

        # information gain >= 0, == 0 在程序中代表没有满足 if 判断条件, 作为叶子节点
        if largest_impurity > 0:
            left_branch = self.__fit(best_sets["left_feature"], best_sets["left_target"])
            right_branch = self.__fit(best_sets["right_feature"], best_sets["right_target"])

            return DecisionNode(
                threshold=best_threshold, left_branch=left_branch, right_branch=right_branch)

        leaf_value = y.mean()

        return DecisionNode(leaf_value=leaf_value)

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return x.squeeze().apply(lambda element: self.__predict_proba(element))

    def __predict_proba(self, x, node=None):
        if node is None:
            node = self.__root_node

        if node.leaf_value is not None:
            return node.leaf_value

        if x <= node.threshold:
            node = node.left_branch
        else:
            node = node.right_branch

        return self.__predict_proba(x, node)

