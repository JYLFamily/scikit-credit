# coding:utf-8

import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def tree_split(X, col, min_samples_bins):
    x = X.copy(deep=True)
    del X
    gc.collect()

    min_element, max_element = x[col].min(), x[col].max()

    # min_impurity_split will be removed in 0.25 此处设置防止叶子节点不存在 positive or negative 样本
    clf = DecisionTreeClassifier(min_impurity_split=1e-7, min_samples_leaf=min_samples_bins, random_state=7)
    clf.fit(x[[col]], x["target"])

    group_list = [[element] for element in np.sort(clf.tree_.threshold) if element != _tree.TREE_UNDEFINED]  # sort
    group_list = [[min_element]] + group_list + [[max_element]]

    return x[col].apply(lambda i: [max(g) for g in group_list if i <= max(g)][0]).tolist(), group_list



