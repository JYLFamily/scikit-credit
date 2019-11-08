# coding:utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def tree_split(X, col, split_bin):
    """
    :param X:
    :param col:
    :param split_bin:
    :return:
    """
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = DecisionTreeClassifier(min_weight_fraction_leaf=split_bin, random_state=7)
    clf.fit(x[[col]], x["target"], sample_weight=x["sample_weight"].to_numpy())

    return clf.apply(x[[col]]).tolist()



