# coding:utf-8

import gc
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def entropy(X, col):
    x = X.copy(deep=True)
    del X
    gc.collect()

    epy = 0.0
    for frequencies in x[col].value_counts(normalize=True):
        epy -= frequencies * np.log2(frequencies)

    return epy


def conditional_entropy(X, col_x, col_y):
    x = X.copy(deep=True)
    del X
    gc.collect()

    con_epy = []
    for category, frequencies in x[col_x].value_counts(normalize=True).items():
        con_epy.append(frequencies * entropy(x.loc[x[col_x] == category, [col_y]], col_y))

    return con_epy


def information_gain(X, col_x, col_y):
    x = X.copy(deep=True)
    del X
    gc.collect()

    return entropy(x, col_y) - np.sum(conditional_entropy(x, col_x, col_y))


def symmetrical_uncertainty(X, col_x, col_y):
    x = X.copy(deep=True)
    del X
    gc.collect()

    return 2 * (information_gain(x, col_x, col_y) / (entropy(x, col_x) + entropy(x, col_y)))
