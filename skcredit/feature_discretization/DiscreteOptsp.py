# coding:utf-8

import gc
import numpy as np
import pandas as pd
from sympy import Interval
from more_itertools import windowed
from optbinning import OptimalBinning
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def optsp_split(X, col):
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = OptimalBinning(
        name=col, dtype="numerical", prebinning_method="cart", solver="cp")
    clf.fit(x[col], x["target"])

    return pd.Series([Interval.Ropen(l, r) for l, r in windowed([-np.inf] + list(clf.splits) + [+np.inf], 2)], name=col)

