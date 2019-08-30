# coding:utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def add(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["ADD(" + ", ".join(col) + ")"] = x[col[0]] + x[col[1]]
        return x


def sub(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["SUB(" + ", ".join(col) + ")"] = x[col[0]] - x[col[1]]
        return x


def mul(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["MUL(" + ", ".join(col) + ")"] = x[col[0]] * x[col[1]]
        return x


def div(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["DIV(" + ", ".join(col) + ")"] = np.round(x[col[0]] / x[col[1]].replace({0.: np.nan}), 2)
        return x


def stats(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["STATS(MIN(" + ", ".join(col) + "))"] = x[col].min(axis=1)
            x["STATS(MAX(" + ", ".join(col) + "))"] = x[col].max(axis=1)
            x["STATS(STD(" + ", ".join(col) + "))"] = x[col].std(axis=1)
            x["STATS(SUM(" + ", ".join(col) + "))"] = x[col].sum(axis=1)
            x["STATS(AVG(" + ", ".join(col) + "))"] = x[col].mean(axis=1)
            x["STATS(MED(" + ", ".join(col) + "))"] = x[col].median(axis=1)
        return x


def concat(x, cols):
    if cols is None:
        return x
    else:
        for col in cols:
            x["CONCAT(" + ", ".join(col) + ")"] = x[col].fillna("missing").apply(
                lambda series: "_".join(series.tolist()), axis=1)
        return x


class Generate(BaseEstimator, TransformerMixin):
    def __init__(self, *, calc):
        self.__calc = calc
        self.__primitives = {
            "add": add,
            "sub": sub,
            "mul": mul,
            "div": div,
            "stats": stats,
            "concat": concat
        }

    def fit(self, X, y=None):
        pass

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        for func, cols in self.__calc.items():
            x = self.__primitives[func](x, cols)

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)