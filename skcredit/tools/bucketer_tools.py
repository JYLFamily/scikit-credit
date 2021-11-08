# coding:utf-8

import pandas as pd
import numpy  as np
from scipy.stats  import  spearmanr
from collections import defaultdict
from operator import lt, le, gt, ge
from portion.const import Bound, _Singleton, _NInf, _PInf
from sklearn.base  import BaseEstimator, TransformerMixin
np.random.seed(7)


l_bound_operator = {Bound.OPEN: gt, Bound.CLOSED: ge}
r_bound_operator = {Bound.OPEN: lt, Bound.CLOSED: le}


class _NaN(_Singleton):
    def __neg__(self):
        return _NaN()

    def __lt__(self, o):
        return False

    def __le__(self, o):
        return pd.isna(o) or pd.isnull(o) or isinstance(o, _NaN)

    def __gt__(self, o):
        return False

    def __ge__(self, o):
        return pd.isna(o) or pd.isnull(o) or isinstance(o, _NaN)

    def __eq__(self, o):
        return pd.isna(o) or pd.isnull(o) or isinstance(o, _NaN)

    def __str__(self):
        return "MISSING"

    def __repr__(self):
        return "MISSING"

    def __hash__(self):
        return hash(float("nan"))


NINF = _NInf()
PINF = _PInf()
NAN  =  _NaN()


def get_splits(x, y):
    if (x.empty or y.empty) or (x.nunique() <= 1 or y.nunique() <= 1):
        return []

    return (temp if len(temp := x.unique()) <= 128 else np.histogram_bin_edges(x, bins=128)).tolist()


def get_direct(x, y):
    if (x.empty or y.empty) or (x.nunique() <= 1 or y.nunique() <= 1):
        return "increasing"

    return "increasing" if spearmanr(x, y)[0] > 0 else "decreasing"


def calc_stats(sub_cnt_negative,  sub_cnt_positive, all_cnt_negative, all_cnt_positive):
    if not sub_cnt_negative and not sub_cnt_positive:
        return 0, 0

    sub_cnt_negative = 0.0005 if not sub_cnt_negative else sub_cnt_negative
    sub_cnt_positive = 0.0005 if not sub_cnt_positive else sub_cnt_positive

    negative_rate = sub_cnt_negative / all_cnt_negative
    positive_rate = sub_cnt_positive / all_cnt_positive

    woe = np.log(positive_rate  /  negative_rate)
    ivs = (positive_rate - negative_rate)  *  woe

    return woe, ivs


class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, target):
        self.column = column
        self.target = target
        self.lookup = defaultdict(dict)

    def fit(self,    x, y):
        for column in self.column:
            self.lookup[column] = y.groupby(x[column]).agg(lambda group:
                    round(np.log((0.0005 if (temp := group.eq(1).sum()) == 0 else temp) /
                                 (0.0005 if (temp := group.eq(0).sum()) == 0 else temp)), 5)).to_dict()

        return self

    def transform(self, x):
        x_transformed = x.copy(deep=True)

        for column in self.column:
            x_transformed[column] = x[column].map(self.lookup[column])

        return x_transformed

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)
