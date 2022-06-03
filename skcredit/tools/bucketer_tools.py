# coding:utf-8

import pandas as pd
import numpy  as np
from scipy.stats   import   spearmanr
from operator  import  lt, le, gt, ge
from portion.const  import Bound,  _Singleton,  _NInf,  _PInf
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
    if (x.empty or y.empty) or  y.nunique() <= 1:
        return []

    return x.unique().tolist() if x.nunique() <= 128 else (np.unique(np.sort(x.to_numpy())
        [[i for i in range(x.shape[0] // 128, x.shape[0], x.shape[0] // 128)]])).tolist( )


def get_direct(x, y):
    if (x.empty or y.empty) or  y.nunique() <= 1:
        return "increasing"

    return  "increasing"  if  spearmanr(x, y)[0] > 0 else "decreasing"


def calc_stats(sub_cnt_positive,  sub_cnt_negative,  all_cnt_positive,  all_cnt_negative):
    if not sub_cnt_negative and not sub_cnt_positive:
        return 0, 0

    sub_cnt_positive = 0.0005 if not sub_cnt_positive else sub_cnt_positive
    sub_cnt_negative = 0.0005 if not sub_cnt_negative else sub_cnt_negative

    positive_rate = sub_cnt_positive / all_cnt_positive
    negative_rate = sub_cnt_negative / all_cnt_negative

    woe = np.log(positive_rate  /  negative_rate)
    ivs = (positive_rate - negative_rate)  *  woe

    return woe, ivs
