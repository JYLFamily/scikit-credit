# coding:utf-8

import pandas as pd
from operator import lt, le, gt, ge
from portion.const import Bound, _Singleton, _NInf, _PInf


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


l_bound_op = {Bound.OPEN: gt, Bound.CLOSED: ge}
r_bound_op = {Bound.OPEN: lt, Bound.CLOSED: le}
