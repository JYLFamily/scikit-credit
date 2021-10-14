# coding:utf-8

import pandas as pd
from portion import Interval as BaseInterval
from portion.const import Bound,  _Singleton, _NInf, _PInf


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

    def __repr__(self):
        return "nan"

    def __hash__(self):
        return hash(float("nan"))


NInf = _NInf()
PInf = _PInf()
NaN  =  _NaN()


def singleton():
    return


class CustInterval(BaseInterval):
    def __repr__(self):
        pass





