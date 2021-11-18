# coding:utf-8

import pandas as pd
import numpy  as np
from portion import to_string
from scipy.stats  import  spearmanr
from operator import lt, le, gt, ge
from portion.const import Bound, _Singleton, _NInf, _PInf
from pandas.io.formats.format import _trim_zeros_single_float
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

    if x.nunique() <= 128:
        return x.unique( )

    return np.histogram_bin_edges(x,   bins=128)


def get_direct(x, y):
    if (x.empty or y.empty) or  y.nunique() <= 1:
        return "increasing"

    return "increasing" if spearmanr(x, y)[0] > 0 else "decreasing"


def calc_stats(sub_cnt_negative, sub_cnt_positive, all_cnt_negative, all_cnt_positive):
    if not sub_cnt_negative and not sub_cnt_positive:
        return 0, 0

    sub_cnt_negative = 0.0005 if not sub_cnt_negative else sub_cnt_negative
    sub_cnt_positive = 0.0005 if not sub_cnt_positive else sub_cnt_positive

    negative_rate = sub_cnt_negative / all_cnt_negative
    positive_rate = sub_cnt_positive / all_cnt_positive

    woe = np.log(positive_rate  /  negative_rate)
    ivs = (positive_rate - negative_rate)  *  woe

    return woe, ivs


def cat_bucket_to_string(bucket, lookup):
    return ("{MISSING}" if bucket.lower == bucket.upper == NAN else
            f"{{{', '.join([cat for cat, woe in lookup.items() if woe in bucket])}}}")


def num_bucket_to_string(bucket):
    return to_string(interval=bucket, conv=lambda element: "MISSING" if isinstance(element, _NaN)
        else _trim_zeros_single_float(f"{element:.6f}"), sep=", ")


def format_table_columns(table, cat_columns, cat_encoder):
    table["Bucket"] = table["Bucket"].apply(lambda buckets: ', '.join([
        f"{column}->{cat_bucket_to_string(bucket, cat_encoder.column_woe_lookup[column])}" if column in cat_columns else
        f"{column}->{num_bucket_to_string(bucket)}"
        for column, bucket in buckets.items()]))

    table["CntPositive(%)"] = table["CntPositive(%)"].apply(lambda element: _trim_zeros_single_float(f"{element:.6f}"))
    table["CntNegative(%)"] = table["CntPositive(%)"].apply(lambda element: _trim_zeros_single_float(f"{element:.6f}"))

    table["WoE"] = table["WoE"].apply(lambda element: _trim_zeros_single_float(f"{element:.6f}"))
    table["IvS"] = table["WoE"].apply(lambda element: _trim_zeros_single_float(f"{element:.6f}"))

    return table
