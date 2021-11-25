# coding:utf-8

import pandas as pd
import numpy  as np
from itertools  import  chain
from portion import to_string
from scipy.stats  import  spearmanr
from operator import lt, le, gt, ge
from portion.const  import Bound,  _Singleton,  _NInf,  _PInf
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
    return ("[MISSING]" if bucket.lower == bucket.upper ==  NAN
        else f"[{', '.join([cat for cat, woe in lookup.items() if woe in bucket])}]")


def num_bucket_to_string(bucket):
    return to_string(interval=bucket, conv=lambda element: "MISSING" if isinstance(element, _NaN)
        else _trim_zeros_single_float(f"{element:.3f}"), sep=', ')


def prepare_table( datas, cat_columns, num_columns, all_columns, cat_encoder):
    columns = list(chain(["Idx"], all_columns, ["CntP", "CntN", "PctP", "PctN", "WoE", "IvS"]))
    total = pd.DataFrame(
        data=[["TOTAL", *["-" for _ in all_columns], 0., 0., 0., 0., 0., 0.]], columns=columns)

    table_list = list()

    for datas in datas:
        table = pd.DataFrame(columns=columns)

        for idx, data in enumerate(datas, 1):
            row = pd.DataFrame(columns=columns )

            row.at[0, "Idx"]  = str(idx)

            for column in data["Bucket"].keys():
                if column in cat_columns:
                    row.at[0, column]  =  cat_bucket_to_string(data["Bucket"][column],
                                          cat_encoder.column_woe_lookup[column])
                if column in num_columns:
                    row.at[0, column]  =  num_bucket_to_string(data["Bucket"][column])

            row.at[0, "CntP"] = _trim_zeros_single_float(f"{data['CntPositive']:.3f}")
            row.at[0, "CntN"] = _trim_zeros_single_float(f"{data['CntNegative']:.3f}")

            row.at[0, "PctP"] = _trim_zeros_single_float(f"{data['PctPositive']:.3f}")
            row.at[0, "PctN"] = _trim_zeros_single_float(f"{data['PctNegative']:.3f}")

            row.at[0, "WoE"] = _trim_zeros_single_float(f"{data['WoE']:.3f}")
            row.at[0, "IvS"] = _trim_zeros_single_float(f"{data['IvS']:.3f}")

            table = table.append(row)

            total.at[0, "CntP"] += data['CntPositive']
            total.at[0, "CntN"] += data['CntNegative']
            total.at[0, "PctP"] += data['PctPositive']
            total.at[0, "PctN"] += data['PctNegative']
            total.at[0, "WoE" ] += data['WoE']
            total.at[0, "IvS" ] += data['IvS']

        table_list.append(table)

    total.at[0, "CntP"] = _trim_zeros_single_float(f"{total.at[0, 'CntP']:.3f}")
    total.at[0, "CntN"] = _trim_zeros_single_float(f"{total.at[0, 'CntN']:.3f}")
    total.at[0, "PctP"] = _trim_zeros_single_float(f"{total.at[0, 'PctP']:.3f}")
    total.at[0, "PctN"] = _trim_zeros_single_float(f"{total.at[0, 'PctN']:.3f}")
    total.at[0, "WoE" ] = _trim_zeros_single_float(f"{total.at[0, 'WoE' ]:.3f}")
    total.at[0, "IvS" ] = _trim_zeros_single_float(f"{total.at[0, 'IvS' ]:.3f}")

    table_list = [pd.concat(table_list).reset_index(drop=True).append(total)] + table_list
    return table_list


def prepare_image(datas, cat_columns, num_columns, all_columns, cat_encoder):
    pass
