# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.feature_discretization.Split import Split, Node
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_cat_prebin(x, y):
    if (x.empty and y.empty) or np.all(x ==  x[0]) or np.all(y == y[0]):
        return {key: None for key in np.unique(x)}

    return y.groupby(x).agg(lambda group: round(np.log((0.0005 if (temp := group.eq(1).sum()) == 0 else temp) /
                                                       (0.0005 if (temp := group.eq(0).sum()) == 0 else temp)),
                                                5)).to_dict()


class SplitCat(Split):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.0001):
        super().__init__(
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

    def fit( self,  x, y):
        super().fit(x, y)

        xy = pd.concat([x.to_frame(self.column), y.to_frame(self.target)], axis=1)
        xy_non = xy.loc[~xy[self.column].isna(), :].reset_index(drop=True)
        xy_mis = xy.loc[ xy[self.column].isna(), :].reset_index(drop=True)

        self.all_cnt_negative_non = xy_non[self.target].tolist().count(0)
        self.all_cnt_positive_non = xy_non[self.target].tolist().count(1)
        self.all_cnt_negative_mis = xy_mis[self.target].tolist().count(0)
        self.all_cnt_positive_mis = xy_mis[self.target].tolist().count(1)

        prebin = get_cat_prebin(xy_non[self.column], xy_non[self.target])

        # non-missing
        self._calc_table_non(
            xy_non,
            prebin,
            self.all_cnt_negative_non,
            self.all_cnt_positive_non,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non),
            float('-inf'), float('+inf'))

        #     missing
        self._calc_table_mis(
            {np.nan: None},
            self.all_cnt_negative_mis,
            self.all_cnt_positive_mis,
            *self._stats(self.all_cnt_negative_non, self.all_cnt_positive_non))

        # non missing & missing
        self.table = pd.DataFrame.from_records(self.datas)

        return self

    def _calc_table_non(self, xy_non, prebin, cnt_negative, cnt_positive, woe, ivs, min_value, max_value):
        info = self._split(   xy_non, prebin, ivs, min_value, max_value)

        if info.split is None:
            self.datas.append({
                "Column":        self.column,
                "Bucket": set(prebin.keys()),
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            })
            return

        midd = (info.xy_l_woe_non + info.xy_r_woe_non) / 2

        # 默认 increasing 因为进行了 WoE 转换
        self._calc_table_non(
            info.xy_l_non,
            {key: val for key, val in prebin.items() if val <= info.split},
            info.xy_l_cnt_negative_non, info.xy_l_cnt_positive_non, info.xy_l_woe_non, info.xy_l_ivs_non,
            min_value, midd)
        self._calc_table_non(
            info.xy_r_non,
            {key: val for key, val in prebin.items() if val >  info.split},
            info.xy_r_cnt_negative_non, info.xy_r_cnt_positive_non, info.xy_r_woe_non, info.xy_r_ivs_non,
            midd, max_value)

    def _calc_table_mis(self, bucket, cnt_negative, cnt_positive, woe, ivs):
        self.datas.append({
                "Column":        self.column,
                "Bucket": set(bucket.keys()),
                "CntPositive": cnt_positive,
                "CntNegative": cnt_negative,
                "WoE": woe,
                "IvS": ivs
            })

    def _split(self, xy_non, prebin, ivs, min_value, max_value):
        largest_ivs_gain = 0.0

        best_split = None
        best_xy_l_non = None
        best_xy_r_non = None
        best_xy_l_cnt_negative_non = None
        best_xy_l_cnt_positive_non = None
        best_xy_r_cnt_negative_non = None
        best_xy_r_cnt_positive_non = None
        best_xy_l_woe_non = None
        best_xy_r_woe_non = None
        best_xy_l_ivs_non = None
        best_xy_r_ivs_non = None

        for temp_split in prebin.values():
            temp_xy_l_non = xy_non.loc[
                            xy_non[self.column].isin({key for key, val in prebin.items() if val <= temp_split}), :]
            temp_xy_r_non = xy_non.loc[
                            xy_non[self.column].isin({key for key, val in prebin.items() if val >  temp_split}), :]

            temp_xy_l_cnt_negative_non = temp_xy_l_non[self.target].tolist().count(0)
            temp_xy_l_cnt_positive_non = temp_xy_l_non[self.target].tolist().count(1)
            temp_xy_r_cnt_negative_non = temp_xy_r_non[self.target].tolist().count(0)
            temp_xy_r_cnt_positive_non = temp_xy_r_non[self.target].tolist().count(1)

            if (temp_xy_l_cnt_negative_non >= self.min_bin_cnt_positive and
                    temp_xy_l_cnt_positive_non >= self.min_bin_cnt_negative and
                    temp_xy_r_cnt_negative_non >= self.min_bin_cnt_positive and
                    temp_xy_r_cnt_positive_non >= self.min_bin_cnt_negative):

                temp_xy_l_woe_non, temp_xy_l_ivs_non = self._stats(
                    temp_xy_l_cnt_negative_non, temp_xy_l_cnt_positive_non)
                temp_xy_r_woe_non, temp_xy_r_ivs_non = self._stats(
                    temp_xy_r_cnt_negative_non, temp_xy_r_cnt_positive_non)

                if temp_xy_l_ivs_non + temp_xy_r_ivs_non - ivs > max(
                        self.min_information_value_split_gain, largest_ivs_gain):

                    if (min_value <= temp_xy_l_woe_non <= max_value and
                            min_value <= temp_xy_r_woe_non <= max_value and
                            temp_xy_l_woe_non <= temp_xy_r_woe_non):

                        largest_ivs_gain = temp_xy_l_ivs_non + temp_xy_r_ivs_non - ivs

                        best_split = temp_split
                        best_xy_l_non = temp_xy_l_non
                        best_xy_r_non = temp_xy_r_non
                        best_xy_l_cnt_negative_non = temp_xy_l_cnt_negative_non
                        best_xy_l_cnt_positive_non = temp_xy_l_cnt_positive_non
                        best_xy_r_cnt_negative_non = temp_xy_r_cnt_negative_non
                        best_xy_r_cnt_positive_non = temp_xy_r_cnt_positive_non
                        best_xy_l_woe_non = temp_xy_l_woe_non
                        best_xy_r_woe_non = temp_xy_r_woe_non
                        best_xy_l_ivs_non = temp_xy_l_ivs_non
                        best_xy_r_ivs_non = temp_xy_r_ivs_non

        return Info(best_split, best_xy_l_non, best_xy_r_non,
                    best_xy_l_cnt_negative_non, best_xy_l_cnt_positive_non,
                    best_xy_r_cnt_negative_non, best_xy_r_cnt_positive_non,
                    best_xy_l_woe_non, best_xy_r_woe_non,
                    best_xy_l_ivs_non, best_xy_r_ivs_non)

    def transform( self, x):
        x_transformed = x.apply(lambda element: self._transform(element))

        return x_transformed

    def _transform(self, x):
        for bucket, woe in zip(self.table["Bucket"],  self.table["WoE"]):
            if x in bucket:
                return woe
        else:
            # test 出现了 train 中没有出现的类别使用 train 中最大的（风险最高的） 类别 WoE 替换
            return self.table["WoE"].max()

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)


def binning_cat(x,  y):
    sc = SplitCat()
    sc.fit(x, y)
    return sc


def replace_cat(x, sc):
    return sc.transform(x)
