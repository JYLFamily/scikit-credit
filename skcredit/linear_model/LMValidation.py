# coding: utf-8

import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
from imblearn.under_sampling import RandomUnderSampler
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def calc_table(lmclassifier, tables, feature, label, proba, col):
    table = copy.deepcopy(tables[col][[col, "WoE"]])

    actual_cnt_positive = label.loc[label == 1].groupby(
        feature.loc[label == 1, col]).agg(len).to_frame("ActualCntPositive")
    actual_cnt_negative = label.loc[label == 0].groupby(
        feature.loc[label == 0, col]).agg(len).to_frame("ActualCntNegative")

    expect_cnt_positive = np.round(proba.groupby(feature[col]).sum().to_frame("ExpectCntPositive"))
    expect_cnt_negative = np.round(proba.groupby(feature[col]).sum().to_frame("ExpectCntNegative"))

    table = table.merge(actual_cnt_positive, left_on=["WoE"], right_index=True, how="left")
    table = table.merge(actual_cnt_negative, left_on=["WoE"], right_index=True, how="left")
    table = table.merge(expect_cnt_positive, left_on=["WoE"], right_index=True, how="left")
    table = table.merge(expect_cnt_negative, left_on=["WoE"], right_index=True, how="left")

    table["ActualOdds"] = table["ActualCntPositive"] / table["ActualCntNegative"]
    table["ExpectOdds"] = table["ExpectCntPositive"] / table["ExpectCntNegative"]
    table["DeltaScore"] = lmclassifier.b_ * np.log(table["ActualOdds"] / table["ExpectOdds"])

    return table.drop(["WoE"], axis=1)


class LMValidation(object):
    @staticmethod
    def intercept_alignment(tra_label, tes_label):
        result = OrderedDict()

        result["tra"] = np.log(tra_label.value_counts()[1] / tra_label.value_counts()[0])
        result["tes"] = np.log(tes_label.value_counts()[1] / tes_label.value_counts()[0])

        return result

    @staticmethod
    def attribute_alignment(discrete, lmclassifier, tra_feature, tra_label, tes_feature, tes_label):
        tables = dict()
        tables.update(discrete.cat_table_)
        tables.update(discrete.num_table_)

        result = OrderedDict()
        result["tra"] = dict()
        result["tes"] = dict()

        # tra
        tra_proba = lmclassifier.predict_proba(tra_feature)["proba_positive"]

        with Pool(mp.cpu_count() - 2) as pool:
            result["tra"] = dict(zip(lmclassifier.feature_subsets_, pool.starmap(
                calc_table,
                [(lmclassifier, tables, tra_feature, tra_label, tra_proba, col)
                 for col in lmclassifier.feature_subsets_])))

        # tes
        # tra_label.mean() > tes_label.mean() under-sample majority class
        # tra_label.mean() < tes_label.mean() under-sample minority class
        tes_cnt_negative, tes_cnt_positive = tes_label.value_counts()[0], tes_label.value_counts()[1],

        rus = (
            RandomUnderSampler({0: int(tes_cnt_positive / tra_label.mean()), 1: tes_cnt_positive}, random_state=7)
            if tra_label.mean() > tes_label.mean() else
            RandomUnderSampler({0: tes_cnt_negative, 1: int(tes_cnt_negative * tra_label.mean())}, random_state=7)
        )
        rus.fit(tes_feature, tes_label)

        tes_feature = tes_feature.iloc[rus.sample_indices_, :].reset_index(drop=True)
        tes_label = tes_label.iloc[rus.sample_indices_].reset_index(drop=True)
        tes_proba = lmclassifier.predict_proba(tes_feature)["proba_positive"]

        with Pool(mp.cpu_count() - 2) as pool:
            result["tes"] = dict(zip(lmclassifier.feature_subsets_, pool.starmap(
                calc_table,
                [(lmclassifier, tables, tes_feature, tes_label, tes_proba, col)
                 for col in lmclassifier.feature_subsets_])))

        return result

