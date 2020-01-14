# coding: utf-8

import copy
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LMCreditcard(object):
    @staticmethod
    def score_alignment(discrete, lmn):
        pass

    @staticmethod
    def attribute_alignment(discrete, lmclassifier, feature, target):
        feature = discrete.transform(feature)
        print(feature.head())

        tables = dict()
        tables.update(discrete.cat_table_)
        tables.update(discrete.num_table_)

        result = dict()

        for col in lmclassifier.feature_subsets_:
            table = copy.deepcopy(tables[col][[col, "WoE", "CntRec", "CntPositive", "CntNegative"]])

            e_cnt_positive = lmclassifier.predict_proba(feature)["proba_positive"].groupby(
                feature[col]).sum().to_frame("ECntPositive")
            e_cnt_negative = lmclassifier.predict_proba(feature)["proba_negative"].groupby(
                feature[col]).sum().to_frame("ECntNegative")

            table = table.merge(e_cnt_positive, left_on=["WoE"], right_index=True, how="left")
            table = table.merge(e_cnt_negative, left_on=["WoE"], right_index=True, how="left")

            table["Delta"] = lmclassifier.b_ * np.log(
                (table["CntPositive"] / table["CntNegative"]) / (table["ECntPositive"] / table["ECntNegative"]))

            result[col] = table

        return result

