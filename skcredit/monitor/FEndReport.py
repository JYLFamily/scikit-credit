# coding: utf-8

import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class FEndReport(object):
    @staticmethod
    def psi(discrete, lmclassifier, tra_feature, tes_feature):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        tra_score = lmclassifier.predict_score(tra_feature)
        tes_score = lmclassifier.predict_score(tes_feature)

        result = dict()

        # np.ceil  向上取整 np.floor 向下取整
        # tra min score [0,    np.ceil(tra_score.min() / 10) * 10]
        # tra max score [np.floor(tra_score.max() / 10) * 10, 999]
        bins = np.append(
            np.array([0]),
            np.arange(np.ceil(tra_score.min() / 10) * 10, np.floor(tra_score.max() / 10) * 10 + 10, 10)
        )
        bins = np.append(bins, np.array([999]))

        table = pd.DataFrame({
            "CntTra": pd.cut(tra_score, bins=bins).value_counts(),
            "CntTes": pd.cut(tes_score, bins=bins).value_counts()
        }).replace({0.0: 0.5})

        table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
        table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
        table["(RatTes - RatTra)"] = table["RatTes"] - table["RatTra"]
        table["LN(RatTes/RatTra)"] = table["RatTes"] / table["RatTra"]
        table = table.reset_index().rename(columns={"index": "score"})

        psi = np.round(np.sum((table["(RatTes - RatTra)"]) * np.log(table["LN(RatTes/RatTra)"])), 5)

        result["table"] = table
        result["score"] = psi

        return result

    @staticmethod
    def csi(discrete, lmclassifier, tra_feature, tes_feature):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        tables = dict()
        tables.update(discrete.cat_table_)
        tables.update(discrete.num_table_)

        coeffs = lmclassifier.coeff_

        result = dict()

        for col in lmclassifier.feature_subsets_:
            table = copy.deepcopy(tables[col][[col, "WoE"]])
            tra_cnt = tra_feature[col].value_counts().to_frame("CntTra").reset_index().rename(columns={"index": "WoE"})
            tes_cnt = tes_feature[col].value_counts().to_frame("CntTes").reset_index().rename(columns={"index": "WoE"})

            table = table.merge(tra_cnt, left_on="WoE", right_on="WoE", how="left")
            table = table.merge(tes_cnt, left_on="WoE", right_on="WoE", how="left")

            # psi
            table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
            table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
            table["(RatTes - RatTra)"] = table["RatTes"] - table["RatTra"]
            table["LN(RatTes/RatTra)"] = table["RatTes"] / table["RatTra"]

            # csi
            table["PScore"] = - table["WoE"] * coeffs[col] * lmclassifier.b_
            table["SScore"] = (table["RatTes"] - table["RatTra"]) * table["PScore"]

            psi = np.round(np.sum((table["(RatTes - RatTra)"]) * np.log(table["LN(RatTes/RatTra)"])), 5)
            csi = np.round(table["SScore"].sum(), 5)

            result[col] = {"table": table, "psi_score": psi, "csi_score": csi}

        return result

    @staticmethod
    def psi_by_week(discrete, lmclassifier, tra_feature, tes_feature):
        week = pd.DataFrame({
            "week": tes_feature[lmclassifier.keep_columns].squeeze().dt.week,
            "date": tes_feature[lmclassifier.keep_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = dict()
        summary["table"] = OrderedDict()
        summary["score"] = OrderedDict()

        for _, row in week.iterrows():
            subset = tes_feature.loc[(tes_feature[lmclassifier.keep_columns].squeeze() >= row["Lower"]) &
                                     (tes_feature[lmclassifier.keep_columns].squeeze() <= row["Upper"])]

            result = FEndReport().psi(discrete, lmclassifier, tra_feature, subset)

            summary["table"]["{}, {}".format(row["Lower"], row["Upper"])] = result["table"]
            summary["score"]["{}, {}".format(row["Lower"], row["Upper"])] = result["score"]

        return summary

    @staticmethod
    def csi_by_week(discrete, lmclassifier, tra_feature, tes_feature):
        week = pd.DataFrame({
            "week": tes_feature[lmclassifier.keep_columns].squeeze().dt.week,
            "date": tes_feature[lmclassifier.keep_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = dict()
        summary["table"] = OrderedDict()
        summary["psi_score"] = OrderedDict()
        summary["csi_score"] = OrderedDict()

        for _, row in week.iterrows():
            subset = tes_feature.loc[(tes_feature[lmclassifier.keep_columns].squeeze() >= row["Lower"]) &
                                     (tes_feature[lmclassifier.keep_columns].squeeze() <= row["Upper"])]

            result = FEndReport().csi(discrete, lmclassifier, tra_feature, subset)

            for col in result.keys():
                summary["table"][("{}, {}".format(row["Lower"], row["Upper"]), col)] = result[col]["table"]
                summary["psi_score"][("{}, {}".format(row["Lower"], row["Upper"]), col)] = result[col]["psi_score"]
                summary["csi_score"][("{}, {}".format(row["Lower"], row["Upper"]), col)] = result[col]["csi_score"]

        return summary





