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
            "CntTra": pd.cut(tra_score, bins=bins).value_counts().fillna(0.5),
            "CntTes": pd.cut(tes_score, bins=bins).value_counts().fillna(0.5)
        })
        table = table.reset_index().rename(columns={"index": "scores"})

        table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
        table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
        table["(RatTes - RatTra)"] = table["RatTes"] - table["RatTra"]
        table["LN(RatTes/RatTra)"] = np.log(table["RatTes"] / table["RatTra"])

        score = np.round(np.sum(table["(RatTes - RatTra)"] * table["LN(RatTes/RatTra)"]), 5)

        result["table"] = table
        result["score"] = score

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
            tra_cnt = tra_feature[col].value_counts().to_frame("CntTra").reset_index().rename(
                columns={"index": "WoE"}).fillna(0.5)
            tes_cnt = tes_feature[col].value_counts().to_frame("CntTes").reset_index().rename(
                columns={"index": "WoE"}).fillna(0.5)

            table = table.merge(tra_cnt, left_on="WoE", right_on="WoE", how="left")
            table = table.merge(tes_cnt, left_on="WoE", right_on="WoE", how="left")

            # psi
            table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
            table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
            table["(RatTes - RatTra)"] = table["RatTes"] - table["RatTra"]
            table["LN(RatTes/RatTra)"] = np.log(table["RatTes"] / table["RatTra"])

            psi_score = np.round(np.sum(table["(RatTes - RatTra)"] * table["LN(RatTes/RatTra)"]), 5)

            # csi
            table["PScore"] = - table["WoE"] * coeffs[col] * lmclassifier.b_
            table["SScore"] = (table["RatTes"] - table["RatTra"]) * table["PScore"]

            csi_score = np.round(table["SScore"].sum(), 5)

            result[col] = {"table": table, "psi_score": psi_score, "csi_score": csi_score}

        return result

    @staticmethod
    def psi_by_week(discrete, lmclassifier, tra_feature, tes_feature):
        time = "[{}, {}]".format

        week = pd.DataFrame({
            "week": tes_feature[lmclassifier.keep_columns].squeeze().dt.week,
            "date": tes_feature[lmclassifier.keep_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = OrderedDict()
        summary["table"] = OrderedDict()
        summary["score"] = np.zeros(shape=(len(week), ))

        for i, (_, row) in enumerate(week.iterrows()):
            subset = tes_feature.loc[(tes_feature[lmclassifier.keep_columns].squeeze() >= row["Lower"]) &
                                     (tes_feature[lmclassifier.keep_columns].squeeze() <= row["Upper"])]
            result = FEndReport().psi(discrete, lmclassifier, tra_feature, subset)

            summary["table"][time(row["Lower"], row["Upper"])] = result["table"]
            summary["score"][i] = result["score"]

        summary["score"] = pd.Series(summary["score"], index=[time(i, j) for i, j in zip(week["Lower"], week["Upper"])])

        return summary

    @staticmethod
    def csi_by_week(discrete, lmclassifier, tra_feature, tes_feature):
        time = "[{}, {}]".format

        week = pd.DataFrame({
            "week": tes_feature[lmclassifier.keep_columns].squeeze().dt.week,
            "date": tes_feature[lmclassifier.keep_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = OrderedDict()
        summary["table"] = OrderedDict()
        summary["psi_score"] = np.zeros(shape=(len(lmclassifier.feature_subsets_), len(week)))
        summary["csi_score"] = np.zeros(shape=(len(lmclassifier.feature_subsets_), len(week)))

        for i, (_, row) in enumerate(week.iterrows()):
            subset = tes_feature.loc[(tes_feature[lmclassifier.keep_columns].squeeze() >= row["Lower"]) &
                                     (tes_feature[lmclassifier.keep_columns].squeeze() <= row["Upper"])]
            result = FEndReport().csi(discrete, lmclassifier, tra_feature, subset)

            summary["table"][time(row["Lower"], row["Upper"])] = OrderedDict()

            for j, col in enumerate(result.keys()):
                summary["table"][time(row["Lower"], row["Upper"])][col] = result[col]["table"]
                summary["psi_score"][j, i] = result[col]["psi_score"]
                summary["csi_score"][j, i] = result[col]["csi_score"]

        summary["psi_score"] = pd.DataFrame(summary["psi_score"],
            index=lmclassifier.feature_subsets_, columns=[time(i, j) for i, j in zip(week["Lower"], week["Upper"])])
        summary["csi_score"] = pd.DataFrame(summary["csi_score"],
            index=lmclassifier.feature_subsets_, columns=[time(i, j) for i, j in zip(week["Lower"], week["Upper"])])

        return summary





