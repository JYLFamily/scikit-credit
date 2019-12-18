# coding: utf-8

import copy
import numpy as np
import pandas as pd
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

        bins = np.append(
            np.array([0]),
            np.arange(np.floor(tra_score.min() / 10) * 10, np.ceil(tra_score.max() / 10) * 10, 10)
        )
        bins = np.append(bins, np.array([999]))

        table = pd.DataFrame({
            "CntTra": pd.cut(tra_score, bins=bins).value_counts(),
            "CntTes": pd.cut(tes_score, bins=bins).value_counts()
        })

        table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
        table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
        table = table.reset_index().rename(columns={"index": "Lower & Upper"})

        psi = np.round(np.sum((table["RatTes"] - table["RatTra"]) * np.log(table["RatTes"] / table["RatTra"])), 5)

        result = {"table": table, "psi": psi}

        return result

    @staticmethod
    def csi(discrete, lmclassifier, tra_feature, tes_feature):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        tables = dict()
        tables.update(discrete.cat_table_)
        tables.update(discrete.num_table_)

        coeffs = lmclassifier.result()

        result = dict()

        for col in discrete.cat_columns_:
            table = copy.deepcopy(tables[col][[col, "WoE"]])
            tra_cnt = tra_feature[col].value_counts().to_frame("CntTra").reset_index().rename(columns={"index": "WoE"})
            tes_cnt = tes_feature[col].value_counts().to_frame("CntTes").reset_index().rename(columns={"index": "WoE"})

            table = table.merge(
                tra_cnt.merge(
                    tes_cnt, left_on="WoE", right_on="WoE", how="left"), left_on="WoE", right_on="WoE", how="left")
            table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
            table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
            table["PScore"] = - table["WoE"] * coeffs[col] * lmclassifier.b_
            table["SScore"] = (table["RatTes"] - table["RatTra"]) * table["PScore"]

            csi = np.round(table["SScore"].sum(), 5)

            result[col] = {"table": table, "csi": csi}

        for col in discrete.num_columns_:
            table = copy.deepcopy(tables[col][["Lower", "Upper", "WoE"]])
            tra_cnt = tra_feature[col].value_counts().to_frame("CntTra").reset_index().rename(columns={"index": "WoE"})
            tes_cnt = tes_feature[col].value_counts().to_frame("CntTes").reset_index().rename(columns={"index": "WoE"})

            table = table.merge(
                tra_cnt.merge(
                    tes_cnt, left_on="WoE", right_on="WoE", how="left"), left_on="WoE", right_on="WoE", how="left")
            table["RatTra"] = table["CntTra"] / table["CntTra"].sum()
            table["RatTes"] = table["CntTes"] / table["CntTes"].sum()
            table["PScore"] = - table["WoE"] * coeffs[col] * lmclassifier.b_
            table["SScore"] = (table["RatTes"] - table["RatTra"]) * table["PScore"]

            csi = np.round(table["SScore"].sum(), 5)

            result[col] = {"table": table, "csi": csi}

        return result









