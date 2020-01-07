# coding: utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def calc_table(X):
    x = X.copy(deep=True)
    del X
    gc.collect()

    cnt_rec = x.groupby("Lower & Upper")["target"].agg(len).to_frame("CntRec")
    cnt_positive = x.loc[x["target"] == 1, :].groupby("Lower & Upper")["target"].agg(len).to_frame("CntPositive")
    cnt_negative = x.loc[x["target"] == 0, :].groupby("Lower & Upper")["target"].agg(len).to_frame("CntNegative")

    table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
    table["LIFT"] = table["CntPositive"] / table["CntPositive"].sum()
    table["ODDS"] = table["CntNegative"] / table["CntPositive"]
    table["LN(ODDS)"] = np.log(table["ODDS"])

    return table.reset_index()


class BEndReport(object):
    @staticmethod
    def scores(discrete, lmclassifier, tra_feature, tra_label, tes_feature, tes_label):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        tra_score = lmclassifier.predict_score(tra_feature)
        tes_score = lmclassifier.predict_score(tes_feature)

        bins = np.append(
            np.array([0]),
            np.arange(np.ceil(tra_score.min() / 10) * 10, np.floor(tra_score.max() / 10) * 10 + 10, 10)
        )
        bins = np.append(bins, np.array([999]))

        tra = pd.DataFrame({
            "Lower & Upper": pd.cut(tra_score, bins=bins),
            "target": tra_label
        })
        tes = pd.DataFrame({
            "Lower & Upper": pd.cut(tes_score, bins=bins),
            "target": tes_label
        })

        tra_table = calc_table(tra)[["Lower & Upper", "LN(ODDS)"]]
        tes_table = calc_table(tes)[["Lower & Upper", "LN(ODDS)"]]

        result = {"tra": tra_table, "tes": tes_table}

        return result

    @staticmethod
    def metric(discrete, lmclassifier, tra_feature, tra_label, tes_feature, tes_label):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        result = dict()

        fpr, tpr, _ = roc_curve(tra_label, lmclassifier.predict_proba(tra_feature)[:, 1])
        auc = roc_auc_score(tra_label, lmclassifier.predict_proba(tra_feature)[:, 1])

        result["tra"] = {"ks": np.round(np.max(tpr - fpr), 5), "auc": np.round(auc, 5)}

        fpr, tpr, _ = roc_curve(tes_label, lmclassifier.predict_proba(tes_feature)[:, 1])
        auc = roc_auc_score(tes_label, lmclassifier.predict_proba(tes_feature)[:, 1])

        result["tes"] = {"ks": np.round(np.max(tpr - fpr), 5), "auc": np.round(auc, 5)}

        return result

    @staticmethod
    def report(discrete, lmclassifier, tra_feature, tra_label, tes_feature, tes_label):
        tra_feature = discrete.transform(tra_feature)
        tes_feature = discrete.transform(tes_feature)

        tra_score = lmclassifier.predict_score(tra_feature)
        tes_score = lmclassifier.predict_score(tes_feature)

        bins = np.append(
            np.array([0]),
            np.arange(np.floor(tra_score.min() / 10) * 10, np.ceil(tra_score.max() / 10) * 10, 10)
        )
        bins = np.append(bins, np.array([999]))

        tra = pd.DataFrame({
            "Lower & Upper": pd.cut(tra_score, bins=bins),
            "target": tra_label
        })
        tes = pd.DataFrame({
            "Lower & Upper": pd.cut(tes_score, bins=bins),
            "target": tes_label
        })

        tra_table = calc_table(tra)[["Lower & Upper", "CntRec", "CntPositive", "CntNegative", "LIFT", "ODDS"]]
        tes_table = calc_table(tes)[["Lower & Upper", "CntRec", "CntPositive", "CntNegative", "LIFT", "ODDS"]]

        result = {"tra": tra_table, "tes": tes_table}

        return result

