# coding: utf-8

import gc
import numpy  as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def calc_table(X):
    x = X.copy(deep=True)
    del X
    gc.collect()

    cnt_rec = x["scores"].value_counts(sort=False).to_frame("CntRec").replace({0: 0.5})
    cnt_positive = x.loc[x["target"] == 1, "scores"].value_counts(sort=False).to_frame("CntPositive").replace(
        {0: 0.5})
    cnt_negative = x.loc[x["target"] == 0, "scores"].value_counts(sort=False).to_frame("CntNegative").replace(
        {0: 0.5})

    table = pd.concat([cnt_rec, cnt_positive, cnt_negative], axis=1)
    table["LIFT"] = table["CntPositive"] / table["CntPositive"].sum()
    table["RATE"] = table["CntPositive"] / table["CntRec"]
    table["ODDS"] = table["CntPositive"] / table["CntNegative"]
    table["LN(ODDS)"] = np.log(table["ODDS"])

    return table.reset_index().rename(columns={"index": "scores"})


class BEndReport(object):
    @staticmethod
    def metric(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label):
        tra_feature = discrete.transform(tra_input)
        tes_feature = discrete.transform(tes_input)

        result = OrderedDict()

        fpr, tpr, _ = roc_curve(tra_label, lmclassifier.predict_proba(tra_feature)["proba_positive"])
        auc = roc_auc_score(tra_label, lmclassifier.predict_proba(tra_feature)["proba_positive"])

        result["tra"] = {"ks": np.round(np.max(tpr - fpr), 5), "auc": np.round(auc, 5)}

        fpr, tpr, _ = roc_curve(tes_label, lmclassifier.predict_proba(tes_feature)["proba_positive"])
        auc = roc_auc_score(tes_label, lmclassifier.predict_proba(tes_feature)["proba_positive"])

        result["tes"] = {"ks": np.round(np.max(tpr - fpr), 5), "auc": np.round(auc, 5)}

        return result

    @staticmethod
    def report(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label):
        tra_feature = discrete.transform(tra_input)
        tes_feature = discrete.transform(tes_input)

        tra_score = lmclassifier.predict_score(tra_feature)
        tes_score = lmclassifier.predict_score(tes_feature)

        result = OrderedDict()

        bins = np.append(
            np.array([0]),
            np.arange(np.ceil(tra_score.min() / 10) * 10, np.floor(tra_score.max() / 10) * 10 + 10, 10)
        )
        bins = np.append(bins, np.array([999]))

        tra = pd.DataFrame({
            "scores": pd.cut(tra_score, bins=bins),
            "target": tra_label
        })
        tes = pd.DataFrame({
            "scores": pd.cut(tes_score, bins=bins),
            "target": tes_label
        })

        result["tra"] = calc_table(tra)
        result["tes"] = calc_table(tes)

        return result

    @staticmethod
    def metric_by_week(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label):
        time = "[{}, {}]".format

        week = pd.DataFrame({
            "week": tes_input[lmclassifier.tim_columns].squeeze().dt.week,
            "date": tes_input[lmclassifier.tim_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = OrderedDict()

        for i, (_, row) in enumerate(week.iterrows()):
            subset_feature = tes_input.loc[(tes_input[lmclassifier.tim_columns].squeeze() >= row["Lower"]) &
                                           (tes_input[lmclassifier.tim_columns].squeeze() <= row["Upper"])]
            subset_label = tes_label.loc[subset_feature.index]

            result = BEndReport().metric(discrete, lmclassifier, tra_input, tra_label, subset_feature, subset_label)

            summary[time(row["Lower"], row["Upper"])] = result

        return summary

    @staticmethod
    def report_by_week(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label):
        time = "[{}, {}]".format

        week = pd.DataFrame({
            "week": tes_input[lmclassifier.tim_columns].squeeze().dt.week,
            "date": tes_input[lmclassifier.tim_columns].squeeze()
        })

        week = pd.concat([
            week.groupby("week")["date"].min().dt.strftime("%Y-%m-%d").to_frame("Lower"),
            week.groupby("week")["date"].max().dt.strftime("%Y-%m-%d").to_frame("Upper")
        ], axis=1)

        summary = OrderedDict()

        for i, (_, row) in enumerate(week.iterrows()):
            subset_feature = tes_input.loc[(tes_input[lmclassifier.tim_columns].squeeze() >= row["Lower"]) &
                                           (tes_input[lmclassifier.tim_columns].squeeze() <= row["Upper"])]
            subset_label = tes_label.loc[subset_feature.index]

            result = BEndReport().report(discrete, lmclassifier, tra_input, tra_label, subset_feature, subset_label)

            summary[time(row["Lower"], row["Upper"])] = result

        return summary

