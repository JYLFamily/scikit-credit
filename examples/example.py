# encoding: utf-8

import os
import yaml
import warnings
import datetime
import numpy as np
import pandas as pd
from skcredit.feature_preprocessings import  Tabular
from skcredit.feature_preprocessings import CTabular
from skcredit.feature_preprocessings import FTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.online_check import FEndReport, BEndReport
from skcredit.feature_selection import SelectBin, SelectVif, SelectViz
from skcredit.linear_model import LMClassifier, LMCreditcard, LMValidation
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    trn = pd.read_csv("C:\\Users\\P1352\\Desktop\\creditcard.csv", nrows=5000)
    trn_input, trn_label = (trn.drop(["Class"], axis="columns"),
                            trn["Class"])

    tim_columns = []
    cat_columns = []
    num_columns = trn_input.columns.tolist()

    ft = FTabular(
        tim_columns=tim_columns,
        cat_columns=cat_columns,
        num_columns=num_columns)
    ft.fit(trn_input, trn_label)
    trn_input = ft.transform(trn_input)

    discrete = DiscreteAuto(
        tim_columns=tim_columns)
    discrete.fit(trn_input, trn_label)
    trn_input = discrete.transform(trn_input)

    # selectbin = SelectBin(
    #     tim_columns=tim_columns)
    # selectbin.fit(trn_input, trn_label)
    # trn_input = selectbin.transform(trn_input)
    # oot_input = selectbin.transform(oot_input)
    #
    # selectvif = SelectVif(
    #     tim_columns=tim_columns)
    # selectvif.fit(trn_input, trn_label)
    # trn_input = selectvif.transform(trn_input)
    # oot_input = selectvif.transform(oot_input)

    # import statsmodels.api as sm
    # logit_mod = sm.GLM(trn_label, sm.add_constant(trn_input), family=sm.families.Binomial())
    # logit_res = logit_mod.fit()
    # print(logit_res.summary())
    #
    # from sklearn.metrics import roc_curve
    # fpr, tpr, _ = roc_curve(trn_label, logit_res.predict(sm.add_constant(trn_input)))
    # print(round(max(tpr - fpr), 5))
    # fpr, tpr, _ = roc_curve(oot_label, logit_res.predict(sm.add_constant(oot_input)))
    # print(round(max(tpr - fpr), 5))
    #
    # logit_res.predict(sm.add_constant(trn_input)).to_frame("pred").to_csv()
    #
    # pd.concat([trn_id, trn_input, logit_res.predict(sm.add_constant(trn_input)).to_frame("pred")], axis=1).to_csv(
    #     "C:\\Users\\P1352\\Desktop\\trn_pred.csv", index=False)
    # pd.concat([oot_id, oot_input, logit_res.predict(sm.add_constant(oot_input)).to_frame("pred")], axis=1).to_csv(
    #     "C:\\Users\\P1352\\Desktop\\oot_pred.csv", index=False)
    # lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    # lmclassifier.fit(trn_input, trn_label)
    # print(lmclassifier.model())
    # print("{:.5f}".format(lmclassifier.score(trn_input, trn_label)))
    # print("{:.5f}".format(lmclassifier.score(oot_input, oot_label)))
