# encoding: utf-8

import os
import yaml
import warnings
import numpy as np
import pandas as pd
from skcredit.feature_preprocessings import Tabular
from skcredit.feature_preprocessings import CTabular
from skcredit.feature_preprocessings import FTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.online_check import FEndReport, BEndReport
from skcredit.feature_selection import SelectBin, SelectVif
from skcredit.linear_model import LMClassifier, LMCreditcard, LMValidation
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    with open("configs.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    # tabular_2_sua = pd.read_excel(
    #     os.path.join(config["path"], "3_6_1_period_sua.xlsx"), na_values=[-1, -2, -3])
    # tabular_3_sua = pd.read_excel(
    #     os.path.join(config["path"], "3_6_2_period_sua.xlsx"), na_values=[-1, -2, -3])
    # tabular_2_sua["IDT"] = pd.to_datetime(tabular_2_sua["IDT"])
    # tabular_3_sua["IDT"] = pd.to_datetime(tabular_3_sua["IDT"])
    #
    # tabular_2_afu = pd.read_excel(os.path.join(config["path"], "3_6_1_period_afu.xlsx"))
    # tabular_3_afu = pd.read_excel(os.path.join(config["path"], "3_6_2_period_afu.xlsx"))
    # tabular_2_afu["IDT"] = pd.to_datetime(tabular_2_afu["IDT"])
    # tabular_3_afu["IDT"] = pd.to_datetime(tabular_3_afu["IDT"])
    #
    # tabular_2 = tabular_2_sua.merge(
    #     tabular_2_afu, on=["NAME", "CID", "MOBILE", "IDT"], how="left")
    # tabular_3 = tabular_3_sua.merge(
    #     tabular_3_afu, on=["NAME", "CID", "MOBILE", "IDT"], how="left")
    #
    # tabular_2 = tabular_2.select_dtypes(exclude="object")
    # tabular_3 = tabular_3.select_dtypes(exclude="object")
    #
    # tabular = pd.concat([
    #     tabular_2,
    #     tabular_3.reindex(columns=tabular_2.columns)
    # ]).drop(["score_zy", "IDT"], axis=1).sort_index()

    # import missingno as msno
    # import matplotlib.pyplot as plt
    # msno.matrix(tabular[["xx31", "xx361", "xx372", "xx396", "score", "score_m1", "pdls013"]])
    # plt.show()

    tabular_1 = pd.read_excel(
        os.path.join(config["path"], "3_6_1_period.xlsx"), na_values=[-1, -2, -3])
    tabular_2 = pd.read_excel(
        os.path.join(config["path"], "3_6_2_period.xlsx"), na_values=[-1, -2, -3])

    tabular = pd.concat([tabular_1, tabular_2.reindex(columns=tabular_1.columns)])
    tabular = tabular.select_dtypes(exclude="object").drop(["IDT", "score_zy", "afu_score"], axis=1)

    t = Tabular(tabular=tabular)
    trn_input = t.input
    trn_label = t.label

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

    discrete.save_order(config["path"])
    discrete.save_table(config["path"])

    selectbin = SelectBin(
        tim_columns=tim_columns)
    selectbin.fit(trn_input, trn_label)
    trn_input = selectbin.transform(trn_input)

    selectvif = SelectVif(
        tim_columns=tim_columns)
    selectvif.fit(trn_input, trn_label)
    trn_input = selectvif.transform(trn_input)

    lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    lmclassifier.fit(trn_input, trn_label)
    print("{:.5f}".format(lmclassifier.score(trn_input, trn_label)))
    # print("{:.5f}".format(lmclassifier.score(val_input, val_label)))
    from pprint import pprint
    pprint(lmclassifier.model())

    lmcreditcard = LMCreditcard(discrete, lmclassifier)
    pprint(lmcreditcard())
    # print("=" * 72)
    # pprint(LMValidation.intercept_alignment(trn_target, tes_target))
    # print("=" * 72)
    # pprint(LMValidation.attribute_alignment(discrete, lmclassifier, trn_feature, trn_target, tes_feature, tes_target))
    # print("=" * 72)
    # pprint(FEndReport.psi_by_week(discrete, lmclassifier, trn_input, tes_input))
    # print("=" * 72)
    # pprint(FEndReport.csi_by_week(discrete, lmclassifier, trn_input, tes_input))
    # print("=" * 72)
    # pprint(BEndReport.metric_by_week(discrete, lmclassifier, trn_input, trn_target, tes_input, tes_target))
    # print("=" * 72)
    # pprint(BEndReport.report_by_week(discrete, lmclassifier, trn_input, trn_target, tes_input, tes_target))
