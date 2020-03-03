# encoding: utf-8

import os
import yaml
import warnings
import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.feature_preprocessing import FormatTabular
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
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    trn = pd.read_csv(os.path.join(config["path"], "trn.csv"))
    tes = pd.read_csv(os.path.join(config["path"], "tes.csv"))

    trn = trn.drop(["nom_0", "bin_1", "bin_3", "bin_4"], axis=1)
    tes = tes.drop(["nom_0", "bin_1", "bin_3", "bin_4"], axis=1)

    trn_input, trn_target = trn.drop(["target"], axis=1).copy(deep=True), trn["target"].copy(deep=True)
    tes_input = tes
    del trn, tes
    import gc
    gc.collect()

    tim_columns = ["id"]
    cat_columns = [col for col in trn_input.columns if col not in tim_columns]
    num_columns = []

    ft = FormatTabular(
        tim_columns=tim_columns,
        cat_columns=cat_columns,
        num_columns=num_columns)
    ft.fit(trn_input, trn_target)
    trn_input = ft.transform(trn_input)
    tes_input = ft.transform(tes_input)

    discrete = DiscreteAuto(
        tim_columns=tim_columns)
    discrete.fit(trn_input, trn_target)
    discrete.save_order(config["path"])
    discrete.save_table(config["path"])
    discrete.save_order_cross(config["path"])
    discrete.save_table_cross(config["path"])

    # trn_feature = discrete.transform(trn_input)
    # tes_feature = discrete.transform(tes_input)

    # selectbin = SelectBin(
    #     tim_columns=tim_columns)
    # selectbin.fit(trn_feature, trn_target)
    # trn_feature = selectbin.transform(trn_feature)
    # tes_feature = selectbin.transform(tes_feature)
    #
    # selectvif = SelectVif(
    #     tim_columns=tim_columns)
    # selectvif.fit(trn_feature, trn_target)
    # trn_feature = selectvif.transform(trn_feature)
    # tes_feature = selectvif.transform(tes_feature)
    #
    # lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    # lmclassifier.fit(trn_feature, trn_target)
    # pprint("{:.5f}".format(lmclassifier.score(trn_feature, trn_target)))
    # pprint("{:.5f}".format(lmclassifier.score(tes_feature, tes_target)))
    # pprint(lmclassifier.model())
    #
    # lmcreditcard = LMCreditcard(discrete, lmclassifier)
    # pprint(lmcreditcard())
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
