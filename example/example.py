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

    tra = pd.read_csv(os.path.join(config["path"], "tra.csv"))
    tes = pd.read_csv(os.path.join(config["path"], "tes.csv"))

    tra_input, tra_label = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_input, tes_label = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    tim_columns = ["apply_time"]
    cat_columns = ["province", "is_midnight"]
    num_columns = [col for col in tra_input.columns if col not in tim_columns + cat_columns]
    # tim_columns = ["period"]
    # cat_columns = ["user_basic.user_province", "user_basic.user_phone_province"]
    # num_columns = [col for col in tra_input.columns if col not in tim_columns + cat_columns]

    ft = FormatTabular(
        tim_columns=tim_columns,
        cat_columns=cat_columns,
        num_columns=num_columns)
    ft.fit(tra_input, tra_label)
    tra_input = ft.transform(tra_input)
    tes_input = ft.transform(tes_input)

    discrete = DiscreteAuto(
        tim_columns=tim_columns)
    discrete.fit(tra_input, tra_label)
    discrete.save_order(config["path"])
    discrete.save_table(config["path"])
    discrete.save_order_cross(config["path"])
    discrete.save_table_cross(config["path"])
    tra_feature = discrete.transform(tra_input)
    tes_feature = discrete.transform(tes_input)

    selectbin = SelectBin(
        tim_columns=tim_columns)
    selectbin.fit(tra_feature, tra_label)
    tra_feature = selectbin.transform(tra_feature)
    tes_feature = selectbin.transform(tes_feature)
    tra_feature.to_csv(os.path.join(config["path"], "tra_feature.csv"), index=False)
    tes_feature.to_csv(os.path.join(config["path"], "tes_feature.csv"), index=False)

    selectvif = SelectVif(
        tim_columns=tim_columns)
    selectvif.fit(tra_feature, tra_label)
    tra_feature = selectvif.transform(tra_feature)
    tes_feature = selectvif.transform(tes_feature)

    lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    lmclassifier.fit(tra_feature, tra_label)
    pprint("{:.5f}".format(lmclassifier.score(tra_feature, tra_label)))
    pprint("{:.5f}".format(lmclassifier.score(tes_feature, tes_label)))
    pprint(lmclassifier.model())
    #
    # lmcreditcard = LMCreditcard(discrete, lmclassifier)
    # pprint(lmcreditcard())
    # print("=" * 72)
    # pprint(LMValidation.intercept_alignment(tra_label, tes_label))
    # print("=" * 72)
    # pprint(LMValidation.attribute_alignment(discrete, lmclassifier, tra_feature, tra_label, tes_feature, tes_label))
    # print("=" * 72)
    # pprint(FEndReport.psi_by_week(discrete, lmclassifier, tra_input, tes_input))
    # print("=" * 72)
    # pprint(FEndReport.csi_by_week(discrete, lmclassifier, tra_input, tes_input))
    # print("=" * 72)
    # pprint(BEndReport.metric_by_week(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label))
    # print("=" * 72)
    # pprint(BEndReport.report_by_week(discrete, lmclassifier, tra_input, tra_label, tes_input, tes_label))
