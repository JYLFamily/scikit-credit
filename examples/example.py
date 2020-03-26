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

    # trn = Tabular(tabular=tabular.loc[trn_index])
    # trn_input = trn.input
    # trn_label = trn.label
    #
    # val = Tabular(tabular=tabular.loc[val_index])
    # val_input = val.input
    # val_label = val.label

    # t = Tabular(tabular=tabular)
    # trn_input = t.input
    # trn_label = t.label
    #
    # tim_columns = []
    # cat_columns = []
    # num_columns = trn_input.columns.tolist()
    #
    # ft = FTabular(
    #     tim_columns=tim_columns,
    #     cat_columns=cat_columns,
    #     num_columns=num_columns)
    # ft.fit(trn_input, trn_label)
    # trn_input = ft.transform(trn_input)
    #
    # discrete = DiscreteAuto(
    #     tim_columns=tim_columns)
    # discrete.fit(trn_input, trn_label)
    # trn_input = discrete.transform(trn_input)
    #
    # selectbin = SelectBin(
    #     tim_columns=tim_columns)
    # selectbin.fit(trn_input, trn_label)
    # trn_input = selectbin.transform(trn_input)
    #
    # selectvif = SelectVif(
    #     tim_columns=tim_columns)
    # selectvif.fit(trn_input, trn_label)
    # trn_input = selectvif.transform(trn_input)
    #
    # lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    # lmclassifier.fit(trn_input, trn_label)
    # print("{:.5f}".format(lmclassifier.score(trn_input, trn_label)))
    # print("{:.5f}".format(lmclassifier.score(val_input, val_label)))
    # from pprint import pprint
    # pprint(lmclassifier.model())
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
