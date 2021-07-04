# encoding: utf-8

import os
import yaml
import warnings
import datetime
import numpy  as np
import pandas as pd
from skcredit.feature_preprocessings import  Tabular
from skcredit.feature_preprocessings import CTabular
from skcredit.feature_preprocessings import FTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.online_check import FEndReport, BEndReport
from skcredit.feature_selection import SelectBin,  SelectVif, SelectViz
from skcredit.linear_model import LMClassifier, LMCreditcard, LMValidation
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    csv = pd.read_csv("D:\\WorkSpace\\20210628ICBC\\AFeatureExtract\\支付标签-183867条-020000条.csv")

    tabular = Tabular(csv)
    trn_input, val_input = tabular.trn_val_input
    trn_label, val_label = tabular.trn_val_label

    tim_columns = [trn_input.columns.tolist()[0]]
    cat_columns = []
    num_columns = trn_input.columns.tolist()[1: ]

    ft = FTabular(
        tim_columns=tim_columns,
        cat_columns=cat_columns,
        num_columns=num_columns)
    ft.fit(trn_input)
    trn_input = ft.transform(trn_input)
    val_input = ft.transform(val_input)

    discrete = DiscreteAuto(tim_columns=tim_columns)
    discrete.fit(trn_input, trn_label)
    trn_input = discrete.transform(trn_input)
    val_input = discrete.transform(val_input)

    selectbin = SelectBin(tim_columns=tim_columns)
    selectbin.fit(trn_input, trn_label)
    trn_input = selectbin.transform(trn_input)
    val_input = selectbin.transform(val_input)

    selectvif = SelectVif(tim_columns=tim_columns)
    selectvif.fit(trn_input, trn_label)
    trn_input = selectvif.transform(trn_input)
    val_input = selectvif.transform(val_input)

    lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    lmclassifier.fit(trn_input, trn_label, trn_input, trn_label)
    print(lmclassifier.model())
    print(lmclassifier.score(trn_input, trn_label))
    print(lmclassifier.score(val_input, val_label))

