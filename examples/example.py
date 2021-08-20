# encoding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.feature_preprocessings import  Tabular
from skcredit.feature_preprocessings import FTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.feature_selection import SelectCMIM
from skcredit.linear_model import LMClassifier

np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    tra = pd.read_csv("H:\\其他\\工作\\QuDian\\tra.csv").drop(["province"], axis=1)
    tes = pd.read_csv("H:\\其他\\工作\\QuDian\\tes.csv").drop(["province"], axis=1)

    tabular = Tabular(tra)
    trn_input, trn_label = tabular.input, tabular.label
    tabular = Tabular(tes)
    val_input, val_label = tabular.input, tabular.label

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
    # discrete.save_order("C:\\Users\\P1352\\Desktop")
    # discrete.save_table("C:\\Users\\P1352\\Desktop")

    # selectbin = SelectBin(tim_columns=tim_columns)
    # selectbin.fit(trn_input, trn_label)
    # trn_input = selectbin.transform(trn_input)
    # val_input = selectbin.transform(val_input)

    selectvif = SelectCMIM(keep_columns=[], date_columns=tim_columns)
    selectvif.fit(trn_input, trn_label)
    trn_input = selectvif.transform(trn_input)
    val_input = selectvif.transform(val_input)

    lmclassifier = LMClassifier(tim_columns=tim_columns)
    lmclassifier.fit(trn_input, trn_label, trn_input, trn_label)
    print(lmclassifier.model())
    print(lmclassifier.score(trn_input, trn_label))
    print(lmclassifier.score(val_input, val_label))

