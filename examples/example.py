# encoding: utf-8

import os
import yaml
import warnings
import datetime
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

    df = pd.read_csv(os.path.join(config["path"], "tra.csv"), usecols=["zmscore", "target"])
    df = df.head(5000)

    trn = Tabular(tabular=df)
    trn_input, trn_label = trn.input, trn.label

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
    discrete.save_order(config["path"])
    discrete.save_table(config["path"])
    trn_input = discrete.transform(trn_input)

    # selectbin = SelectBin(
    #     tim_columns=tim_columns)
    # selectbin.fit(trn_input, trn_label)
    # trn_input = selectbin.transform(trn_input)
    #
    # selectvif = SelectVif(
    #     tim_columns=tim_columns)
    # selectvif.fit(trn_input, trn_label)
    # trn_input = selectvif.transform(trn_input)

    # lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    # lmclassifier.fit(trn_input, trn_label)
    import statsmodels.api as sm
    logit_mod = sm.GLM(trn_label, sm.add_constant(trn_input), family=sm.families.Binomial())
    logit_res = logit_mod.fit()
    print(logit_res.summary())

    # print("{:.5f}".format(lmclassifier.score(trn_input, trn_label)))
    # print(lmclassifier.predict_score(trn_input))
    # print(lmclassifier.predict_proba(trn_input))

    # from sklearn.model_selection import ShuffleSplit
    # splits = ShuffleSplit(n_splits=4, random_state=7)
    # for trn_idx, tes_idx in splits.split(df):
    #     trn = Tabular(tabular=df.loc[trn_idx])
    #     trn_input, trn_label = trn.input, trn.label
    #
    #     tes = Tabular(tabular=df.loc[tes_idx])
    #     tes_input, tes_label = tes.input, tes.label
    #
    #     tim_columns = []
    #     cat_columns = []
    #     num_columns = trn_input.columns.tolist()
    #
    #     ft = FTabular(
    #         tim_columns=tim_columns,
    #         cat_columns=cat_columns,
    #         num_columns=num_columns)
    #     ft.fit(trn_input, trn_label)
    #     trn_input = ft.transform(trn_input)
    #     tes_input = ft.transform(tes_input)
    #
    #     discrete = DiscreteAuto(
    #         tim_columns=tim_columns)
    #     discrete.fit(trn_input, trn_label)
    #     trn_input = discrete.transform(trn_input)
    #     tes_input = discrete.transform(tes_input)
    #
    #     selectbin = SelectBin(
    #         tim_columns=tim_columns)
    #     selectbin.fit(trn_input, trn_label)
    #     trn_input = selectbin.transform(trn_input)
    #     tes_input = selectbin.transform(tes_input)
    #
    #     selectvif = SelectVif(
    #         tim_columns=tim_columns)
    #     selectvif.fit(trn_input, trn_label)
    #     trn_input = selectvif.transform(trn_input)
    #     tes_input = selectvif.transform(tes_input)
    #
    #     lmclassifier = LMClassifier(tim_columns=tim_columns, PDO=20, BASE=600, ODDS=1)
    #     lmclassifier.fit(trn_input, trn_label)
    #     print("{:.5f}".format(lmclassifier.score(trn_input, trn_label)))
    #     print("{:.5f}".format(lmclassifier.score(tes_input, tes_label)))
    #     from pprint import pprint
    #     pprint(lmclassifier.model())
    #     lmcreditcard = LMCreditcard(discrete, lmclassifier)
    #     pprint(lmcreditcard())
    #     print("=" * 72)
