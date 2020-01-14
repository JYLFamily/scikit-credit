# encoding: utf-8

import warnings
import numpy as np
import pandas as pd
from skcredit.feature_preprocessing import FormatTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.feature_selection import SelectBin, SelectVif
from skcredit.linear_model import LMClassifier, LMCreditcard
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    columns = [
        "zmscore",
        "avg_frd_zms",
        "tot_pay_amt_1m",
        "ebill_pay_amt_1m",
        "credit_pay_amt_1m",
        "pre_1y_pay_amount",
        "mobile_fixed_days",
        "adr_stability_days",
        "positive_biz_cnt_1y",
        "last_6m_avg_asset_total",
        "last_1m_avg_asset_total",
        "target"]

    tra = pd.read_csv("F:\\work\\QuDian\\tra.csv")
    tes = pd.read_csv("F:\\work\\QuDian\\tes.csv")

    tra = tra[columns]
    tes = tes[columns]

    tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    ft = FormatTabular(
        keep_columns=[],
        cat_columns=[],
        num_columns=[col for col in columns if col not in ["target"]]
    )
    ft.fit(tra_feature, tra_target)
    tra_feature = ft.transform(tra_feature)
    tes_feature = ft.transform(tes_feature)

    discrete = DiscreteAuto(keep_columns=[])
    discrete.fit(tra_feature, tra_target)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)

    lmclassifier = LMClassifier(keep_columns=[], PDO=20, BASE=600, ODDS=1)
    lmclassifier.fit(tra_feature, tra_target)
    print("{:.5f}".format(lmclassifier.score(tra_feature, tra_target)))
    print("{:.5f}".format(lmclassifier.score(tes_feature, tes_target)))
    print(lmclassifier.model())

    LMCreditcard.attribute_alignment(discrete, lmclassifier, tra_feature, tra_target)