# encoding: utf-8

import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.linear_model import LMClassifier
from skcredit.feature_preprocessing import TidyTabula
from skcredit.feature_discretization import save_table, DiscreteAuto
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)


if __name__ == "__main__":
    tra = pd.read_csv("H:\\work\\QuDian\\tra.csv")
    tes = pd.read_csv("H:\\work\\QuDian\\tes.csv")

    columns = [
        "hl_trd_amt_1y", "hl_trd_cnt_1y",
        "ovd_order_cnt_1m", "ovd_order_amt_1m",
        "ovd_order_cnt_3m", "ovd_order_amt_3m",
        "ovd_order_cnt_6m", "ovd_order_amt_6m",
        "ovd_order_cnt_12m", "ovd_order_amt_12m",
        "ovd_order_cnt_3m_m1_status", "ovd_order_cnt_6m_m1_status", "ovd_order_cnt_12m_m1_status",
        "ovd_order_cnt_12m_m3_status", "ovd_order_cnt_12m_m6_status","ovd_order_cnt_2y_m3_status", "ovd_order_cnt_2y_m6_status", "ovd_order_cnt_5y_m3_status", "ovd_order_cnt_5y_m6_status",
        "ovd_order_cnt_1m", "ovd_order_amt_1m", "is_midnight", "province"
    ]

    tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    tra_feature = tra_feature.drop(columns, axis=1)
    tes_feature = tes_feature.drop(columns, axis=1)

    tidy = TidyTabula(keep_columns=[], cat_columns=[], num_columns=tra_feature.columns.tolist())
    tidy.fit(tra_feature, tra_target)
    tra_feature = tidy.transform(tra_feature)
    tes_feature = tidy.transform(tes_feature)

    discrete = DiscreteAuto(keep_columns=[], information_value_threshold=0.1)
    discrete.fit(tra_feature, tra_target)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)

    model = LMClassifier(keep_columns=[], PDO=20, BASE=600, ODDS=1)
    model.fit(tra_feature, tra_target)