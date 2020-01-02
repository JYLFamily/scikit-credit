# encoding: utf-8

import numpy as np
import pandas as pd
from skcredit.linear_model import LMClassifier
from skcredit.feature_preprocessing import FormatTabular
from skcredit.feature_selection import SelectBin, SelectVif
from skcredit.feature_discretization import save_table, DiscreteAuto
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)


if __name__ == "__main__":
    # tra = pd.read_csv("F:\\work\\QuDian\\tra.csv")
    # tes = pd.read_csv("F:\\work\\QuDian\\tes.csv")
    #
    # columns = [
    #     "hl_trd_amt_1y", "hl_trd_cnt_1y",
    #     "ovd_order_cnt_1m",  "ovd_order_amt_1m",
    #     "ovd_order_cnt_3m",  "ovd_order_amt_3m",
    #     "ovd_order_cnt_6m",  "ovd_order_amt_6m",
    #     "ovd_order_cnt_12m", "ovd_order_amt_12m",
    #     "ovd_order_cnt_3m_m1_status", "ovd_order_cnt_6m_m1_status", "ovd_order_cnt_12m_m1_status",
    #     "ovd_order_cnt_12m_m3_status", "ovd_order_cnt_12m_m6_status",
    #     "ovd_order_cnt_2y_m3_status", "ovd_order_cnt_2y_m6_status",
    #     "ovd_order_cnt_5y_m3_status", "ovd_order_cnt_5y_m6_status",
    #     "is_midnight",
    #     "province"
    # ]
    #
    # tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    # tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)
    #
    # tra_feature = tra_feature.drop(columns, axis=1)
    # tes_feature = tes_feature.drop(columns, axis=1)
    #
    # ft = FormatTabular(keep_columns=[], cat_columns=[], num_columns=tra_feature.columns.tolist())
    # ft.fit(tra_feature, tra_target)
    # tra_feature = ft.transform(tra_feature)
    # tes_feature = ft.transform(tes_feature)
    #
    # discrete = DiscreteAuto(keep_columns=[], information_value_threshold=0.1)
    # discrete.fit(tra_feature, tra_target)
    # tra_feature = discrete.transform(tra_feature)
    # tes_feature = discrete.transform(tes_feature)
    #
    # sbin = SelectBin(keep_columns=[])
    # sbin.fit(tra_feature, tra_target)
    # tra_feature = sbin.transform(tra_feature)
    # tes_feature = sbin.transform(tes_feature)
    #
    # svif = SelectVif(keep_columns=[])
    # svif.fit(tra_feature, tra_target)
    # tra_feature = svif.transform(tra_feature)
    # tes_feature = svif.transform(tes_feature)
    #
    # save_table(discrete, "C:\\Users\\P1352\\Desktop")
    # tra_feature["target"] = tra_target
    # tes_feature["target"] = tes_target
    #
    # tra_feature.to_csv("C:\\Users\\P1352\\Desktop\\tra.csv")
    # tes_feature.to_csv("C:\\Users\\P1352\\Desktop\\tes.csv")

    tra = pd.read_csv("C:\\Users\\P1352\\Desktop\\tra.csv")
    tes = pd.read_csv("C:\\Users\\P1352\\Desktop\\tes.csv")

    tra = tra.drop(["credit_duration"], axis=1)
    tes = tes.drop(["credit_duration"], axis=1)

    tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    model = LMClassifier(keep_columns=[], PDO=20, BASE=600, ODDS=1)
    model.fit(tra_feature, tra_target)
    print(model.score(tra_feature, tra_target))
    print(model.score(tes_feature, tes_target))
    print(model.result())
