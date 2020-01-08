# encoding: utf-8

import warnings
import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.linear_model import LMClassifier
from skcredit.monitor import FEndReport, BEndReport
from skcredit.feature_preprocessing import FormatTabular
from skcredit.feature_discretization import DiscreteAuto
from skcredit.feature_selection import SelectBin, SelectVif
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    tra = pd.read_csv("C:\\Users\\P1352\\Desktop\\tra.csv")
    tes = pd.read_csv("C:\\Users\\P1352\\Desktop\\tes.csv")

    columns = [
        "apply_time",
        "last_1m_avg_asset_total",
        "last_1y_total_active_biz_cnt",
        "adr_stability_days",
        "mobile_fixed_days",
        "fnd_ern_amt_1m",
        "target"
    ]

    num_columns = [col for col in columns if col not in ("apply_time", "target")]

    tra = tra[columns].copy(deep=True)
    tes = tes[columns].copy(deep=True)

    tra["apply_time"] = pd.to_datetime(tra["apply_time"], format="%Y/%m/%d")
    tes["apply_time"] = pd.to_datetime(tes["apply_time"], format="%Y/%m/%d")

    tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    tra_feature_tmp = tra_feature.copy(deep=True)
    tes_feature_tmp = tes_feature.copy(deep=True)

    ft = FormatTabular(keep_columns=["apply_time"], cat_columns=[], num_columns=num_columns)
    ft.fit(tra_feature, tra_target)
    tra_feature = ft.transform(tra_feature)
    tes_feature = ft.transform(tes_feature)

    discrete = DiscreteAuto(keep_columns=["apply_time"])
    discrete.fit(tra_feature, tra_target)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)

    sbin = SelectBin(keep_columns=["apply_time"])
    sbin.fit(tra_feature, tra_target)
    tra_feature = sbin.transform(tra_feature)
    tes_feature = sbin.transform(tes_feature)

    svif = SelectVif(keep_columns=["apply_time"])
    svif.fit(tra_feature, tra_target)
    tra_feature = svif.transform(tra_feature)
    tes_feature = svif.transform(tes_feature)

    lmclassifier = LMClassifier(keep_columns=["apply_time"], PDO=20, BASE=600, ODDS=1)
    lmclassifier.fit(tra_feature, tra_target)

    result = FEndReport.psi_by_week(discrete, lmclassifier, tra_feature_tmp, tes_feature_tmp)
    pprint(result["table"])
    pprint(result["score"])

    result = FEndReport.csi_by_week(discrete, lmclassifier, tra_feature_tmp, tes_feature_tmp)
    pprint(result["table"])
    pprint(result["psi_score"])
    pprint(result["csi_score"])

    result = BEndReport.metric_by_week(discrete, lmclassifier, tra_feature_tmp, tra_target, tes_feature_tmp, tes_target)
    pprint(result)

    result = BEndReport.report_by_week(discrete, lmclassifier, tra_feature_tmp, tra_target, tes_feature_tmp, tes_target)
    pprint(result)
    pprint(result)