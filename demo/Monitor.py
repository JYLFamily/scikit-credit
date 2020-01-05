# encoding: utf-8

import numpy as np
import pandas as pd
import warnings
from skcredit.linear_model import LMClassifier
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
    # tra = pd.read_csv("H:\\work\\QuDian\\tra.csv")
    # tes = pd.read_csv("H:\\work\\QuDian\\tes.csv")
    #
    # cat_columns = ["province", "is_midnight"]
    # num_columns = [col for col in tra.columns if col not in cat_columns + ["apply_time"] + ["target"]]
    #
    # tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    # tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)
    #
    # ft = FormatTabular(keep_columns=["apply_time"], cat_columns=cat_columns, num_columns=num_columns)
    # ft.fit(tra_feature, tra_target)
    # tra_feature = ft.transform(tra_feature)
    # tes_feature = ft.transform(tes_feature)
    #
    # discrete = DiscreteAuto(keep_columns=["apply_time"])
    # discrete.fit(tra_feature, tra_target)
    # tra_feature = discrete.transform(tra_feature)
    # tes_feature = discrete.transform(tes_feature)
    # discrete.save_order("C:\\Users\\15795\\Desktop")
    # discrete.save_table("C:\\Users\\15795\\Desktop")
    #
    # sbin = SelectBin(keep_columns=["apply_time"])
    # sbin.fit(tra_feature, tra_target)
    # tra_feature = sbin.transform(tra_feature)
    # tes_feature = sbin.transform(tes_feature)
    #
    # svif = SelectVif(keep_columns=["apply_time"])
    # svif.fit(tra_feature, tra_target)
    # tra_feature = svif.transform(tra_feature)
    # tes_feature = svif.transform(tes_feature)
    #
    # tra_feature["target"] = tra_target
    # tes_feature["target"] = tes_target
    #
    # tra_feature.to_csv("C:\\Users\\15795\\Desktop\\tra.csv", index=False)
    # tes_feature.to_csv("C:\\Users\\15795\\Desktop\\tes.csv", index=False)

    tra = pd.read_csv("C:\\Users\\15795\\Desktop\\tra.csv")
    tes = pd.read_csv("C:\\Users\\15795\\Desktop\\tes.csv")

    tra_feature, tra_target = tra.drop(["target"], axis=1).copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_target = tes.drop(["target"], axis=1).copy(deep=True), tes["target"].copy(deep=True)

    model = LMClassifier(keep_columns=["apply_time"], PDO=20, BASE=600, ODDS=1)
    model.fit(tra_feature, tra_target)
    print(model.score(tra_feature, tra_target))
    print(model.score(tes_feature, tes_target))
    print(model.result())
    print(type(model.coeff_))
