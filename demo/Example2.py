# encoding: utf-8

import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.feature_discretization import save_table, Discrete
from skcredit.feature_preprocessing import TidyTabula, SaveMemory
from skcredit.feature_selection import SelectBin, SelectVif
from skcredit.linear_model import LRClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    # tra = pd.read_csv("E:\\work\\QuDian\\tra.csv")
    # tes = pd.read_csv("E:\\work\\QuDian\\tes.csv")
    #
    # tra = tra.drop(["province"], axis=1)
    # tes = tes.drop(["province"], axis=1)
    #
    # tra_feature, tra_label = tra.drop(["target"], axis=1), tra["target"]
    # tes_feature, tes_label = tes.drop(["target"], axis=1), tes["target"]
    #
    # tidy = TidyTabula(keep_columns=[], cat_columns=[], num_columns=tra_feature.columns.tolist())
    # save = SaveMemory(keep_columns=[], cat_columns=[], num_columns=tra_feature.columns.tolist())
    # discrete = Discrete(keep_columns=[], cat_columns=[], num_columns=tra_feature.columns.tolist(), merge_bin=0.05)
    #
    # tidy.fit(tra_feature, tra_label)
    # tra_feature = tidy.transform(tra_feature)
    # tes_feature = tidy.transform(tes_feature)
    #
    # save.fit(tra_feature, tra_label)
    # tra_feature = save.transform(tra_feature)
    # tes_feature = save.transform(tes_feature)
    #
    # discrete.fit(tra_feature, tra_label)
    # tra_feature = discrete.transform(tra_feature)
    # tes_feature = discrete.transform(tes_feature)
    #
    # save_table(discrete, "E:\\work\\QuDian")
    # pickle.dump({"feature": tra_feature, "label": tra_label}, open("E:\\work\\QuDian\\tra.pkl", "wb"))
    # pickle.dump({"feature": tes_feature, "label": tes_label}, open("E:\\work\\QuDian\\tes.pkl", "wb"))

    tra = pickle.load(open("F:\\work\\QuDian\\tra.pkl", "rb"))
    tes = pickle.load(open("F:\\work\\QuDian\\tes.pkl", "rb"))

    tra_feature, tra_label = tra["feature"], tra["label"]
    tes_feature, tes_label = tes["feature"], tes["label"]

    tra_feature = tra_feature[
        ["ebill_pay_amt_6m", "last_1m_avg_asset_total",
         "adr_stability_days", "mobile_fixed_days", "xfdc_index"]]
    tes_feature = tes_feature[
        ["ebill_pay_amt_6m", "last_1m_avg_asset_total",
         "adr_stability_days", "mobile_fixed_days", "xfdc_index"]]

    tra_feature = tra_feature.rename(columns={
        "ebill_pay_amt_6m": "近六月消费金额",
        "last_1m_avg_asset_total": "近六月消费笔数",
        "adr_stability_days": "地址稳定指数",
        "mobile_fixed_days": "风险等级",
        "xfdc_index": "消费等级"})
    tes_feature = tes_feature.rename(columns={
        "ebill_pay_amt_6m": "近六月消费金额",
        "last_1m_avg_asset_total": "近六月消费笔数",
        "adr_stability_days": "地址稳定指数",
        "mobile_fixed_days": "风险等级",
        "xfdc_index": "消费等级"})

    pickle.dump({"feature": tra_feature, "label": tra_label}, open("F:\\work\\QuDian\\tra_woe.pkl", "wb"))
    pickle.dump({"feature": tes_feature, "label": tes_label}, open("F:\\work\\QuDian\\tes_woe.pkl", "wb"))

    from keras.callbacks import Callback
