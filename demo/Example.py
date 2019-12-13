# encoding: utf-8

import os
import yaml
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.linear_model import ks_score, LRClassifier
from skcredit.feature_selection import SelectBin, SelectVif
from skcredit.feature_discretization import save_table, Discrete
from skcredit.feature_preprocessing import TidyTabula, SaveMemory
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    tra = pd.read_csv(os.path.join(config["path"], config["dataset"]["tra"]), na_values=[r"\N"])
    tes = pd.read_csv(os.path.join(config["path"], config["dataset"]["tes"]), na_values=[r"\N"])

    cat_columns = [
        "gender",
        "province",
        "消费偏好",
        "持卡人价值",
        "卡使用状态"]
    num_columns = [col for col in tra.columns if col.endswith(("金额", "笔数"))]
    num_columns = num_columns + [
        "age",
        "消费趋势",
        "消费能力",
        "钱包位置",
        "近1个月交易商户数",
        "近6个月交易商户数",
        "近3个月信贷需求强度",
        "近6个月信贷需求强度",
        "近三个月查询天数",
        "近六个月查询天数",
        "近一个月所有失败交易天数",
        "近三个月所有失败交易天数",
        "近六个月所有失败交易天数",
        "近一个月资金不足天数",
        "近三个月资金不足天数",
        "近六个月资金不足天数",
        "近一个月发生资金不足交易商户数",
        "近三个月发生资金不足交易商户数",
        "近六个月发生资金不足交易商户数",
        "近1个月交易天数",
        "近3个月交易天数",
        "近6个月发生交易月份数",
        "近12个月发生交易月份数",
        "逾期风险评分（借记卡）",
        "逾期风险评分（贷记卡）",
        "消费自由度得分"]

    tra_feature, tra_label = tra[cat_columns + num_columns], tra["target"]
    tes_feature, tes_label = tes[cat_columns + num_columns], tes["target"]

    tidy = TidyTabula(keep_columns=[], cat_columns=[cat_columns], num_columns=num_columns)
    save = SaveMemory(keep_columns=[], cat_columns=[cat_columns], num_columns=num_columns)
    discrete = Discrete(keep_columns=[], cat_columns=[cat_columns], num_columns=num_columns, merge_bin=0.05)

    tidy.fit(tra_feature, tra_label)
    tra_feature = tidy.transform(tra_feature)
    tes_feature = tidy.transform(tes_feature)

    # save.fit(tra_feature, tra_label)
    # tra_feature = save.transform(tra_feature)
    # tes_feature = save.transform(tes_feature)

    discrete.fit(tra_feature, tra_label)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)
    save_table(discrete, config["path"])

    pickle.dump({"feature": tra_feature, "label": tra_label}, open("E:\\work\\QuDian\\tra.pkl", "wb"))
    pickle.dump({"feature": tes_feature, "label": tes_label}, open("E:\\work\\QuDian\\tes.pkl", "wb"))

    # sbin = SelectBin(keep_columns=[])
    # sbin.fit(tra_feature, tra_label)
    # tra_feature = sbin.transform(tra_feature)
    # tes_feature = sbin.transform(tes_feature)
    #
    # svif = SelectVif(keep_columns=[])
    # svif.fit(tra_feature, tra_label)
    # tra_feature = svif.transform(tra_feature)
    # tes_feature = svif.transform(tes_feature)
    #
    # clf = LRClassifier(c=1, keep_columns=[], random_state=7)
    # clf.fit(tra_feature, tra_label)
    # pprint(clf.score(tra_feature, tra_label))
    # pprint(clf.score(tes_feature, tes_label))
    # pprint(clf.result())






