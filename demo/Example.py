# encoding: utf-8

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from skcredit.feature_discrete import Discrete
from skcredit.feature_discrete import save_table
from skcredit.feature_selection import SelectBIN
from skcredit.feature_selection import SelectVIF
from sklearn.linear_model import LogisticRegression
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    tra = pd.read_csv(os.path.join(config["path"], config["dataset"]["diff"]), na_values=[r"\N"])
    tes = pd.read_csv(os.path.join(config["path"], config["dataset"]["same"]), na_values=[r"\N"])

    cat_columns = ["province", "借贷标记"]
    num_columns = [col for col in tra.columns if col.endswith(("金额", "笔数"))]
    num_columns = num_columns + [
        "近1个月信贷需求强度",
        "近3个月信贷需求强度",
        "近6个月信贷需求强度",
        "近一个月查询天数",
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
        "近一个月密码错误天数",
        "近三个月密码错误天数",
        "近1个月交易天数",
        "近3个月交易天数",
        "近6个月发生交易月份数",
        "近12个月发生交易月份数",
        "逾期风险评分（借记卡）",
        "逾期风险评分（贷记卡）"
    ]

    encoder = TargetEncoder(cols=cat_columns)
    encoder.fit(tra[cat_columns], tra["target"])
    tra[cat_columns] = encoder.transform(tra[cat_columns])
    tes[cat_columns] = encoder.transform(tes[cat_columns])

    # feature, label
    tra_feature, tra_label = tra[cat_columns + num_columns], tra["target"]
    tes_feature, tes_label = tes[cat_columns + num_columns], tes["target"]

    # discrete
    discrete = Discrete(
        keep_columns=[],
        cat_columns=None,
        num_columns=cat_columns + num_columns,
        merge_gap=0.2,
        merge_bin=0.05,
        information_value_threshold=0.1
    )
    discrete.fit(tra_feature, tra_label)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)
    save_table(discrete, config["path"])

    sbin = SelectBIN(keep_columns=[])
    sbin.fit(tra_feature, tra_label)
    tra_feature = sbin.transform(tra_feature)
    tes_feature = sbin.transform(tes_feature)

    svif = SelectVIF(keep_columns=[], vif_threshold=2.5)
    svif.fit(tra_feature, tra_label)
    tra_feature = svif.transform(tra_feature)
    tes_feature = svif.transform(tes_feature)

    model = LogisticRegression(C=0.025, solver="lbfgs", random_state=7)
    model.fit(tra_feature, tra_label)
    print(tra_feature.columns)
    print(model.coef_)
    print(roc_auc_score(tra_label, model.predict_proba(tra_feature)[:, 1]))
    print(roc_auc_score(tes_label, model.predict_proba(tes_feature)[:, 1]))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(tes_label, model.predict_proba(tes_feature)[:, 1])
    print(np.max(tpr - fpr))