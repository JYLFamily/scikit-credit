# encoding: utf-8

import numpy as np
import pandas as pd
from pprint import pprint
from skcredit.linear_model import LMClassifier
from skcredit.monitor import FEndReport, BEndReport
from skcredit.feature_preprocessing import TidyTabula
from skcredit.feature_discretization import save_table, DiscreteCust
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)


if __name__ == "__main__":
    cat_columns = ["user_basic.user_province", "user_basic.user_phone_province"]
    num_columns = [
        "user_searched_history_by_day.m_12.cnt_cc",
        "user_gray.contacts_rfm.call_cnt_to_applied",
        "user_searched_history_by_day.d_90.cnt_org_cash",
        "user_gray.contacts_rfm.time_spent_be_applied",
        "user_gray.contacts_gray_score.be_mean",
        "user_gray.contacts_relation_distribution.be_not_familiar",
        "user_searched_history_by_day.d_7.pct_cnt_org_cash",
        "user_gray.contacts_query.to_org_cnt_3",
        "user_gray.phone_gray_score",
        "user_searched_history_by_day.m_9.cnt_org",
        "user_searched_history_by_day.m_9.pct_cnt_cf",
        "user_gray.contacts_number_statistic.pct_black_ratio",
        "user_gray.contacts_number_statistic.pct_router_ratio",
        "user_basic.user_age",
        "user_searched_history_by_day.m_24.pct_cnt_org_cc",
        "user_gray.contacts_number_statistic.pct_cnt_to_black",
        "user_gray.contacts_query.org_cnt_2"
    ]

    group_dict = {
        "user_basic.user_province": [
            ["上海市", "北京市", "吉林省", "天津市", "宁夏回族自治区", "山西省",
             "新疆维吾尔自治区", "江西省", "河北省", "河南省", "海南省", "西藏自治区"],
            ["云南省", "内蒙古自治区", "四川省", "安徽省", "山东省", "江苏省", "浙江省", "湖北省", "陕西省"],
            ["广东省", "广西壮族自治区", "湖南省", "甘肃省", "福建省", "贵州省", "辽宁省", "重庆市", "青海省", "黑龙江省"]],
        "user_basic.user_phone_province": [
            ["北京", "吉林", "新疆", "海南"],
            ["上海", "云南", "湖北", "西藏"],
            ["内蒙古", "四川", "宁夏", "安徽", "山东", "山西", "江苏", "江西", "河北", "河南", "浙江", "陕西"],
            ["天津", "广东", "湖南", "甘肃", "辽宁", "重庆", "黑龙江"],
            ["广西", "福建", "贵州", "青海"]]
    }

    break_dict = {
        "user_searched_history_by_day.m_12.cnt_cc": pd.IntervalIndex.from_breaks([-np.inf, 1.0, np.inf], closed="left"),
        "user_gray.contacts_rfm.call_cnt_to_applied": pd.IntervalIndex.from_breaks([-np.inf, 7, np.inf], closed="left"),
        "user_searched_history_by_day.d_90.cnt_org_cash": pd.IntervalIndex.from_breaks([-np.inf, 1.0, 2.0, 4.0, 6.0, 11.0, 15.0, np.inf], closed="left"),
        "user_gray.contacts_rfm.time_spent_be_applied": pd.IntervalIndex.from_breaks([-np.inf, 1405.0, 7647.16, np.inf], closed="left"),
        "user_gray.contacts_gray_score.be_mean": pd.IntervalIndex.from_breaks([-np.inf, 44.21, 51.774, np.inf], closed="left"),
        "user_gray.contacts_relation_distribution.be_not_familiar": pd.IntervalIndex.from_breaks([-np.inf, 37.0, np.inf], closed="left"),
        "user_searched_history_by_day.d_7.pct_cnt_org_cash": pd.IntervalIndex.from_breaks([-np.inf, 0.696, 0.871, 0.958, np.inf], closed="left"),
        "user_gray.contacts_query.to_org_cnt_3": pd.IntervalIndex.from_breaks([-np.inf, 10.0, 29.0, np.inf], closed="left"),
        "user_gray.phone_gray_score": pd.IntervalIndex.from_breaks([-np.inf, 3.291, 43.04, np.inf], closed="left"),
        "user_searched_history_by_day.m_9.cnt_org": pd.IntervalIndex.from_breaks([-np.inf, 10.0, 14.0, 31.0, np.inf], closed="left"),
        "user_searched_history_by_day.m_9.pct_cnt_cf": pd.IntervalIndex.from_breaks([-np.inf, 0.601, 0.828, 0.994, np.inf], closed="left"),
        "user_gray.contacts_number_statistic.pct_black_ratio": pd.IntervalIndex.from_breaks([-np.inf, 0.863, np.inf], closed="left"),
        "user_gray.contacts_number_statistic.pct_router_ratio": pd.IntervalIndex.from_breaks([-np.inf, 0.366, 0.666, 0.827, 0.967, np.inf], closed="left"),
        "user_basic.user_age": pd.IntervalIndex.from_breaks([-np.inf, 27.0, 32.0, np.inf], closed="left"),
        "user_searched_history_by_day.m_24.pct_cnt_org_cc": pd.IntervalIndex.from_breaks([-np.inf, 0.697, 0.768, 0.996, np.inf], closed="left"),
        "user_gray.contacts_number_statistic.pct_cnt_to_black": pd.IntervalIndex.from_breaks([-np.inf, 0.83, np.inf], closed="left"),
        "user_gray.contacts_query.org_cnt_2": pd.IntervalIndex.from_breaks([-np.inf, 6.0, 17.0, 27.0, np.inf], closed="left")
    }

    tra = pd.read_csv("H:\\work\\ZhongHe\\tra_sy.csv", encoding="GBK")
    tes = pd.read_csv("H:\\work\\ZhongHe\\tes_sy.csv", encoding="GBK")

    tra_tabular = tra[cat_columns + num_columns].copy(deep=True)
    tes_tabular = tes[cat_columns + num_columns].copy(deep=True)

    tra_feature, tra_label = tra[cat_columns + num_columns].copy(deep=True), tra["target"].copy(deep=True)
    tes_feature, tes_label = tes[cat_columns + num_columns].copy(deep=True), tes["target"].copy(deep=True)
    del tra, tes

    tidy = TidyTabula(keep_columns=[], cat_columns=cat_columns, num_columns=num_columns)
    tidy.fit(tra_feature, tra_label)
    tra_feature = tidy.transform(tra_feature)
    tes_feature = tidy.transform(tes_feature)

    discrete = DiscreteCust(keep_columns=[], group_dict=group_dict, break_dict=break_dict)
    discrete.fit(tra_feature, tra_label)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)
    save_table(discrete, "C:\\Users\\15795\\Desktop")

    model = LMClassifier(C=0.05, PDO=20, BASE=600, ODDS=1, keep_columns=[], random_state=7)
    model.fit(tra_feature, tra_label)
    pprint(model.result())

    result = FEndReport.psi(discrete, model, tidy.transform(tra_tabular), tidy.transform(tes_tabular))
    print("=" * 36 + "PSI" + "=" * 36)
    print(result["table"])
    print(result["psi"])
    print("=" * 75)

    result = FEndReport.csi(discrete, model, tidy.transform(tra_tabular), tidy.transform(tes_tabular))
    for col in result.keys():
        print("=" * 36 + "CSI" + "=" * 36)
        print(col)
        print(result[col]["table"])
        print(result[col]["csi"])
        print("=" * 75)

    result = BEndReport.scores(
        discrete,
        model,
        tidy.transform(tra_tabular),
        tra_label,
        tidy.transform(tes_tabular),
        tes_label
    )
    print("=" * 36 + "scores" + "=" * 36)
    print(result["tra"])
    print(result["tes"])
    print("=" * 78)

    result = BEndReport.metric(
        discrete,
        model,
        tidy.transform(tra_tabular),
        tra_label,
        tidy.transform(tes_tabular),
        tes_label
    )
    print("=" * 36 + "metric" + "=" * 36)
    print(result["tra"]["ks"], result["tra"]["auc"])
    print(result["tes"]["ks"], result["tes"]["auc"])
    print("=" * 78)

    result = BEndReport.report(
        discrete,
        model,
        tidy.transform(tra_tabular),
        tra_label,
        tidy.transform(tes_tabular),
        tes_label
    )
    print("=" * 36 + "report" + "=" * 36)
    print(result["tra"])
    print(result["tes"])
    print("=" * 78)