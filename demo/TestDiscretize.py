# coding:utf-8

import os
import numpy as np
import pandas as pd
from skcredit.feature_discretize.Discretize import Discretize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
np.random.seed(7)


class TestDiscretize(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_label = [None for _ in range(2)]

        self.__discrete = None
        self.__linear_model = None

    def read_data(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "train_subset.csv"), encoding="GBK")
        self.__test = pd.read_csv(os.path.join(self.__path, "test_subset.csv"), encoding="GBK")

    def prepare_data(self):
        self.__train_feature = self.__train[[
            "user_gray.contacts_relation_distribution.be_not_familiar",
            "user_gray.contacts_rfm.time_spent_be_applied",
            "user_gray.phone_gray_score",
            "user_gray.contacts_gray_score.be_mean",
            "user_searched_history_by_day.d_7.pct_cnt_org_cash",
            "user_gray.contacts_number_statistic.pct_cnt_to_black",
            "user_searched_history_by_day.m_12.cnt_cc",
            "user_gray.contacts_number_statistic.pct_router_ratio",
            "user_searched_history_by_day.d_90.cnt_org_cash",
            "user_basic.user_province",
            "user_searched_history_by_day.m_9.cnt_org",
            "user_gray.contacts_query.to_org_cnt_3",
            "user_gray.contacts_number_statistic.pct_black_ratio",
            "user_searched_history_by_day.m_9.pct_cnt_cf", "user_basic.user_age",
            "user_gray.contacts_rfm.call_cnt_to_applied",
            "user_gray.contacts_gray_score.max",
            "user_basic.user_phone_province",
            "user_searched_history_by_day.m_24.pct_cnt_org_cc",
            "user_gray.contacts_query.org_cnt_2"
        ]].copy(deep=True)
        self.__test_feature = self.__test[self.__train_feature.columns].copy(deep=True)
        self.__train_label = self.__train["target"].copy(deep=True)
        self.__test_label = self.__test["target"].copy(deep=True)

        del self.__train, self.__test

    def model_fit(self):
        cat_columns = ["user_basic.user_province", "user_basic.user_phone_province"]
        num_columns = [col for col in self.__train_feature.columns if col not in cat_columns]

        self.__discrete = Discretize(
            cat_columns=cat_columns,
            num_columns=num_columns,
            threshold=0
        )
        self.__discrete.fit(self.__train_feature, self.__train_label)
        self.__train_feature = self.__discrete.transform(self.__train_feature)
        self.__test_feature = self.__discrete.transform(self.__test_feature)

    def model_predict(self):
        self.__linear_model = LogisticRegression(C=0.025, solver="lbfgs")
        self.__linear_model.fit(self.__train_feature, self.__train_label)
        print("train set auc {}".format(roc_auc_score(
            self.__train_label, self.__linear_model.predict_proba(self.__train_feature)[:, 1])))
        print("test set auc {}".format(roc_auc_score(
            self.__test_label, self.__linear_model.predict_proba(self.__test_feature)[:, 1])))
        print(self.__linear_model.coef_)


if __name__ == "__main__":
    td = TestDiscretize(path="D:\\Work\\Data\\WeCash")
    td.read_data()
    td.prepare_data()
    td.model_fit()
    td.model_predict()