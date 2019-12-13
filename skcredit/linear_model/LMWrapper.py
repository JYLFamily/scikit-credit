# coding: utf-8

import gc
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool
from skcredit.linear_model import ks_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class LMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C, keep_columns, random_state):
        self.C = C
        self.keep_columns = keep_columns
        self.random_state = random_state

        self.feature_columns_ = None
        self.feature_subsets_ = None

    def __calc(self, x, y, subsets):
        clf = LogisticRegression(C=self.C, solver="lbfgs", random_state=self.random_state)
        clf.fit(x[list(subsets)], y)

        return ks_score(clf, x[list(subsets)], y), np.all(clf.coef_ > 0)

    def __inclusion(self, x, y, columns, subsets):
        remains_feature, remains_score = [np.array([]) for _ in range(2)]
        feature = None

        if subsets:
            cur_score, _ = self.__calc(x, y, subsets)

            remains = columns - set(subsets)

            if remains:
                for remain in remains:
                    new_score, sign = self.__calc(x, y, subsets | {remain})

                    if new_score > cur_score and sign:
                        remains_feature = np.append(remains_feature, np.array([remain]))
                        remains_score = np.append(remains_score, np.array([new_score]))

                if len(remains_feature) != 0:
                    feature = remains_feature[remains_score.argmax()]
        else:
            remains = columns - set(subsets)

            if remains:
                for remain in remains:
                    new_score, sign = self.__calc(x, y, subsets | {remain})

                    if sign:
                        remains_feature = np.append(remains_feature, np.array([remain]))
                        remains_score = np.append(remains_score, np.array([new_score]))

                if len(remains_feature) != 0:
                    feature = remains_feature[remains_score.argmax()]

        return feature

    def __exclusion(self, x, y, subsets, fixed):
        subsets_feature, subsets_score = [np.array([]) for _ in range(2)]
        feature = None

        cur_score, _ = self.__calc(x, y, subsets)

        if len(subsets) > 2:
            for subset in subsets if fixed is None else subsets - {fixed}:
                new_score, sign = self.__calc(x, y, subsets - {subset})

                if new_score > cur_score and sign:
                    subsets_feature = np.append(subsets_feature, np.array([subset]))
                    subsets_score = np.append(subsets_score, np.array([new_score]))

                if len(subsets_feature) != 0:
                    feature = subsets_feature[subsets_score.argmax()]

        return feature

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = set(x.columns.tolist())
        self.feature_subsets_ = set()

        feature_in = self.__inclusion(x, y, self.feature_columns_, self.feature_subsets_)

        while feature_in:
            self.feature_subsets_ = self.feature_subsets_ | {feature_in}
            feature_ex = self.__exclusion(x, y, self.feature_subsets_, feature_in)

            while feature_ex:
                self.feature_subsets_ = self.feature_subsets_ - {feature_ex}
                feature_ex = self.__exclusion(x, y, self.feature_subsets_, feature_in)

            feature_in = self.__inclusion(x, y, self.feature_columns_, self.feature_subsets_)
            print(self.feature_subsets_)

        return self


if __name__ == "__main__":
    # import pickle
    # tra = pickle.load(open("F:\\work\\QuDian\\tra.pkl", "rb"))
    # tes = pickle.load(open("F:\\work\\QuDian\\tes.pkl", "rb"))

    # lm = LMWrapper(C=1, keep_columns=[], random_state=7)
    # lm.fit(tra["feature"], tra["label"])

    # tra_feature, tra_label = tra["feature"], tra["label"]
    # tes_feature, tes_label = tes["feature"], tes["label"]
    #
    # tra_feature = tra_feature[
    #     ['avg_frd_zms', 'credit_pay_amt_3m', 'last_1m_avg_asset_total', 'adr_stability_days', 'zmscore', 'car_trd_amt_1y', 'relevant_stability', 'ebill_pay_amt_1m', 'positive_biz_cnt_1y', 'my_trd_amt_1y']
    # ]
    # tes_feature = tes_feature[tra_feature.columns]
    #
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import roc_curve
    # clf = LogisticRegression(C=1, solver="lbfgs", random_state=7)
    # clf.fit(tra_feature, tra_label)
    # print(clf.coef_)
    # fpr, tpr, _ = roc_curve(tra_label, clf.predict_proba(tra_feature)[:, 1])
    # print(np.max(tpr - fpr))
    # fpr, tpr, _ = roc_curve(tes_label, clf.predict_proba(tes_feature)[:, 1])
    # print(np.max(tpr - fpr))
    tra = pd.read_csv("F:\\tra_woe.csv", encoding="GBK")
    tes = pd.read_csv("F:\\tes_woe.csv", encoding="GBK")

    tra_feature, tra_label = tra.drop(["target"], axis=1), tra["target"]
    tes_feature, tes_label = tes.drop(["target"], axis=1), tes["target"]

    from mlxtend.feature_selection import SequentialFeatureSelector
    sffs = SequentialFeatureSelector(
        estimator=LogisticRegression(C=1, solver="lbfgs", random_state=7),
        k_features=10,
        forward=True, floating=False,
        verbose=2,
        cv=None
    )
    sffs.fit(tra_feature, tra_label)


