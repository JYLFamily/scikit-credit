# coding:utf-8

import logging
import numpy  as np
import pandas as pd
import statsmodels.api as sm
from heapq import heappush
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def train(x, y):
    logit_mod = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
    logit_res = logit_mod.fit()

    return logit_res


def inclusion(x, y, eval_x, eval_y, feature_subsets, feature_remains):
    feature_metrics = list()

    for feature in feature_remains:
        logit_res = train(x[list(feature_subsets | {feature})], y)

        # if not (logit_res.params.drop("const") < 0).any() and \
        #         not (logit_res.pvalues.drop("const") >= 0.05).any():
        #     fpr, tpr, _ = roc_curve(
        #         eval_y, logit_res.predict(sm.add_constant(eval_x[list(feature_subsets | {feature})])))
        #     heappush(feature_metrics, (1 - max(tpr - fpr), feature))
        fpr, tpr, _ = roc_curve(
            eval_y, logit_res.predict(sm.add_constant(eval_x[list(feature_subsets | {feature})])))
        heappush(feature_metrics, (1 - max(tpr - fpr), feature))
    # return (feature_metrics[0][1], 1 - feature_metrics[0][0]) if feature_metrics else (None, None)

    return feature_metrics[0][1], 1 - feature_metrics[0][0]


def exclusion(x, y, eval_x, eval_y, feature_subsets):
    feature_metrics = list()

    for feature in feature_subsets:
        logit_res = train(x[list(feature_subsets - {feature})], y)

        # if not (logit_res.params.drop("const") < 0).any() and \
        #         not (logit_res.pvalues.drop("const") >= 0.05).any():
        #     fpr, tpr, _ = roc_curve(
        #         eval_y, logit_res.predict(sm.add_constant(eval_x[list(feature_subsets - {feature})])))
        #     heappush(feature_metrics, (1 - max(tpr - fpr), feature))
        fpr, tpr, _ = roc_curve(
            eval_y, logit_res.predict(sm.add_constant(eval_x[list(feature_subsets - {feature})])))
        heappush(feature_metrics, (1 - max(tpr - fpr), feature))

    # return (feature_metrics[0][1], 1 - feature_metrics[0][0]) if feature_metrics else (None, None)
    return feature_metrics[0][1], 1 - feature_metrics[0][0]


def stepwises(x, y, eval_x, eval_y, feature_columns, feature_subsets):
    best_ks = np.finfo(np.float64).min

    stop_include_flag = False

    while not stop_include_flag:
        include_feature, curr_ks = inclusion(
            x, y, eval_x, eval_y,
            feature_subsets,
            feature_columns - feature_subsets)

        if include_feature is None or curr_ks <= best_ks:
            stop_include_flag = True
        else:
            best_ks = curr_ks
            feature_subsets = feature_subsets | {include_feature}
        print(feature_subsets)
        stop_exclude_flag = False

        while not stop_exclude_flag:
            exclude_feature, curr_ks = exclusion(
                x, y, eval_x, eval_y,
                feature_subsets)

            if stop_exclude_flag is None or curr_ks <= best_ks:
                stop_exclude_flag = True
            else:
                best_ks = curr_ks
                feature_subsets = feature_subsets - {exclude_feature}
        print(feature_subsets)

    return train(x[list(feature_subsets)], y)


class LMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns

        self.feature_subsets_ = None
        self.model_ = None
        self.coeff_ = None

    def fit(self, x, y, eval_x, eval_y):
        self.model_ = stepwises(
            x, y, eval_x, eval_y,
            set([col for col in x.columns if col not in self.tim_columns]),
            set(),
        )

        self.coeff_ = self.model_.params
        self.feature_subsets_ = self.coeff_.drop("const").index.tolist()

        return self

    def model(self):
        return self.model_.summary()

    def score(self, x, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(x)["proba_positive"])

        return round(max(tpr - fpr), 5)

    def predict_proba(self, x):
        proba_positive = self.model_.predict(sm.add_constant(x[self.feature_subsets_])).to_frame("proba_positive")
        proba_negative = (1 - proba_positive.squeeze()).to_frame("proba_negative")

        return pd.concat([proba_negative, proba_positive], axis=1)

