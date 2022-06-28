# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
from heapq import heappush
import statsmodels.api as sm
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


def inclusion(x, y, feature_subsets, feature_remains):
    feature_metrics = list()

    for feature in feature_remains:
        logit_res = sm.GLM(
            y, sm.add_constant(x[feature_subsets | {feature}], has_constant="add"),
            family=sm.families.Binomial()).fit()

        if not (logit_res.params.drop("const") < 0).any() and \
                not (logit_res.pvalues.drop("const") >= 0.05).any():
            fpr, tpr, _ = roc_curve(
                y, logit_res.predict(sm.add_constant(x[feature_subsets | {feature}])))
            heappush(feature_metrics, (1 - max(tpr - fpr), feature))

    return (feature_metrics[0][1], 1 - feature_metrics[0][0]) if feature_metrics else (None, None)


def exclusion(x, y, feature_subsets):
    feature_metrics = list()

    for feature in feature_subsets:
        logit_res = sm.GLM(
            y, sm.add_constant(x[feature_subsets - {feature}], has_constant="add"),
            family=sm.families.Binomial()).fit()

        if not (logit_res.params.drop("const") < 0).any() and \
                not (logit_res.pvalues.drop("const") >= 0.05).any():
            fpr, tpr, _ = roc_curve(
                y, logit_res.predict(sm.add_constant(x[feature_subsets - {feature}])))
            heappush(feature_metrics, (1 - max(tpr - fpr), feature))

    return (feature_metrics[0][1], 1 - feature_metrics[0][0]) if feature_metrics else (None, None)


def stepwises(x, y, feature_columns, feature_subsets):
    best_ks = float("-inf")

    while True:
        include_feature, curr_ks =     inclusion(
                x,
                y,
                feature_subsets,
                feature_columns - feature_subsets
        )

        if      include_feature is None or curr_ks <= best_ks:
            break
        else:
            best_ks =     curr_ks
            feature_subsets     =  feature_subsets | {include_feature}

        while True:
            exclude_feature, curr_ks = exclusion(
                x,
                y,
                feature_subsets
            )

            if exclude_feature is None or curr_ks <= best_ks:
                break
            else:
                best_ks = curr_ks
                feature_subsets =  feature_subsets - {exclude_feature}

    return sm.GLM(
            y, sm.add_constant(x[feature_subsets], has_constant="add"),
            family=sm.families.Binomial()).fit( )


class LMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keep_columns, date_columns):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.model = None
        self.coeff = None

        self.feature_columns = None
        self.feature_subsets = None

    def fit(  self, x, y):
        self.feature_columns = np.array(
            [col for col in x.columns if col not in  self.keep_columns and col not in self.date_columns])

        self.model = stepwises(x, y, set(self.feature_columns), set())

        self.coeff  =  self.model.params
        self.feature_subsets = np.array(
            self.coeff.drop("const").index.tolist())

        return self

    def score(self, x, y,      sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(x)["proba_positive"], sample_weight=sample_weight)

        return round(max(tpr - fpr), 5)

    def predict_proba(self, x):
        proba_positive = self.model.predict(
              sm.add_constant(x[self.feature_subsets])).to_frame("proba_positive")
        proba_negative = (1 - proba_positive.squeeze()).to_frame("proba_negative")

        return pd.concat([proba_negative, proba_positive], axis="columns")


if __name__ == "__main__":
    from skcredit.feature_discrete import SplitMixND
    from skcredit.feature_discrete import CXDiscrete
    from skcredit.linear_model     import LMCreditcard

    application_train = pd.read_csv("C:\\Users\\Administrator\\Desktop\\application_train.csv").head(100000)

    application_train_input = application_train.drop("TARGET", axis=1)
    application_train_label = application_train["TARGET"]

    num_columns = application_train_input.select_dtypes(exclude="object").columns.tolist()
    cat_columns = application_train_input.select_dtypes(include="object").columns.tolist()

    application_train_input[cat_columns] = application_train_input[cat_columns].astype("category")

    cx = CXDiscrete(
        keep_columns=[],
        date_columns=[],
        transformers=[(SplitMixND(), [col]) for col in application_train_input.columns],
        iv_threshold=0.02, nthread=-1, verbose=20)
    cx.fit(
        application_train_input,
        application_train_label,
    )
    application_train_input = cx.transform(application_train_input)

    lm = LMClassifier([], [])
    lm.fit(
        application_train_input,
        application_train_label,
    )

    lm = LMCreditcard([], [], cx, lm, 500,    200,    application_train_label.sum() / (application_train_label.shape[0] - application_train_label.sum()))
    print(lm.show_scorecard())


