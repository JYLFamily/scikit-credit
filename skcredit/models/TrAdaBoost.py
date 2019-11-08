# coding:utf-8

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LearnerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, random_stated):
        self.__learner = None
        self.__polarity = None
        self.__random_stated = random_stated

    def fit(self, feature, label, sample_weight):
        self.__learner = GradientBoostingClassifier(random_state=self.__random_stated)
        self.__learner.fit(feature, label, sample_weight=sample_weight)

        return self

    def predict(self, feature):
        return self.__learner.predict(feature)

    def predict_proba(self, feature):
        return self.__learner.predict_proba(feature)

    def set_polarity(self, *, polarity):
        self.__polarity = polarity

    def get_polarity(self):
        return self.__polarity


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, n_iterations):
        self.__n_iterations = n_iterations

        self.__learner_list = []
        self.__beta_iter_list = []

    def fit(self, feature_diff, label_diff, feature_same, label_same):
        num_diff = feature_diff.shape[0]

        feature = np.concatenate([feature_diff, feature_same])
        label = np.concatenate([label_diff, label_same])

        beta = 1 / (1 + np.sqrt(2 * np.log(num_diff / self.__n_iterations)))
        sample_weight = compute_sample_weight(class_weight="balanced", y=label)

        for _ in range(self.__n_iterations):
            sample_weight = sample_weight / sample_weight.sum()

            learner = LearnerClassifier(random_stated=7)
            learner.fit(feature, label, sample_weight)
            predict = learner.predict(feature)

            error = ((sample_weight[num_diff:] * np.abs(predict[num_diff:] - label[num_diff:])).sum() /
                     sample_weight[num_diff:].sum())

            if error > 0.5:
                error = 1 - error

                learner.set_polarity(polarity=-1)
                self.__learner_list.append(learner)
            else:
                learner.set_polarity(polarity=+1)
                self.__learner_list.append(learner)

            beta_t = error / (1 - error)
            self.__beta_iter_list.append(beta_t)

            sample_weight[:num_diff] = (
                    sample_weight[:num_diff] * np.power(beta, np.abs(predict[:num_diff] - label[:num_diff])))
            sample_weight[num_diff:] = (
                    sample_weight[num_diff:] * np.power(beta_t, - np.abs(predict[num_diff:] - label[num_diff:])))

        return self

    def predict(self, feature):
        predict = self.decision_function(feature)

        return np.where(predict >= 0, 1, 0)

    def predict_proba(self, feature):
        predict_proba = np.ones((feature.shape[0], 2))
        predict_proba[:, 1] = expit(self.decision_function(feature))
        predict_proba[:, 0] -= predict_proba[:, 1]

        return predict_proba

    def decision_function(self, feature):
        l_part = np.zeros((feature.shape[0], np.int(self.__n_iterations - np.ceil(self.__n_iterations / 2))))
        r_part = np.zeros((feature.shape[0], np.int(self.__n_iterations - np.ceil(self.__n_iterations / 2))))

        for i, t in enumerate(range(np.int(np.ceil(self.__n_iterations / 2)), self.__n_iterations)):
            polarity = self.__learner_list[t].get_polarity()

            if polarity == -1:
                predict = self.__learner_list[t].predict(feature)
                predict = np.where(predict == 0, 1, 0)
            else:
                predict = self.__learner_list[t].predict(feature)

            l_part[:, i] = np.power(self.__beta_iter_list[t], - predict)
            r_part[:, i] = np.power(self.__beta_iter_list[t], - 1 / 2)

        return l_part.prod(axis=1) - r_part.prod(axis=1)

