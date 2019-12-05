# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def ks_score(estimator, X, y):
    fpr, tpr, _ = roc_curve(y, estimator.predict_proba(X)[:, 1])

    return np.max(tpr - fpr)