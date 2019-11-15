# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def cv(params, pipeline, X, y, random_state):
    result = {str(param): [] for param in params}

    for param in params:
        sfk = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        clf = pipeline.set_params(model__c=param)

        for tra_idx, val_idx in sfk.split(X, y):
            tra_feature, val_feature = X.iloc[tra_idx], X.iloc[val_idx]
            tra_label, val_label = y.iloc[tra_idx], y.iloc[val_idx]

            clf.fit(tra_feature, tra_label)
            result[str(param)].append(clf.score(val_feature, val_label))

    return result
