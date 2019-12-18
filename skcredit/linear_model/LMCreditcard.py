# coding: utf-8

import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LMCreditcard(object):
    def __init__(self, *, discrete, lmclassifier):
        self.discrete = discrete
        self.lmclassifier = lmclassifier