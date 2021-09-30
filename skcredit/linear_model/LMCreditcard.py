# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class LMCreditcard(object):
    def __init__(self, keep_columns, date_columns, discrete, lmclassifier, BASE,  PDO,  ODDS):
        self.keep_columns = keep_columns
        self.date_columns = date_columns
        self.discrete, self.lmclassifier = discrete, lmclassifier

        self.B = PDO / np.log(2)
        self.A = BASE + self.B * np.log(ODDS)

        self.offset_scores = self.A - self.B * self.lmclassifier.coeff_["const"]
        self.credit_scores = dict()

    def show_scorecard(self):
        tables = dict()
        tables.update({column: spliter.table for column, spliter in self.discrete.cat_spliter.items()})
        tables.update({column: spliter.table for column, spliter in self.discrete.num_spliter.items()})

        for column in self.lmclassifier.feature_subsets_:
            table = tables[column][["Column", "Bucket", "WoE"]].copy(deep=True)
            table["Coefficients"] =   self.lmclassifier.coeff_[column]
            table["PartialScore"] = - self.lmclassifier.coeff_[column] * self.B * table["WoE"]
            table["OffsetScores"] = self.offset_scores
            self.credit_scores[column] = table

        return pd.concat(self.credit_scores.values())

