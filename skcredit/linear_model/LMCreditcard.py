# coding:utf-8

import numpy  as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)


class LMCreditcard(object):
    def __init__(self, keep_columns, date_columns, discrete, lmclassifier, BASE,  PDO,  ODDS):
        self.keep_columns = keep_columns
        self.date_columns = date_columns
        self.discrete, self.lmclassifier = discrete, lmclassifier

        self.B_ = PDO / np.log(2)
        self.A_ = BASE + self.B_ * np.log(ODDS)

        self.offset_scores = self.A_ - self.B_ * self.lmclassifier.coeff_["const"]
        self.credit_scores = dict()

    def show_scorecard(self):
        tables = dict()
        tables.update({column: spliter.table for column, spliter in self.discrete.cat_spliter.items()})
        tables.update({column: spliter.table for column, spliter in self.discrete.num_spliter.items()})

        for column in self.lmclassifier.feature_subsets_:
            table = tables[column][["Column", "Bucket", "WoE"]].copy(deep=True)
            table["Coefficients"] =   self.lmclassifier.coeff_[column]
            table["PartialScore"] = - self.lmclassifier.coeff_[column] * self.B_ * table["WoE"]
            table["OffsetScores"] = self.offset_scores
            self.credit_scores[column] = table

        return pd.concat(self.credit_scores.values())

