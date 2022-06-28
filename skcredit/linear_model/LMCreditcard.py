# coding:utf-8

import warnings
import numpy  as np
import pandas as pd
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class LMCreditcard(object):
    def __init__(self, keep_columns, date_columns, cx, lm, BASE, PDO, ODDS):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.cx = cx
        self.lm = lm

        self.B =                         PDO / np.log(2)
        self.A = BASE + (PDO / np.log(2)) * np.log(ODDS)

        self.offset_scores = self.A - self.B * self.lm.coeff[  "const"]
        self.credit_scores = dict()

    def show_scorecard(self):
        for column in self.lm.feature_subsets:
            table = self.cx.named_transformers_table[column][["Column", "Bucket", "WoE"]].copy(deep=True)
            table = table.iloc[:-1]

            table["Coefficients"] =    self.lm.coeff[column]
            table["PartialScore"] = -  (self.lm.coeff[column] * self.B *    table["WoE"]).astype(    int)
            table["OffsetScores"] =    int(self.offset_scores)
            self.credit_scores[column] = table

        return pd.concat(self.credit_scores.values())

