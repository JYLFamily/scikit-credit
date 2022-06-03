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
    def __init__(self, keep_columns, date_columns, c1, cx, lm, BASE,  PDO,  ODDS):
        self.keep_columns = keep_columns
        self.date_columns = date_columns

        self.c1 = c1
        self.cx = cx
        self.lm = lm

        self.B, self.A = PDO / np.log(2), BASE + (PDO / np.log(2)) * np.log(ODDS)

        self.offset_scores = self.A    -    self.B    *    self.lm.coeff["const"]
        self.credit_scores = dict()

    def show_scorecard(self):
        tables = dict()
        tables.update(
            {column: spliter.build_table().iloc[:-1, :] for column, spliter in self.c1.feature_spliter.items()})
        tables.update(
            {column: spliter.build_table().iloc[:-1, :] for column, spliter in self.cx.feature_spliter.items()})

        for column in self.lm.feature_subsets:
            table = tables[column][["Column", "Bucket", "WoE"]].copy(    deep=True)
            table["Coefficients"] =   self.lm.coeff[column]
            table["PartialScore"] = - self.lm.coeff[column] * self.B * table["WoE"]
            table["OffsetScores"] =   self.offset_scores
            self.credit_scores[column] = table

        return pd.concat(self.credit_scores.values())

