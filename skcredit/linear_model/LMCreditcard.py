# coding:utf-8

import numpy  as np
import pandas as pd
from sympy import Interval
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LMCreditcard(object):
    def __init__(self, tim_columns, discrete, lmclassifier,  BASE,  PDO,  ODDS):
        self.tim_columns = tim_columns
        self.discrete, self.lmclassifier = discrete, lmclassifier

        self.B_ = PDO / np.log(2)
        self.A_ = BASE + self.B_ * np.log(ODDS)

        self.OffsetScores = self.A_ - self.B_ * self.lmclassifier.coeff_["const"]

        self.scorecard_dict = dict()
        self.scorecard_form = pd.DataFrame(
            columns=["Column", "Bucket", "Coefficients", "PartialScore", "OffsetScores"])

        tables = dict()
        tables.update(self.discrete.cat_table_)
        tables.update(self.discrete.num_table_)

        for col in self.lmclassifier.feature_subsets_:
            table = tables[col][[col, "WoE"]].copy(deep=True)
            table["Coefficients"] =   lmclassifier.coeff_[col]
            table["PartialScore"] = - lmclassifier.coeff_[col] * self.B_ * table["WoE"]
            self.scorecard_dict[col] = table

    def predict_parts(self, x):
        result = x[self.tim_columns].copy(deep=True)

        for col in self.lmclassifier.feature_subsets_:
            result[col] = x[col].replace(self.scorecard_dict[col]["WoE"].tolist(),
                                self.scorecard_dict[col]["PartialScore"].tolist())

        return result

    def predict_score(self, x):
        result = x[self.tim_columns].copy(deep=True)
        probas = self.lmclassifier.predict_proba(x )

        result["CreditScores"] = self.A_ - self.B_ * np.log(probas["proba_positive"] /
                                                           (probas["proba_negative"]))

        return result

    def plot_sample(self, x):
        pass

    def plot_global(self, x):
        pass

    def save_scorecard(self):
        for col, table in self.scorecard_dict.items():
            result = pd.DataFrame(index=np.arange(len(table)),
                columns=["Column", "Bucket", "Coefficients", "PartialScore", "OffsetScores"])

            result["Column"] = col
            result["Bucket"] = table[col].apply(lambda element:
                "[{:.3f} <= {:.3f})".format(np.round(np.float_(element.start), 6), np.round(np.float_(element.end), 6))
                if isinstance(element, Interval) else "NA")
            result["Coefficients"] = table["Coefficients"]
            result["PartialScore"] = table["PartialScore"].round()
            result["OffsetScores"] = round(self.OffsetScores)
            self.scorecard_form = self.scorecard_form.append(result)

        print(self.scorecard_form)

