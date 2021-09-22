# coding:utf-8
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
np.random.seed(7)
pd.set_option("max_rows",    None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    dataset = pd.read_csv("../UCI_Credit_Card.csv")

    # 数据准备
    cat_columns = ["SEX", "EDUCATION", "MARRIAGE"]
    num_columns = ["LIMIT_BAL", "AGE",
                   "PAY_0",     "PAY_2",     "PAY_3",     "PAY_4",     "PAY_5",     "PAY_6",
                   "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                   "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]

    target = "default.payment.next.month"

    train_x, test_x, train_y, test_y = train_test_split(
        dataset.drop([target], axis=1), dataset[target], train_size=0.75, shuffle=True, random_state=7)

    # 类别特征使用 missing 填充缺失值 连续变量使用 -999999.0 填充缺失值
    from skcredit.feature_preprocessings import FormatTabular
    ft = FormatTabular(keep_columns=["ID"], date_columns=[], cat_columns=cat_columns, num_columns=num_columns)
    ft.fit(train_x, train_y)
    train_x = ft.transform(train_x)
    test_x  = ft.transform(test_x )

    # 模型训练
    from skcredit.feature_discretization import Discrete
    discrete = Discrete(keep_columns=["ID"], date_columns=[])
    discrete.fit(train_x, train_y)
    train_x = discrete.transform(train_x)
    test_x  = discrete.transform(test_x )

    from skcredit.feature_selection import SelectBins
    s = SelectBins(keep_columns=["ID"], date_columns=[])
    s.fit(train_x, train_y)
    train_x = s.transform(train_x)
    test_x =  s.transform(test_x )

    from skcredit.feature_selection import SelectCIFE
    s = SelectCIFE(keep_columns=["ID"], date_columns=[], nums_feature=10)
    s.fit(train_x, train_y)
    train_x = s.transform(train_x)
    test_x =  s.transform(test_x )

    from skcredit.linear_model import LMClassifier
    lmclassifier = LMClassifier(keep_columns=["ID"], date_columns=[])
    lmclassifier.fit(train_x, train_y)
    print(lmclassifier.score(train_x, train_y))
    print(lmclassifier.score(test_x,  test_y ))

    from skcredit.linear_model import LMCreditcard
    lmcreditcard = LMCreditcard(
        keep_columns=["ID"], date_columns=[], discrete=discrete, lmclassifier=lmclassifier, BASE=500,  PDO=20,  ODDS=1)
    print(lmcreditcard.show_scorecard())

