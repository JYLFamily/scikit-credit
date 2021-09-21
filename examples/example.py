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
                   "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                   "PAY_AMT1" , "PAY_AMT2" , "PAY_AMT3" , "PAY_AMT4" , "PAY_AMT5" , "PAY_AMT6" ]
    target = "default.payment.next.month"
    # 分类变量 missing 填充 数值变量 -999999.0 填充
    dataset[cat_columns] = dataset[cat_columns].fillna("missing").astype(str)
    dataset[num_columns] = dataset[num_columns].fillna(-999999.0)

    train_x, test_x, train_y, test_y = train_test_split(
        dataset.drop(["ID", target], axis=1), dataset[target], train_size=0.75, shuffle=True, random_state=7)

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
    lm = LMClassifier(keep_columns=["ID"], date_columns=[])
    lm.fit(train_x, train_y)
    print(lm.score(train_x, train_y))
    print(lm.score(test_x,  test_y ))



