# scikit-credit
scikit-credit 是用于信贷风控评分卡建模的自动化工具。如下是 scikit-credit 的简介：
* 自动分箱
* 特征选择
* 模型训练 
* 评分生成

# 特性
* 自动分箱：分箱算法使用以信息量（IV）为分裂标准的分类与回归树（CART）。分类与回归树（CART）分裂前自动探测特征与特征WOE之间的单调关系，分裂过程中借鉴了XGBoost限制单调性的实现，使得分箱结果特征与特征WOE单调。
* 特征选择：
* 模型训练：使用以KS为标准的逐步回归，保证入模特征回归系数为正且统计显著。

# 玩家
信贷风控模型分析师

# 示例

## 完整代码请参见 [example.ipynb](https://github.com/JYLFamily/scikit-credit/blob/master/examples/example.ipynb)

## 单特征分箱

### 分类特征分箱
```
from skcredit.feature_discretization.SplitCat import binning_cat

sc = binning_cat(
    x=dataset["EDUCATION"].fillna("missing").astype(str),
    y=dataset["default.payment.next.month"] ,
    column="EDUCATION", target="default.payment.next.month",
)
sc.table
```
| Column    | Bucket                    |   CntNegative |   Cntpositive |       WoE |        IVS |
|:----------|:--------------------------|--------------:|--------------:|----------:|-----------:|
| EDUCATION | {'6', '1', '0', '5', '4'} |          8984 |          2069 | -0.209693 | 0.0152528  |
| EDUCATION | {'2', '3'}                |         14380 |          4567 |  0.111705 | 0.00812532 |
| EDUCATION | {'missing'}               |             0 |             0 |  0        | 0          |

### 连续特征分箱
```
from skcredit.feature_discretization.SplitNum import binning_num

sn = binning_num(
    x=dataset["LIMIT_BAL"].fillna(-999999.0),
    y=dataset["default.payment.next.month"] ,
    column="LIMIT_BAL", target="default.payment.next.month",
)
sn.table()
```
| Column    | Bucket             |   CntNegative |   Cntpositive |       WoE |       IVS |
|:----------|:-------------------|--------------:|--------------:|----------:|----------:|
| LIMIT_BAL | (-inf,40000.0]     |          2756 |          1555 |  0.686382 | 0.0798734 |
| LIMIT_BAL | (40000.0,140000.0] |          8212 |          2767 |  0.170854 | 0.0111888 |
| LIMIT_BAL | (140000.0,+inf)    |         12396 |          2314 | -0.419709 | 0.0763266 |
| LIMIT_BAL | [-999999]          |             0 |             0 |  0        | 0         |

## 多特征分箱

### 自动分箱
```
from skcredit.feature_preprocessings import FormatTabular
from skcredit.feature_discretization import Discrete

ft = FormatTabular(keep_columns=["ID"], date_columns=[], cat_columns=cat_columns, num_columns=num_columns)
ft.fit(train_x, train_y)
train_x = ft.transform(train_x)
test_x  = ft.transform(test_x )

discrete = Discrete(keep_columns=["ID"], date_columns=[])
discrete.fit(train_x, train_y)
train_x = discrete.transform(train_x)
test_x  = discrete.transform(test_x )
```
```
discrete.information_value_score.head()
```
|       |      IVS |
|:------|---------:|
| PAY_0 | 0.874626 |
| PAY_2 | 0.563306 |
| PAY_3 | 0.421203 |
| PAY_4 | 0.365194 |
| PAY_5 | 0.330149 |
```
discrete.information_value_table.head()
```
| Column   | Bucket    |   CntNegative |   Cntpositive |       WoE |       IVS |
|:---------|:----------|--------------:|--------------:|----------:|----------:|
| PAY_0    | (-inf,0]  |         14987 |          2391 | -0.578847 | 0.217663  |
| PAY_0    | (0,1]     |          1821 |           959 |  0.615374 | 0.0544047 |
| PAY_0    | (1,+inf)  |           707 |          1635 |  2.09499  | 0.602558  |
| PAY_0    | [-999999] |             0 |             0 |  0        | 0         |
| PAY_2    | (-inf,1]  |         16080 |          3108 | -0.386973 | 0.114002  |

## 特征选择
```
select = SelectBins(keep_columns=["ID"], date_columns=[])
select.fit(train_x, train_y)
train_x = select.transform(train_x)
test_x  = select.transform(test_x )
```
```
select = SelectCIFE(keep_columns=["ID"], date_columns=[], nums_feature=10)
select.fit(train_x, train_y)
train_x = select.transform(train_x)
test_x  = select.transform(test_x )
```

## 模型训练

```
lmclassifier = LMClassifier(keep_columns=["ID"], date_columns=[])
lmclassifier.fit(train_x, train_y)
print("train ks {}".format(lmclassifier.score(train_x, train_y)))
print("test  ks {}".format(lmclassifier.score(test_x,  test_y )))
```

## 评分卡生成
```
lmcreditcard = LMCreditcard(
        keep_columns=["ID"], date_columns=[], discrete=discrete, lmclassifier=lmclassifier, BASE=500,  PDO=20,  ODDS=1)
lmcreditcard.show_scorecard()
```
| Column    | Bucket             |        WoE |   Coefficients |   PartialScore |   OffsetScores |
|:----------|:-------------------|-----------:|---------------:|---------------:|---------------:|
| PAY_5     | (-inf,0]           | -0.228131  |       0.158436 |        1.0429  |        535.934 |
| PAY_5     | (0,+inf)           |  1.48685   |       0.158436 |       -6.79711 |        535.934 |
| PAY_5     | [-999999]          |  0         |       0.158436 |       -0       |        535.934 |
| LIMIT_BAL | (-inf,40000.0]     |  0.690275  |       0.442031 |       -8.804   |        535.934 |
| LIMIT_BAL | (40000.0,140000.0] |  0.178152  |       0.442031 |       -2.27221 |        535.934 |
| LIMIT_BAL | (140000.0,+inf)    | -0.421951  |       0.442031 |        5.38171 |        535.934 |
| LIMIT_BAL | [-999999]          |  0         |       0.442031 |       -0       |        535.934 |
| PAY_AMT4  | (-inf,0.0]         |  0.475069  |       0.591888 |       -8.11335 |        535.934 |
| PAY_AMT4  | (0.0,1900.0]       |  0.0656103 |       0.591888 |       -1.12051 |        535.934 |
| PAY_AMT4  | (1900.0,+inf)      | -0.332999  |       0.591888 |        5.68705 |        535.934 |
| PAY_AMT4  | [-999999]          |  0         |       0.591888 |       -0       |        535.934 |
| PAY_3     | (-inf,1]           | -0.318642  |       0.212158 |        1.95059 |        535.934 |
| PAY_3     | (1,+inf)           |  1.36873   |       0.212158 |       -8.37882 |        535.934 |
| PAY_3     | [-999999]          |  0         |       0.212158 |       -0       |        535.934 |
| PAY_0     | (-inf,0]           | -0.578847  |       0.757377 |       12.6497  |        535.934 |
| PAY_0     | (0,1]              |  0.615374  |       0.757377 |      -13.4479  |        535.934 |
| PAY_0     | (1,+inf)           |  2.09499   |       0.757377 |      -45.7824  |        535.934 |
| PAY_0     | [-999999]          |  0         |       0.757377 |       -0       |        535.934 |
| PAY_4     | (-inf,1]           | -0.265879  |       0.167904 |        1.2881  |        535.934 |
| PAY_4     | (1,+inf)           |  1.41545   |       0.167904 |       -6.85739 |        535.934 |
| PAY_4     | [-999999]          |  0         |       0.167904 |       -0       |        535.934 |
| PAY_2     | (-inf,1]           | -0.386973  |       0.120601 |        1.34659 |        535.934 |
| PAY_2     | (1,+inf)           |  1.52513   |       0.120601 |       -5.30718 |        535.934 |
| PAY_2     | [-999999]          |  0         |       0.120601 |       -0       |        535.934 |