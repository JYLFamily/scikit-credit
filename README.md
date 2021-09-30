# scikit-credit
scikit-credit 是用于信贷风控评分卡建模的自动化工具。如下是 scikit-credit 的简介：
* 自动分箱
* 特征选择
* 模型训练 
* 评分生成

# 特性
* 自动分箱：分箱算法使用以信息量（IV）为分裂标准的分类与回归树（CART）。分类与回归树（CART）分裂前自动探测特征与特征WOE之间的单调关系，分裂过程中借鉴了XGBoost限制单调性的实现，使得分箱结果特征与特征WOE单调。
* 特征选择：通过分箱结果的检查与信息论为标准的特征选择算法进行特征选择。
* 模型训练：使用以KS为标准的逐步回归，保证入模特征回归系数为正且统计显著。

# 玩家
信贷风控模型分析师

# 安装

```
pip install skcredit
```

# 示例

## 完整代码请参见 [example.ipynb](https://github.com/JYLFamily/scikit-credit/blob/master/examples/example.ipynb)

## 单特征分箱

### 分类单特征分箱

#### 默认参数
```
from skcredit.feature_discretization import SplitCat

# min_bin_cnt_negative=75
# min_bin_cnt_positive=75
# min_information_value_split_gain=0.015

sc = SplitCat()
sc.fit(dataset["EDUCATION"], dataset["default.payment.next.month"])
print(sc.table.to_markdown())
```
|    | Column    | Bucket       |   CntPositive |   CntNegative |       WoE |        IvS |
|---:|:----------|:-------------|--------------:|--------------:|----------:|-----------:|
|  0 | EDUCATION | {0, 1, 4, 5} |          1529 |          6756 | -0.229193 | 0.0181077  |
|  1 | EDUCATION | {2, 3, 6}    |          3456 |         10759 |  0.120993 | 0.00955926 |
|  2 | EDUCATION | {nan}        |             0 |             0 |  0        | 0          |


#### 调整参数

```
from skcredit.feature_discretization import SplitNum

sc = SplitCat(min_information_value_split_gain=0.0001)
sc.fit(dataset["EDUCATION"], dataset["default.payment.next.month"])
print(sc.table.to_markdown())
```
|    | Column    | Bucket          |   CntPositive |   CntNegative |        WoE |        IvS |
|---:|:----------|:----------------|--------------:|--------------:|-----------:|-----------:|
|  0 | EDUCATION | {0, 1, 4, 5, 6} |          2069 |          8984 | -0.209693  | 0.0152528  |
|  1 | EDUCATION | {2}             |          3330 |         10700 |  0.0914156 | 0.00400755 |
|  2 | EDUCATION | {3}             |          1237 |          3680 |  0.168463  | 0.00486862 |
|  3 | EDUCATION | {nan}           |             0 |             0 |  0         | 0          |

### 连续单特征分箱

#### 默认参数 

```
from skcredit.feature_discretization import SplitNum

# min_bin_cnt_negative=75
# min_bin_cnt_positive=75
# min_information_value_split_gain=0.015

sn = SplitNum()
sn.fit(dataset["LIMIT_BAL"], dataset["default.payment.next.month"])
print(sn.table.to_markdown())
```
|    | Column    | Bucket             |   CntPositive |   CntNegative |       WoE |       IvS |
|---:|:----------|:-------------------|--------------:|--------------:|----------:|----------:|
|  0 | LIMIT_BAL | (-inf,40000.0]     |          1555 |          2756 |  0.686382 | 0.0798734 |
|  1 | LIMIT_BAL | (40000.0,140000.0] |          2767 |          8212 |  0.170854 | 0.0111888 |
|  2 | LIMIT_BAL | (140000.0,+inf)    |          2314 |         12396 | -0.419709 | 0.0763266 |
|  3 | LIMIT_BAL | [nan,nan]          |             0 |             0 |  0        | 0         |

#### 调整参数

```
from skcredit.feature_discretization import SplitNum

sn = SplitNum(min_information_value_split_gain=0.0001)
sn.fit(dataset["LIMIT_BAL"], dataset["default.payment.next.month"])
print(sn.table.to_markdown())
```
|    | Column    | Bucket              |   CntPositive |   CntNegative |          WoE |         IvS |
|---:|:----------|:--------------------|--------------:|--------------:|-------------:|------------:|
|  0 | LIMIT_BAL | (-inf,10000.0]      |           197 |           296 |  0.851531    | 0.0144909   |
|  1 | LIMIT_BAL | (10000.0,40000.0]   |          1358 |          2460 |  0.664539    | 0.0660227   |
|  2 | LIMIT_BAL | (40000.0,70000.0]   |          1328 |          3593 |  0.263374    | 0.0122039   |
|  3 | LIMIT_BAL | (70000.0,120000.0]  |          1112 |          3468 |  0.121269    | 0.00232077  |
|  4 | LIMIT_BAL | (120000.0,140000.0] |           327 |          1151 |  0.000260766 | 3.35032e-09 |
|  5 | LIMIT_BAL | (140000.0,220000.0] |          1103 |          5184 | -0.288856    | 0.0160792   |
|  6 | LIMIT_BAL | (220000.0,240000.0] |           223 |          1133 | -0.366765    | 0.00546071  |
|  7 | LIMIT_BAL | (240000.0,260000.0] |           138 |           733 | -0.411205    | 0.00434948  |
|  8 | LIMIT_BAL | (260000.0,360000.0] |           556 |          3164 | -0.480137    | 0.0247926   |
|  9 | LIMIT_BAL | (360000.0,460000.0] |           165 |          1160 | -0.691543    | 0.0171397   |
| 10 | LIMIT_BAL | (460000.0,+inf)     |           129 |          1022 | -0.811017    | 0.0197102   |
| 11 | LIMIT_BAL | [nan,nan]           |             0 |             0 |  0           | 0           |

## 多特征分箱

### 手动分箱

```
from skcredit.feature_discretization import DiscreteCust

sc = SplitCat(min_information_value_split_gain=0.0001)
sc.fit(train_x["EDUCATION"], train_y)
sn = SplitNum(min_information_value_split_gain=0.0001)
sn.fit(train_x["LIMIT_BAL"], train_y)

cust = DiscreteCust(keep_columns=["ID"], date_columns=[], cat_spliter={"EDUCATION": sc}, num_spliter={"LIMIT_BAL":sn})
cust.fit(train_x, train_y)

cust = DiscreteCust(keep_columns=["ID"], date_columns=[], cat_spliter={"EDUCATION": sc}, num_spliter={"LIMIT_BAL":sn})
cust.fit(train_x, train_y)
```


### 自动分箱

```
auto = DiscreteAuto(keep_columns=["ID"], date_columns=[], cat_columns=cat_columns, num_columns=num_columns)
auto.fit(train_x, train_y)
train_x = auto.transform(train_x)
test_x  = auto.transform(test_x )
```

### 查看分箱后信息
```
# cust.information_value_score.head() 
print(auto.information_value_score.head().to_markdown())
```
|       |      IvS |
|:------|---------:|
| PAY_0 | 0.874626 |
| PAY_2 | 0.560568 |
| PAY_3 | 0.420644 |
| PAY_4 | 0.364907 |
| PAY_5 | 0.330149 |

```
# cust.information_value_table.head()
print(auto.information_value_table.head().to_markdown())
```
|    | Column   | Bucket    |   CntPositive |   CntNegative |       WoE |       IvS |
|---:|:---------|:----------|--------------:|--------------:|----------:|----------:|
|  0 | PAY_0    | (-inf,0]  |          2391 |         14987 | -0.578847 | 0.217663  |
|  1 | PAY_0    | (0,1]     |           959 |          1821 |  0.615374 | 0.0544047 |
|  2 | PAY_0    | (1,+inf)  |          1635 |           707 |  2.09499  | 0.602558  |
|  3 | PAY_0    | [nan,nan] |             0 |             0 |  0        | 0         |
|  0 | PAY_2    | (-inf,0]  |          3105 |         16066 | -0.387067 | 0.113953  |

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
train ks 0.41442 \
test  ks 0.3914

## 评分卡生成
```
lmcreditcard = LMCreditcard(
        keep_columns=["ID"], date_columns=[], discrete=discrete, lmclassifier=lmclassifier, BASE=500,  PDO=20,  ODDS=1)
lmcreditcard.show_scorecard()
```
|    | Column   | Bucket           |        WoE |   Coefficients |   PartialScore |   OffsetScores |
|---:|:---------|:-----------------|-----------:|---------------:|---------------:|---------------:|
|  0 | PAY_AMT2 | (-inf,30.0]      |  0.555291  |      0.284123  |      -4.55231  |        536.095 |
|  1 | PAY_AMT2 | (30.0,4931.0]    |  0.0159422 |      0.284123  |      -0.130696 |        536.095 |
|  2 | PAY_AMT2 | (4931.0,15282.0] | -0.424907  |      0.284123  |       3.48342  |        536.095 |
|  3 | PAY_AMT2 | (15282.0,+inf)   | -1.21984   |      0.284123  |      10.0003   |        536.095 |
|  4 | PAY_AMT2 | [nan,nan]        |  0         |      0.284123  |      -0        |        536.095 |
|  0 | PAY_AMT1 | (-inf,10.0]      |  0.688936  |      0.283557  |      -5.63668  |        536.095 |
|  1 | PAY_AMT1 | (10.0,4861.0]    | -0.0125914 |      0.283557  |       0.10302  |        536.095 |
|  2 | PAY_AMT1 | (4861.0,+inf)    | -0.591584  |      0.283557  |       4.84017  |        536.095 |
|  3 | PAY_AMT1 | [nan,nan]        |  0         |      0.283557  |      -0        |        536.095 |
|  0 | PAY_0    | (-inf,0]         | -0.578847  |      0.749677  |      12.5211   |        536.095 |
|  1 | PAY_0    | (0,1]            |  0.615374  |      0.749677  |     -13.3112   |        536.095 |
|  2 | PAY_0    | (1,+inf)         |  2.09499   |      0.749677  |     -45.317    |        536.095 |
|  3 | PAY_0    | [nan,nan]        |  0         |      0.749677  |      -0        |        536.095 |
|  0 | PAY_3    | (-inf,0]         | -0.318517  |      0.204782  |       1.88204  |        536.095 |
|  1 | PAY_3    | (0,+inf)         |  1.36739   |      0.204782  |      -8.07961  |        536.095 |
|  2 | PAY_3    | [nan,nan]        |  0         |      0.204782  |      -0        |        536.095 |
|  0 | PAY_2    | (-inf,0]         | -0.387067  |      0.0994362 |       1.11054  |        536.095 |
|  1 | PAY_2    | (0,+inf)         |  1.51702   |      0.0994362 |      -4.35252  |        536.095 |
|  2 | PAY_2    | [nan,nan]        |  0         |      0.0994362 |      -0        |        536.095 |
|  0 | PAY_4    | (-inf,0]         | -0.265818  |      0.331433  |       2.54205  |        536.095 |
|  1 | PAY_4    | (0,+inf)         |  1.41463   |      0.331433  |     -13.5283   |        536.095 |
|  2 | PAY_4    | [nan,nan]        |  0         |      0.331433  |      -0        |        536.095 |
|  0 | PAY_AMT4 | (-inf,0.0]       |  0.475069  |      0.407116  |      -5.58058  |        536.095 |
|  1 | PAY_AMT4 | (0.0,1813.0]     |  0.0656848 |      0.407116  |      -0.771592 |        536.095 |
|  2 | PAY_AMT4 | (1813.0,4000.0]  | -0.128344  |      0.407116  |       1.50764  |        536.095 |
|  3 | PAY_AMT4 | (4000.0,+inf)    | -0.51447   |      0.407116  |       6.04342  |        536.095 |
|  4 | PAY_AMT4 | [nan,nan]        |  0         |      0.407116  |      -0        |        536.095 |