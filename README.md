# scikit-credit
scikit-credit 是用于信贷风控评分卡建模的自动化工具。如下是 scikit-credit 的简介：
* 自动分箱 
* 特征筛选
* 模型训练 
* 评分生成

# 玩家
信贷风控模型分析师

# 示例

完整代码请参见 examples/example.ipynb

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

|Column|Bucket|CntNegative|Cntpositive|WoE|IVS|
|-----|-----|-----|-----|-----|-----|
|EDUCATION|{6, 1, 0, 5, 4}|8984|2069|-0.209693|0.015253|
|EDUCATION|{2, 3}|14380|4567|0.111705|0.008125|
|EDUCATION|{missing}|0|0|0.000000|0.000000|

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

|Column|Bucket|CntNegative|Cntpositive|WoE|IVS|
|-----|-----|-----|-----|-----|-----|
|LIMIT_BAL|(-inf,40000.0]|2756|1555|0.686382|0.079873|
|LIMIT_BAL|(40000.0,140000.0]|8212|2767|0.170854|0.011189|
|LIMIT_BAL|(140000.0,+inf)|12396|2314|-0.419709|0.076327|
|LIMIT_BAL|[-999999]|0|0|0.000000|0.000000|

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
| |IVS|
|-----|-----|
|PAY_0|0.874626|
|PAY_2|0.563306|
|PAY_3|0.421203|
|PAY_4|0.365194|
|PAY_5|0.330149|
```
discrete.information_value_table.head()
```
|Column|Bucket|CntNegative|Cntpositive|WoE|IVS|
|-----|-----|-----|-----|-----|-----|
|PAY_0|(-inf,0]|14987|2391|-0.578847|0.217663|
|PAY_0|(0,1]|1821|959|0.615374|0.054405|
|PAY_0|(1,+inf)|707|1635|2.094992|0.602558|
|PAY_0|[-999999]|0|0|0.000000|0.000000|
|PAY_2|(-inf,1]|16080|3108|-0.386973|0.114002|

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
|Column|Bucket|CntNegative|Cntpositive|WoE|IVS|
|-----|-----|-----|-----|-----|-----|
|PAY_5|(-inf,0]              | -0.228131|0.158436| 1.042898 | 535.93372|
|PAY_5|(0,+inf)	             |  1.486849|0.158436|-6.797114 | 535.93372|
|PAY_5|[-999999]             |	0.000000|0.158436|-0.000000 | 535.93372|
|LIMIT_BAL|(-inf,40000.0]    |	0.690275|0.442031|-8.803997 | 535.93372|
|LIMIT_BAL|(40000.0,140000.0]|	0.178152|0.442031|-2.272213 | 535.93372|
|LIMIT_BAL|(140000.0,+inf)   | -0.421951|0.442031| 5.381706 | 535.93372|
|LIMIT_BAL|[-999999]         |	0.000000|0.442031|-0.000000 | 535.93372|
|PAY_AMT4|(-inf,0.0]         |	0.475069|0.591888|-8.113349 |	535.93372|
|PAY_AMT4|(0.0,1900.0]       |	0.065610|0.591888|-1.120511 |	535.93372|
|PAY_AMT4|(1900.0,+inf)      | -0.332999|0.591888| 5.687054 |	535.93372|
|PAY_AMT4|[-999999]          |	0.000000|0.591888|-0.000000 |	535.93372|
|PAY_3|(-inf,1]              | -0.318642|0.212158| 1.950595 |	535.93372|
|PAY_3|(1,+inf)              |	1.368731|0.212158|-8.378816 |	535.93372|
|PAY_3|[-999999]             |	0.000000|0.212158|-0.000000 |	535.93372|
|PAY_0|(-inf,0]              | -0.578847|0.757377| 12.649712|	535.93372|
|PAY_0|(0,1]                 |	0.615374|0.757377|-13.447947|	535.93372|
|PAY_0|(1,+inf)              |	2.094992|0.757377|-45.782445|	535.93372|
|PAY_0|[-999999]             |	0.000000|0.757377|-0.000000 |	535.93372|
|PAY_4|(-inf,1]              | -0.265879|0.167904| 1.288098 |	535.93372|
|PAY_4|(1,+inf)              |	1.415448|0.167904|-6.857387 |	535.93372|
|PAY_4|[-999999]             |	0.000000|0.167904|-0.000000 |	535.93372|
|PAY_2|(-inf,1]              | -0.386973|0.120601| 1.346593 |	535.93372|
|PAY_2|(1,+inf)              |	1.525134|0.120601|-5.307185 |	535.93372|
|PAY_2|[-999999]             |	0.000000|0.120601|-0.000000 |	535.93372|