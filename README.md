我正在开发一个企业内部 Metadata Field Matching 机器学习 POC。

目标是根据两个字段名：

A = source field
B = target field

判断它们属于：

* DIRECT
* DERIVATION
* NO_MATCH

原始数据文件叫：

data.csv

CSV 必须只有或者至少包含以下三列：

A,B,label

例如：

A,B,label
sale_price,SalePrice,DIRECT
customer_no,client_id,DIRECT
trade_date,settlement_date,DERIVATION
sale_price,customer_id,NO_MATCH

请帮我编写一个 Python 数据准备程序，用来检查 data.csv，并生成可靠的机器学习训练集、验证集和测试集。

这个任务最重要的要求是：

**测试集必须尽量模拟未来第一次出现的新字段，不能因为数据泄漏导致测试准确率虚高。**

请严格按照下面要求实现。

---

## 1. 输入数据检查

读取：

data.csv

检查以下问题：

1. A 是否为空
2. B 是否为空
3. label 是否为空
4. label 是否只属于：

DIRECT
DERIVATION
NO_MATCH

label 大小写可以自动标准化，例如：

direct → DIRECT
Derivation → DERIVATION
no_match → NO_MATCH

删除 A/B 完全为空的数据。

输出以下统计：

* 总 row 数
* DIRECT 数量
* DERIVATION 数量
* NO_MATCH 数量
* 重复 pair 数量
* conflicting label 数量

所谓 conflicting label，例如：

price,SalePrice,DIRECT
price,SalePrice,DERIVATION

如果存在 conflicting label：

不要自动猜测哪个正确。

把它们输出到：

conflicting_pairs.csv

并在报告中明确列出数量。

---

## 2. Duplicate 处理

完全相同的：

A,B,label

只保留一条。

但是：

A,B

相同而 label 不同的属于 conflict，不允许简单删除。

---

## 3. 字段 normalization

为了检测 data leakage，请建立一个用于比较字段名称的 normalized version。

例如：

SalePrice
sale_price
sale-price
sale.price

normalization 后尽量统一，例如：

sale_price

Normalization 至少包括：

* Unicode normalization
* trim 空格
* lowercase
* CamelCase 拆分
* "-" "." whitespace 转换为 "_"
* 连续多个 "_" 合并
* 去除字符串首尾 "_"

注意：

最终输出 CSV 中仍然保留原始 A 和 B。

normalized value 只用于 split 和 leakage checking。

---

# 4. 最重要：不要使用普通随机 row split

禁止直接使用类似：

train_test_split(rows)

进行普通随机划分。

因为这种方式可能出现：

TRAIN:

price → sale_price

TEST:

sale_price → SalePrice

或者：

TRAIN:

customer_no → client_id

TEST:

customer_no → customer_number

这样测试集中的字段其实已经在训练过程中出现过，会导致测试结果虚高。

---

# 5. Primary split：FIELD-DISJOINT SPLIT

目标比例：

Train = 70%
Validation = 15%
Test = 15%

但是优先级是：

FIELD DISJOINT > 精确比例

严格要求：

如果一个 normalized field name 出现在 TEST 中，则这个 field name 不应该出现在 TRAIN 中。

无论这个字段是在 A column 还是 B column 中出现，都算出现。

例如：

如果：

sale_price

出现在 test.A 或 test.B：

那么 train.A 和 train.B 中都不应该出现：

sale_price

建议实现方式：

把每个 normalized field name 当作 graph node。

每一个：

A → B

建立一条 edge。

计算 connected components。

一个 connected component 中的所有 rows 必须全部进入：

TRAIN

或者全部进入：

VALIDATION

或者全部进入：

TEST

不能把同一个 connected component 拆到不同的数据集。

然后尽量按照：

70 / 15 / 15

分配 component。

分配时尽量保持 DIRECT / DERIVATION / NO_MATCH 的 label distribution 接近原始数据。

可以使用 deterministic greedy algorithm。

必须支持：

--seed 42

保证运行结果可重复。

---

# 6. Giant Connected Component 问题

现实数据中可能存在：

id
date
type
price

这类非常常见的 field name。

它们可能导致一个巨大的 connected component。

如果因为 giant component 导致严格 70/15/15 无法实现：

不要静默退化成 random split。

请：

1. 输出 component size distribution
2. 输出最大的 20 个 connected components
3. 在报告中说明问题
4. 尽量做最接近的 field-disjoint split
5. 明确显示最终实际比例

例如：

Train: 73.2%
Validation: 13.5%
Test: 13.3%

这是可以接受的。

Field disjoint 比严格的 70/15/15 更重要。

---

# 7. Test 数据必须是真实数据

TEST 和 VALIDATION 中：

**禁止为了提高数量而自动制造 synthetic records。**

禁止：

* 随机交换 A/B 后直接认为 NO_MATCH
* GAN 生成测试数据
* typo 生成测试数据
* 自动 rename 生成测试数据
* 随机字段组合后直接作为真实 NO_MATCH

Test set 必须来自 data.csv 中原本已经存在并经过人工确认的数据。

如果 data.csv 中没有 NO_MATCH：

不要在正式 TEST 中自动生成假的 NO_MATCH。

请在 report 中明确写：

"Formal NO_MATCH evaluation is unavailable because the source dataset contains no human-verified NO_MATCH examples."

训练集以后可以生成 synthetic negative，但是正式 test dataset 不可以。

---

# 8. Hard Test Cases

除了标准 test.csv，请额外从 TEST 中识别并输出：

test_hard_cases.csv

Hard case 可以包含：

1. A/B 字符串 similarity 很低，但是 label 是 DIRECT
2. A/B 字符串 similarity 很低，但是 label 是 DERIVATION
3. A/B 字符串 similarity 很高，但是 label 是 NO_MATCH
4. 名称只差一点，但是结果不同
5. abbreviation 情况
6. typo / separator / camelCase 等情况

可以使用 Python 标准库：

difflib.SequenceMatcher

计算一个简单 string similarity。

建议：

DIRECT 或 DERIVATION：

similarity < 0.4

可以认为是潜在 hard positive。

NO_MATCH：

similarity > 0.7

可以认为是潜在 hard negative。

这些 threshold 做成参数，不要 hard code 到逻辑中。

---

# 9. 输出文件

生成：

data_split/
train.csv
validation.csv
test.csv
test_hard_cases.csv
conflicting_pairs.csv
split_report.md
split_statistics.json

train.csv / validation.csv / test.csv 保持：

A,B,label

三列。

---

# 10. Leakage validation

程序结束前必须自动执行 leakage check。

检查：

TRAIN 和 TEST 是否存在相同 normalized field。

TRAIN 和 VALIDATION 是否存在相同 normalized field。

VALIDATION 和 TEST 是否存在相同 normalized field。

输出例如：

Train/Test shared fields: 0
Train/Validation shared fields: 0
Validation/Test shared fields: 0

理想情况必须全部为 0。

如果不是 0：

程序返回明显 WARNING 或 ERROR。

---

# 11. split_report.md

生成完整 Markdown 报告，至少包含：

# Dataset Split Report

## Original Dataset

总数量及 label distribution。

## Data Quality

duplicate 数量
conflict 数量
missing 数量

## Split Strategy

解释为什么不能普通 random row split。

解释 field-disjoint / connected-component split。

## Dataset Sizes

Train / Validation / Test 数量与百分比。

## Label Distribution

分别显示三个 dataset：

DIRECT
DERIVATION
NO_MATCH

数量和比例。

## Leakage Check

列出 shared normalized field count。

## Connected Components

component 数量
最大 component
Top 20 component sizes

## Hard Test Cases

hard positive 数量
hard negative 数量

## Limitations

特别说明：

测试集是否存在人工确认的 NO_MATCH。

---

# 12. 工程要求

请写成一个可以直接运行的 Python 文件，例如：

prepare_dataset.py

运行方式：

python prepare_dataset.py 
--input data.csv 
--output-dir data_split 
--train-ratio 0.70 
--val-ratio 0.15 
--test-ratio 0.15 
--seed 42

仅使用常见 Python package：

pandas
numpy

如果可以避免，不要引入复杂第三方 ML library。

代码要求：

* functions 清晰
* type hints
* logging
* deterministic
* exception handling
* main()
* argparse
* 注释清楚
* Windows/Linux 都能运行
* 不访问互联网

最后请给出：

1. 完整 Python 代码
2. requirements.txt 中需要增加的 dependency
3. 一个命令行运行示例
4. 输出文件说明
5. 如何判断 split 是否成功
