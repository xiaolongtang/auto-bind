# Data Guide

真实企业数据不应提交到 Git。把本地文件放在 `data/train.csv`、`data/validation.csv`、`data/test.csv`；这些路径已被 `.gitignore` 排除。`data/examples/` 只包含虚构的小型功能演示数据。

## Pair datasets

train、validation、test 必须是上游已经固定拆分的 UTF-8 CSV，且都有以下列：

```csv
A,B,label
sale_price,SalePrice,DIRECT
trade_date,settlement_date,DERIVATION
sale_price,employee_id,NO_MATCH
```

标签只能是 `NO_MATCH`、`DIRECT`、`DERIVATION`。训练程序不会重新 random split。推荐使用 normalized field-disjoint split，至少保证相同的规范化 A/B pair 不跨 split。示例 `train.csv`、`validation.csv`、`test.csv` 之间字段完全分离，以便 strict leakage check 通过。

CSV 默认按 UTF-8 读取并兼容 UTF-8 BOM。A、B、label 不应为空；包含逗号、引号或换行的字段需要标准 CSV quoting。

同一 split 内，相同标准化 `(A, B)` 不能出现不同标签。预检会指出冲突的 CSV 记录行号和标签；相同标签的重复行仍允许。此检查不会自动修正原始数据。

## Group evaluation completeness

字段对 `NO_MATCH` 不等于 A 与整个 B 组无匹配。validation/test 可额外包含 `group_truth_complete` 列，值为 `true/false`；缺列默认全部 `false`，同一标准化 A 的所有行必须一致。例如：

```csv
A,B,label,group_truth_complete
customer_no,client_id,DIRECT,true
customer_no,employee_id,NO_MATCH,true
legacy_code,employee_id,NO_MATCH,true
partial_code,employee_id,NO_MATCH,false
```

`true` 表示已人工确认：相对于当前 CSV **全部标准化去重后的 B**，该 A 的所有有效匹配及关系已完整列出。不必列全负例；无正例且标记 `true` 才能认定组级无匹配。新增 B 时要重新核验；不同候选组请分开评估。

标注不完整的 A 不进入组级准确率和 precision/coverage 分母，报告会列出排除数量；没有完整 A 时这些指标为 `null`。Pair classification 和已知正例的检索诊断不受此标记影响。勿把此列批量设为 `true` 来消除警告。仓库演示数据保留原始三列，不自动声称标注完整。

## Training-only synthetic negatives

如果且仅当 train 没有 `NO_MATCH`，程序可以从 train positive pairs 生成 random negatives 和基于名称相似度的 hard negatives。它会排除所有已知 `DIRECT`/`DERIVATION` pair，以及标准化后 A 与 B 相同的候选。无合法候选时不会伪造负例；整个 train 无法生成负例时会在创建 run 前失败。

生成行带有 `synthetic=true`、`negative_type=random/hard`。配置 `synthetic_negative_weight` 可降低其在两阶段训练中的损失贡献（范围 `(0,1]`，默认 `1.0`），人工数据和验证损失维持权重 `1.0`。训练历史保存生成数量、类型计数和权重。

**Synthetic negatives 只能用于训练，不等同于人工确认的 `NO_MATCH` 评估数据。** validation/test 不会自动生成负例；若它们没有人工确认的 `NO_MATCH`，相关指标必须视为 unavailable 或 incomplete。

## Leakage and vocabulary

运行：

```bash
python scripts/validate_data.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --test data/test.csv \
  --strict-leakage-check
```

strict 模式发现规范化字段重叠会失败。字符 vocabulary 只能使用 train 构建；不要为了降低 `UNK` 数量而读取 validation/test。

## Inference CSVs

- pair prediction 输入：`A,B`；示例见 `input_pairs.csv`；
- source group 输入：单列 `A`；示例见 `source_fields.csv`；
- target group 输入：单列 `B`；示例见 `target_fields.csv`。

这些演示行刻意很少，只验证 schema、训练、保存、加载和推理链路，不用于衡量准确率。
