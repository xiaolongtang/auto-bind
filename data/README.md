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

## Training-only synthetic negatives

如果且仅当 train 没有 `NO_MATCH`，程序可以从 train positive pairs 生成 random negatives 和基于名称相似度的 hard negatives。它会排除所有已知 `DIRECT`/`DERIVATION` pair。

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
