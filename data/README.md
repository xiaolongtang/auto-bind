# Data Guide

真实企业数据不应提交到 Git。原始文件可放在仓库根目录的 `data.csv`，清洗后默认输出到 `data_split/`；这两处和旧的 `data/*.csv` 路径都已被 `.gitignore` 排除。使用自定义目录时，请相应添加忽略规则。`data/examples/` 只包含虚构的小型功能演示数据。

## Prepare raw data

取得原始 `data.csv` 后，在仓库根目录运行：

```bash
python prepare_dataset.py --input data.csv --output-dir data_split --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

数据准备程序只使用 Python 标准库，无需安装新增依赖。输入必须至少包含 `A,B,label`，使用 UTF-8（兼容 BOM）；A/B 原文会保留，label 去除首尾空白并转成大写。缺失或非法值、规范化后为空的字段会写入 `rejected_rows.csv` 并注明原因。完全相同的 `A,B,label` 仅保留一条；原始或规范化 `(A,B)` 的不同标签记录全部写入 `conflicting_pairs.csv`，不参与划分，也不自动修改标签。

可选 `synthetic` 列会先去除首尾空白、转小写：`true/1/yes` 表示合成记录，会隔离；`false/0/no` 可继续清洗；空值及其他值按无效标记隔离。缺少此列仍需数据提供方确认人工来源。处理虚构示例可加 `--example-data`，让报告明确标记数据不用于正式评估。

程序按字段连通分量划分：规范化字段在 A/B 两列中共享节点，每条记录（包括 `NO_MATCH`）连接其 A 与 B，整个分量只能属于一个集合。greedy 分配尽量保持 70/15/15 和标签分布，同一输入、参数及 seed 可重复得到相同划分。字段隔离优先于精确比例，不会退回随机行划分；高频字段或负例边产生的巨型分量会在报告中显式列出。

| 输出文件 | 内容 |
| --- | --- |
| `train.csv` | 训练集，仅 `A,B,label` 三列 |
| `validation.csv` | 验证集，仅 `A,B,label` 三列 |
| `test.csv` | 测试集，仅 `A,B,label` 三列 |
| `test_hard_cases.csv` | 从 test 识别的困难样本，列为 `A,B,label,source_row,similarity,reasons` |
| `conflicting_pairs.csv` | 标签冲突记录，列为 `A,B,label,source_row,normalized_A,normalized_B,conflict_type` |
| `rejected_rows.csv` | 其他不可用记录，列为 `A,B,label,source_row,reason` |
| `split_report.md` | 原始数据及质量统计、实际划分比例、各类数量、两两泄漏检查、分量规模分布和 Top 20、困难样本及限制 |
| `split_statistics.json` | 机器可读统计 |

困难样本根据规范化 A/B 的 `difflib.SequenceMatcher` 相似度识别：正例低于 `--hard-positive-threshold`（默认 `0.4`）或 `NO_MATCH` 高于 `--hard-negative-threshold`（默认 `0.7`）。此外标记规范化后同名的格式变体、相似度高于后者阈值的正例拼写变体和可能的缩写，供人工复核，不能据此确认 typo 或语义关系。`reasons` 用分号分隔多个原因；`source_row` 为原 CSV 记录号，表头计为 1。`test_hard_cases.csv` 是 test 的子集，不能混入训练或作为独立样本与 test 合并统计。

成功条件：清洗命令退出码为 `0`，目标比例大于零的集合均非空，train/test、train/validation、validation/test 的 shared normalized fields 都为 `0`。接入现有训练还需 `ready_for_training=true`，即三个集合均非空。比例必须在 `[0,1]` 内且总和为 `1`；清洗允许零比例用于其他实验，完整训练建议保持三个比例均大于零。仍应检查报告中的实际比例、类别缺失和隔离记录警告。若无有效记录，或分量不能填满所有目标集合，程序输出诊断并以 `2` 退出；请先修复或补充原始数据，不能通过拆开连通分量规避隔离要求。

清洗不生成任何 synthetic records，程序也无法自动核实人工标注来源。输入标签必须已由数据提供方人工确认；没有 `NO_MATCH` 时报告会说明正式 `NO_MATCH` 评估不可用，validation/test 缺该类时评估也不完整。输入中的额外列不会进入输出的三个主 CSV；如需组级评估完整性声明，必须在最终候选范围上重新核验，参见下文。

报告检查通过后可直接运行现有程序：

```bash
python scripts/validate_data.py --data-dir data_split --strict-leakage-check
python scripts/train_model.py --config config.example.json --data-dir data_split --artifacts-dir artifacts
python scripts/evaluate_model.py --model-dir artifacts/run_YYYYMMDD_HHMMSS --data-dir data_split --split test --output artifacts/run_YYYYMMDD_HHMMSS/evaluation.json
```

将 run 目录替换为实际训练日志中给出的目录。`--data-dir` 默认是 `data_split`；评估默认 `--split test`，也支持 `--split validation`。训练和校验的显式 `--train`、`--validation`、`--test` 覆盖对应默认文件路径；评估的 `--test` 覆盖所选文件，仍可使用现有自备拆分文件和 `data/examples/` 演示文件。

## Pair datasets

train、validation、test 是清洗程序生成或上游固定拆分的 UTF-8 CSV，且都有以下列：

```csv
A,B,label
sale_price,SalePrice,DIRECT
trade_date,settlement_date,DERIVATION
sale_price,employee_id,NO_MATCH
```

标签只能是 `NO_MATCH`、`DIRECT`、`DERIVATION`。训练程序不会重新 random split。应使用 normalized field-disjoint split，任意两个集合不能出现相同的规范化字段，无论字段在 A 还是 B 列。示例 `train.csv`、`validation.csv`、`test.csv` 之间字段完全分离，以便 strict leakage check 通过。

CSV 默认按 UTF-8 读取并兼容 UTF-8 BOM。A、B、label 不应为空；包含逗号、引号或换行的字段需要标准 CSV quoting。

同一 split 内，相同标准化 `(A, B)` 不能出现不同标签。训练预检会指出冲突的 CSV 记录行号和标签；相同标签的重复行仍允许。预检不会自动修正原始数据；数据准备程序会在划分前隔离冲突并对完全相同记录去重。

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
  --data-dir data_split \
  --strict-leakage-check
```

strict 模式发现规范化字段重叠会失败。字符 vocabulary 只能使用 train 构建；不要为了降低 `UNK` 数量而读取 validation/test。

## Inference CSVs

- pair prediction 输入：`A,B`；示例见 `input_pairs.csv`；
- source group 输入：单列 `A`；示例见 `source_fields.csv`；
- target group 输入：单列 `B`；示例见 `target_fields.csv`。

这些演示行刻意很少，只验证 schema、训练、保存、加载和推理链路，不用于衡量准确率。
