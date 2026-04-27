# 材料级分选诊断记录

这份记录用于回答一个关键问题：当前十种材料的 XRT 仿真数据，能不能支持“直接识别十个材料/矿物种类”的说法。

结论很明确：**不能**。当前已通过的公开结论仍然是低/高吸收组二分类基线；材料级分选脚本和结果应被视为未通过验收的诊断证据，而不是成功结果。

## 1. 为什么要做这次诊断

此前公开包已经有十种材料、500 个虚拟样本、250/250 训练/测试拆分，并在低/高吸收组任务上取得 `0.9960` 测试 accuracy。但这个数字只对应二分类吸收组，不等于十材料物种识别。

从 Nature 编辑或项目导师视角看，如果标题和开头一直说“矿物分选”，读者可能误以为模型已经能分清 Quartz、Calcite、Pyrite、Galena 等十个材料标签。为了防止这个误读，我们补充了材料级诊断：让模型直接预测十个材料名，并设置 top-3、置信度、margin、复核/未知机制和开放集检查。

## 2. 新增了哪些工程能力

这次补充不是改文字，而是补了可运行的实验骨架：

| 文件 | 作用 |
| --- | --- |
| `exampleB1.cc`、`include/ExperimentConfig.hh`、`src/ExperimentConfig.cc`、`src/RunAction.cc` | 增加 `random_seed` 配置读取、随机种子设置和 metadata 输出 |
| `analysis/generate_material_sorting_matrix.py` | 生成材料级分选矩阵配置 |
| `analysis/run_material_sorting_matrix.py` | 批量或分批运行矩阵配置 |
| `analysis/material_sorting.py` | 构建材料级特征表，训练多模型，输出 top-1/top-3、复核决策和开放集检查 |
| `analysis/configs/run_material_sorting_pilot.mac` | pilot 配置使用的 Geant4 宏，单配置 2000 events |
| `analysis/configs/run_material_sorting_full.mac` | full 配置使用的 Geant4 宏，单配置 10000 events |
| `source_models/config/material_sorting_matrix/` | pilot/full 各 270 个配置：10 材料 x 3 厚度 x 3 源项 x 3 seed |
| `results/material_sorting/` | 当前材料级诊断输出 |

矩阵设计为 `10 materials x 3 thicknesses x 3 sources x 3 seeds = 270 runs`。这为后续更严谨的材料级实验留好了结构，但完整矩阵尚未全部运行。

## 3. 当前实际跑了什么

当前材料级诊断使用两类证据：

| 证据 | 状态 | 解释 |
| --- | --- | --- |
| 旧的 10 mm、120 kV spectrum、十材料公开数据 | 已用于材料级 baseline 诊断 | 因完整矩阵数据不足，`analysis/material_sorting.py` 自动回退到旧公开数据 |
| pilot 矩阵 smoke run | 已跑 3 个配置 | Quartz、5 mm、mono 60 keV、seed 101/202/303，验证 seed 配置和 metadata 输出 |
| 完整 pilot 矩阵 | 未运行 | 需要 270 个配置，每个 2000 events |
| 完整 full 矩阵 | 未运行 | 需要 270 个配置，每个 10000 events |

`results/material_sorting/material_sorting_manifest.json` 会记录 `matrix_raw_status`。当前状态显示只找到 3 个矩阵 metadata，材料集合只有 Quartz，因此材料级分析没有把这 3 个 smoke run 当作完整材料级证据，而是回退到旧十材料 baseline 数据。

## 4. 当前材料级结果

在旧公开数据上直接做十材料分类，结果如下：

| 方法 | 测试样本 | 特征数 | top-1 accuracy | top-3 accuracy | macro-F1 | 最小类别召回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 250 | 35 | 0.448 | 0.872 | 0.4429 | 0.16 |
| Random Forest | 250 | 35 | 0.420 | 0.852 | 0.3879 | 0.08 |
| SVM RBF | 250 | 35 | 0.420 | 0.868 | 0.4063 | 0.12 |
| Calibrated Extra Trees | 250 | 35 | 0.464 | 0.876 | 0.4486 | 0.08 |

主方法 `Calibrated Extra Trees` 的复核门控结果为：

| 指标 | 当前值 | 验收要求 | 是否通过 |
| --- | ---: | ---: | --- |
| closed-set top-1 accuracy | 0.464 | >= 0.85 | 否 |
| closed-set macro-F1 | 0.4486 | >= 0.80 | 否 |
| min class recall | 0.08 | >= 0.70 | 否 |
| top-3 accuracy | 0.876 | >= 0.95 | 否 |
| auto-sort precision | 0.8947 | >= 0.90 | 否 |
| review rate | 0.848 | <= 0.30 | 否 |
| open-set review recall | 0.808 | >= 0.90 | 否 |

因此 `all_criteria_met = false`。这不是代码失败，而是一个必要的科学结论：旧的单厚度、单源项、同分布数据不足以支撑十材料物种级自动分选。

## 5. 应该怎样表述

可以说：

- 当前公开包已经完成十材料仿真数据和低/高吸收组二分类验证。
- `0.9960` 是吸收组二分类结果，不是十材料识别结果。
- 新增材料级诊断显示，直接用旧数据做十材料分类时 top-1 只有 `0.464`，所有验收条件均未通过。
- 项目已经具备继续做材料级实验的配置矩阵、seed 记录、批量运行脚本和评价脚本。

不应说：

- 当前模型已经能准确识别十种矿物。
- `0.9960` 代表十材料分类准确率。
- 当前结果已经能支持工业自动分选或所有矿物覆盖。

## 6. 下一阶段怎么做

如果导师要求继续推进材料级分选，建议按以下顺序走：

1. 先运行完整 pilot 矩阵，确认 270 个配置都能产出 metadata、events 和 hits。
2. 用 `analysis/material_sorting.py` 对完整矩阵数据做 seed holdout 或 run-level holdout，而不是继续依赖旧的 within-run half split。
3. 只在 top-1、macro-F1、min recall、top-3、auto-sort precision、review rate 和 open-set review recall 达标后，才把材料级结果写成正向结论。
4. 如果仍未达标，应转向“物理/化学描述符 + 矿物字典候选检索 + 人工复核”的产品原型路线，而不是强行输出自动分选动作。
