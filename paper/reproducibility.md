# 复现说明

这份说明对应公开仓库的本科级交付包。公开包保留了可编译的 Geant4 仿真代码、十材料公开配置、Python 分类脚本、紧凑验证证据、结果表和图表；没有提交 `build/`、本机绝对路径或大型临时文件。

## 环境前提

在仓库根目录运行命令：

```bash
cd xrt-sorter-geant4-undergrad-guide
```

需要 Geant4、CMake、支持 C++17 的编译器、Python 3.10 或更高版本，以及 `pandas`、`scikit-learn`。如果运行时出现 Geant4 动态库找不到的问题，需要先加载本机 Geant4 环境脚本，例如：

```bash
source /path/to/geant4-install/bin/geant4.sh
```

## 十材料复现链路

公开分类脚本会读取 `source_models/materials/material_catalog.csv` 中启用的材料。当前启用材料为 Quartz、Calcite、Orthoclase、Albite、Dolomite、Pyrite、Hematite、Magnetite、Chalcopyrite 和 Galena。

```bash
pip install pandas scikit-learn
cmake -S . -B build
cmake --build build

cd build
for material in quartz calcite orthoclase albite dolomite pyrite hematite magnetite chalcopyrite galena; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
python analysis/classify_absorption_groups.py
```

Geant4 程序会在 `build/` 中生成事件级 CSV、命中级 CSV 和 metadata。Python 脚本会读取十个事件级 CSV，并在 `results/undergrad_validation/` 下生成验证证据包。

## 可复查结果

| 输出 | 路径 |
| --- | --- |
| 验证 manifest | `results/undergrad_validation/validation_manifest.json` |
| 事件行数摘要 | `results/undergrad_validation/event_row_summary.csv` |
| 虚拟样本表 | `results/undergrad_validation/absorption_group_virtual_samples.csv` |
| 训练/测试拆分表 | `results/undergrad_validation/train_test_split_samples.csv` |
| 分组特征统计 | `results/undergrad_validation/feature_group_summary.csv` |
| 逐材料特征统计 | `results/undergrad_validation/material_feature_summary.csv` |
| 分类汇总 | `results/undergrad_validation/absorption_group_classification_summary.csv` |
| 混淆矩阵 | `results/undergrad_validation/absorption_group_confusion_threshold.csv`, `results/undergrad_validation/absorption_group_confusion_logistic_1f.csv`, `results/undergrad_validation/absorption_group_confusion_logistic_3f.csv` |
| 测试集逐样本预测 | `results/undergrad_validation/test_predictions.csv` |
| 兼容入口结果表 | `results/absorption_group_classification_summary.csv` 和 `results/*confusion*.csv` |

## 当前已验证状态

- 本科级状态：目录驱动十材料公开验证包已生成。
- 当前证据包材料数：10。
- 每种材料事件数：5000。
- 每种材料虚拟样本数：50。
- 训练集样本数：250。
- 测试集样本数：250。
- 当前证据包测试 accuracy：
  - 阈值法：`246/250 = 0.9840`
  - 单特征 Logistic Regression：`248/250 = 0.9920`
  - 三特征 Logistic Regression：`249/250 = 0.9960`
- 结果边界：仅对应当前十材料、固定几何、仿真数据和粗粒度吸收组二分类任务。

## 复现验收清单

复现后至少确认：

1. `event_row_summary.csv` 中十种材料都是 `event_count=5000`。
2. `validation_manifest.json` 中 `total_virtual_samples=500`、`total_train_samples=250`、`total_test_samples=250`。
3. `absorption_group_confusion_logistic_3f.csv` 中高吸收组测试样本全部判对，低吸收组 125 个测试样本中 124 个判对。
4. `absorption_group_classification_summary.csv` 中三特征 Logistic Regression 为 `249/250 = 0.9960`，复跑时可因随机性小幅波动。

## 注意事项

Geant4 仿真具有随机性。如果重新生成事件 CSV 且没有固定随机种子，accuracy 可能出现小幅波动。论文和讲解中应以提交的 `results/undergrad_validation/` 证据包为准，同时说明该结果不是所有矿物、真实设备或复杂现场矿流的普适指标。

不要把 `.venv`、CMake 缓存、`__pycache__`、`build/` 或本机路径作为核心交付内容。组员学习时优先阅读 `README.md`、`docs/` 和 `paper/`；复查证据时优先查看 `results/undergrad_validation/`。
