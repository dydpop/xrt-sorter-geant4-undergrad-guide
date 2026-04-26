# 复现说明

这份说明对应公开仓库的本科级交付包。公开包保留了可编译的 Geant4 仿真代码、六材料公开配置、Python 分类脚本、紧凑验证证据、结果表和图表；没有提交 `build/`、本机绝对路径或大型临时文件。

## 环境前提

在仓库根目录运行命令：

```bash
cd xrt-sorter-geant4-undergrad-guide
```

需要 Geant4、CMake、支持 C++17 的编译器、Python 3.10 或更高版本，以及 `pandas`、`scikit-learn`。如果运行时出现 Geant4 动态库找不到的问题，需要先加载本机 Geant4 环境脚本，例如：

```bash
source /path/to/geant4-install/bin/geant4.sh
```

## 六材料复现链路

公开分类脚本期望 `build/` 下存在六个事件文件：`xrt_real_source_quartz_events.csv`、`xrt_real_source_orthoclase_events.csv`、`xrt_real_source_calcite_events.csv`、`xrt_real_source_pyrite_events.csv`、`xrt_real_source_hematite_events.csv` 和 `xrt_real_source_magnetite_events.csv`。因此，应运行六材料配置，而不是只运行默认单个配置。

```bash
pip install pandas scikit-learn
cmake -S . -B build
cmake --build build

cd build
for material in quartz orthoclase calcite pyrite hematite magnetite; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
python analysis/classify_absorption_groups.py
```

Geant4 程序会在 `build/` 中生成事件级 CSV、命中级 CSV 和 metadata。Python 脚本会读取六个事件级 CSV，并在 `results/undergrad_validation/` 下生成验证证据包。

## 可复查结果

| 输出 | 路径 |
| --- | --- |
| 验证 manifest | `results/undergrad_validation/validation_manifest.json` |
| 事件行数摘要 | `results/undergrad_validation/event_row_summary.csv` |
| 虚拟样本表 | `results/undergrad_validation/absorption_group_virtual_samples.csv` |
| 训练/测试拆分表 | `results/undergrad_validation/train_test_split_samples.csv` |
| 特征统计 | `results/undergrad_validation/feature_group_summary.csv` |
| 分类汇总 | `results/undergrad_validation/absorption_group_classification_summary.csv` |
| 混淆矩阵 | `results/undergrad_validation/absorption_group_confusion_threshold.csv`, `results/undergrad_validation/absorption_group_confusion_logistic_1f.csv`, `results/undergrad_validation/absorption_group_confusion_logistic_3f.csv` |
| 测试集逐样本预测 | `results/undergrad_validation/test_predictions.csv` |
| 兼容入口结果表 | `results/absorption_group_classification_summary.csv` 与 `results/*confusion*.csv` |

## 当前已验证状态

- 本科级状态：`UNDERGRADUATE_PROJECT_READY`
- 当前证据包材料数：6
- 每种材料事件数：5000
- 每种材料虚拟样本数：50
- 训练集样本数：150
- 测试集样本数：150
- 当前证据包最高测试 accuracy：`0.9933`
- 结果边界：仅对应当前六材料、固定几何、仿真数据和粗粒度吸收组二分类任务。

## 注意事项

Geant4 仿真具有随机性。如果重新生成事件 CSV 且没有固定随机种子，accuracy 可能出现小幅波动。论文和讲解中应以提交的 `results/undergrad_validation/` 证据包为准，同时说明该结果不是所有矿物、真实设备或复杂现场矿流的普适指标。

不要把 `.venv`、CMake 缓存、`__pycache__`、`build/` 或本机路径作为核心交付内容。组员学习时优先阅读 `README.md`、`docs/` 和 `paper/`；复查证据时优先查看 `results/undergrad_validation/`。
