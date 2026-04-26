# 复现说明

这份说明对应公开仓库的本科级交付包。公开包保留了可编译的 Geant4 仿真代码、Python 分类脚本、已整理结果表和图表；没有携带大型中间文件或个人机器路径。

## 环境前提

在仓库根目录运行命令：

```bash
cd xrt-sorter-geant4-undergrad-guide
```

需要的基础环境包括 Geant4、CMake、支持 C++17 的编译器、Python 3.10 或更高版本，以及 `pandas`、`scikit-learn`。

## 最小复现链路

```bash
cmake -S . -B build
cmake --build build
export XRT_EXPERIMENT_CONFIG=source_models/config/experiment_config.txt
cd build
./xrt_sorter ../analysis/configs/run_research.mac
cd ..
python analysis/classify_absorption_groups.py
```

Geant4 程序会生成事件级 CSV，Python 脚本会读取这些 CSV 并在 `analysis/results/` 下生成分类样本表、分类汇总表和混淆矩阵。

## 公开包中可直接复查的结果

| 输出 | 路径 |
| --- | --- |
| 完成状态摘要 | `results/elementary_completion_status.json` |
| 演示报告 | `results/elementary_demo_report_zh.md` |
| 分类汇总表 | `results/absorption_group_classification_summary.csv` |
| 混淆矩阵 | `results/absorption_group_confusion_threshold.csv`, `results/absorption_group_confusion_logistic_1f.csv`, `results/absorption_group_confusion_logistic_3f.csv` |
| 特征对比表 | `results/directscatter_feature_comparison.csv` |
| 核心图表 | `figures/` |

## 已验证状态

- 本科级状态：`ELEMENTARY_PROJECT_COMPLETION_READY`
- 当前仿真数据上的粗粒度吸收组分类最高 accuracy：`0.98`
- 该结果只对应当前公开包中的仿真数据、特征构造和分类目标。

## 注意事项

不要把 `.venv`、CMake 缓存、`__pycache__` 或大型可重建中间文件当作核心交付内容。组员学习时优先阅读 `README.md`、`docs/`、`paper/`，复查结果时优先查看 `results/` 和 `figures/`。
