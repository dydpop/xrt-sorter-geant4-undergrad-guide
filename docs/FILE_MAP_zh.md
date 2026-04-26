# 文件地图

这份文件解释公开仓库里每个目录和关键文件的作用。

## 顶层文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | GitHub 首页，负责快速展示项目成果和学习入口。 |
| `LICENSE.md` | 权利声明：保留权利，不给公开复用授权。 |
| `NOTICE.md` | 使用提醒：组员可以学习运行，但不能盗用或冒名发布。 |
| `PACKAGE_MANIFEST.json` | 公开包文件清单，便于检查发布内容是否完整。 |
| `CMakeLists.txt` | CMake 构建配置，用来编译 Geant4 程序。 |
| `GNUmakefile` | 旧式 make 入口，保留给熟悉 make 的读者。 |
| `exampleB1.cc` | C++ 程序主入口，启动 Geant4、物理过程、可视化和运行宏。 |
| `*.mac` | Geant4 宏文件，用于初始化、可视化或批量运行。 |

## C++ 仿真代码

| 路径 | 用途 |
| --- | --- |
| `include/` | C++ 头文件，声明几何、源项、运行、事件和步进逻辑。 |
| `src/` | C++ 实现文件，真正定义仿真几何、材料、数据输出和事件处理。 |

建议先看：

1. `exampleB1.cc`
2. `include/DetectorConstruction.hh`
3. `src/DetectorConstruction.cc`
4. `src/RunAction.cc`
5. `src/EventAction.cc`

## 配置和物理输入

| 路径 | 用途 |
| --- | --- |
| `source_models/config/experiment_config.txt` | 默认实验配置，控制材料、源项、矿石厚度和探测器参数。 |
| `source_models/config/undergrad_batch/` | 公开复现用的六材料配置，输出文件名与 Python 分类脚本匹配。 |
| `source_models/config/source_config.txt` | 源项配置示例。 |
| `source_models/materials/material_catalog.csv` | 矿物材料表。 |
| `source_models/spectra/w_target_120kV_1mmAl.csv` | W 靶 120 kV X 射线能谱。 |

## Python 分析

| 路径 | 用途 |
| --- | --- |
| `analysis/classify_absorption_groups.py` | 读取仿真输出，构造样本，做粗粒度吸收组分类。 |
| `analysis/configs/run_research.mac` | 批量仿真运行宏。 |

运行 Python 分类前，需要先通过 Geant4 生成 `build/` 目录下的事件 CSV。

## 结果和图表

| 路径 | 用途 |
| --- | --- |
| `results/absorption_group_classification_summary.csv` | 分类方法汇总，包含当前最高 `0.98` accuracy。 |
| `results/*confusion*.csv` | 混淆矩阵，显示分类错在哪里。 |
| `results/directscatter_feature_comparison.csv` | 直接/散射相关特征对比。 |
| `results/elementary_demo_report_zh.md` | 本科级演示报告。 |
| `figures/` | README 和论文可直接引用的图。 |

## 文档

| 路径 | 用途 |
| --- | --- |
| `docs/TEAM_GUIDE_zh.md` | 给零基础组员看的主指南。 |
| `docs/GLOSSARY_BY_FIRST_APPEARANCE.md` | 按首次出现顺序排列的术语表。 |
| `docs/RUN_LOCALLY_zh.md` | 本机运行步骤和排错。 |
| `docs/public_explainer_zh.md` | 通俗讲解文章。 |
| `paper/main_thesis_HIT_revised_zh.md` | 本科论文式主文档。 |

## 不在公开仓库里的内容

公开仓库只保留本科级成果，不放内部探索路线、未公开讨论、个人备份路径和不适合组员直接学习的中间材料。
