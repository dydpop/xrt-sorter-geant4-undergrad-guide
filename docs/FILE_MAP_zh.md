# 文件地图

这份文件解释公开仓库里每个目录和关键文件的用途。第一次阅读建议先看 `README.md` 和 `docs/TEAM_GUIDE_zh.md`，再用本文件定位证据和代码。

## 顶层文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | GitHub 首页，负责快速展示项目成果、图文导览、运行入口和边界声明。 |
| `LICENSE.md` | 权利声明：保留权利，不给公开复用授权。 |
| `NOTICE.md` | 使用提醒：组员可学习运行，但不能盗用或冒名发布。 |
| `PACKAGE_MANIFEST.json` | 公开包文件清单，用于检查发布内容是否完整。 |
| `CMakeLists.txt` | CMake 构建配置，用来编译 Geant4 程序。 |
| `GNUmakefile` | 旧式 make 入口，保留给熟悉 make 的读者。 |
| `exampleB1.cc` | C++ 程序主入口，启动 Geant4、物理过程、可视化和运行宏。 |
| `*.mac` | Geant4 宏文件，用于初始化、可视化或批量运行。 |

## 讲解、运行、论文、开发入口

| 类型 | 推荐文件 | 新手是否需要打开 |
| --- | --- | --- |
| 讲解入口 | `docs/TEAM_GUIDE_zh.md`, `docs/GLOSSARY_BY_FIRST_APPEARANCE.md`, `docs/public_explainer_zh.md` | 必须读 |
| 运行入口 | `docs/RUN_LOCALLY_zh.md`, `analysis/configs/run_research.mac`, `source_models/config/undergrad_batch/*.txt` | 准备复现时读 |
| 论文入口 | `paper/main_thesis_HIT_revised_zh.md`, `paper/reproducibility.md`, `paper/references.md` | 写论文/答辩时读 |
| 开发入口 | `exampleB1.cc`, `include/`, `src/`, `analysis/classify_absorption_groups.py` | 需要改代码时读 |
| 证据入口 | `results/undergrad_validation/`, `figures/` | 必须会查 |

## C++ 仿真代码

| 路径 | 用途 |
| --- | --- |
| `include/` | C++ 头文件，声明几何、源项、运行、事件和步进逻辑。 |
| `src/` | C++ 实现文件，定义仿真几何、材料、数据输出和事件处理。 |
| `src/DetectorConstruction.cc` | 构造世界体、矿物样本和探测器。 |
| `src/PrimaryGeneratorAction.cc` | 读取源项配置并生成 gamma。 |
| `src/SteppingAction.cc` | 在粒子进入探测器时记录命中、能量、偏转角和 primary 标记。 |
| `src/EventAction.cc` | 在每个 event 结束时汇总事件级数据。 |
| `src/RunAction.cc` | 创建 CSV、写事件表、写命中表和 metadata。 |

## 配置和物理输入

| 路径 | 用途 |
| --- | --- |
| `source_models/config/experiment_config.txt` | 默认实验配置示例。 |
| `source_models/config/undergrad_batch/` | 十材料公开复现配置，输出文件名与材料目录和 Python 分类脚本匹配。 |
| `source_models/config/source_config.txt` | 源项配置示例。 |
| `source_models/materials/material_catalog.csv` | 矿物材料表，记录材料名称、化学式、密度、分组标签、配置文件和证据状态，是 Python 分类脚本读取的材料索引。 |
| `source_models/spectra/w_target_120kV_1mmAl.csv` | W 靶 120 kV X 射线能谱。 |

## Python 分析

| 路径 | 用途 |
| --- | --- |
| `analysis/classify_absorption_groups.py` | 读取材料目录启用的事件 CSV，构造虚拟样本，划分训练/测试集，运行阈值法和 Logistic Regression，并生成验证证据包。 |
| `analysis/configs/run_research.mac` | 批量仿真运行宏，核心命令为 `/run/beamOn 5000`。 |

运行 Python 分类前，需要先通过 Geant4 在 `build/` 目录下生成材料目录启用的十个 `xrt_real_source_*_events.csv` 文件。

## 结果和图表

| 路径 | 用途 |
| --- | --- |
| `results/undergrad_validation/validation_manifest.json` | 当前证据包的总说明，记录材料、样本政策、训练/测试规模、软件版本和边界。 |
| `results/undergrad_validation/event_row_summary.csv` | 十材料事件行数检查，每种材料 5000 events。 |
| `results/undergrad_validation/absorption_group_virtual_samples.csv` | 500 个虚拟样本表，每 100 个 event 聚合成一个样本。 |
| `results/undergrad_validation/train_test_split_samples.csv` | 每个虚拟样本的训练/测试归属。 |
| `results/undergrad_validation/material_feature_summary.csv` | 按材料统计透射率、能量沉积和 gamma 命中率，便于解释材料差异。 |
| `results/undergrad_validation/absorption_group_classification_summary.csv` | 分类方法汇总，包含测试样本数、正确数和 accuracy。 |
| `results/undergrad_validation/*confusion*.csv` | 混淆矩阵，显示分类错在低吸收组还是高吸收组。 |
| `results/undergrad_validation/test_predictions.csv` | 测试集逐样本预测结果。 |
| `results/absorption_group_classification_summary.csv` | 兼容入口结果表，方便只打开 `results/` 的读者快速查看。 |
| `results/directscatter_feature_comparison.csv` | 直接/散射相关特征对比。 |
| `results/elementary_demo_report_zh.md` | 本科级演示报告。 |
| `figures/` | README 和论文可直接引用的图。 |

## 文档

| 路径 | 用途 |
| --- | --- |
| `docs/TEAM_GUIDE_zh.md` | 给零基础组员看的主指南，解释数据、变量、训练/测试拆分和结果边界。 |
| `docs/GLOSSARY_BY_FIRST_APPEARANCE.md` | 按首次出现顺序排列的术语表。 |
| `docs/RUN_LOCALLY_zh.md` | 本机运行步骤和排错。 |
| `docs/public_explainer_zh.md` | 更通俗的讲解文章。 |
| `paper/main_thesis_HIT_revised_zh.md` | 本科论文式主文档。 |
| `paper/reproducibility.md` | 论文式复现说明。 |
| `paper/references.md` | 参考文献。 |

## 不在公开仓库里的内容

公开仓库只保留本科级成果，不放内部探索路线、未公开讨论、个人备份路径和不适合组员直接学习的中间材料。`build/`、`.vscode/`、`analysis/__pycache__/` 和 `CMakeUserPresets.json` 属于本机运行产物或本机配置，不作为公开提交内容。
