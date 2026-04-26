# 基于 Geant4 的 X 射线透射矿物分选仿真系统设计与实现

## 摘要

本项目设计并实现了一个基于 Geant4 的 X 射线透射矿物分选仿真原型。系统包括 X 射线源、矿石样本、探测器响应、事件数据输出和 Python 数据分析流程。项目通过材料配置、源项配置和能谱输入构建仿真场景，并基于探测器输出提取透射率、能量沉积、直接命中和散射命中等特征。

在当前仿真数据和粗粒度吸收组任务中，基础分类方法最高 accuracy 为 `0.98`。该结果说明仿真系统能够产生有区分度的物理相关特征，并能支撑本科项目层面的系统实现和基础算法验证。本文结论仅适用于公开仓库中的仿真设置和分类任务，不代表现场设备效果或所有矿物条件下的通用表现。

## 1. 研究背景

矿物分选需要利用不同材料在物理性质上的差异。X 射线透射方法关注射线穿过材料后的信号变化，适合用来研究不同吸收特性材料的区分问题。直接搭建完整实验环境成本较高，因此本项目先使用 Geant4 完成仿真系统，为本科项目展示和后续扩展打下基础。

## 2. 系统目标

本项目目标包括：

- 搭建 X 射线源、矿石样本和探测器仿真场景。
- 使用配置文件描述材料、源项和几何参数。
- 输出可供 Python 分析的事件数据。
- 提取可解释的探测器特征。
- 使用基础分类方法完成粗粒度吸收组验证。
- 整理图表、结果表和组内学习文档。

## 3. 系统设计

系统由三部分组成。

第一部分是 Geant4 C++ 仿真。`exampleB1.cc` 是程序入口，`include/` 和 `src/` 负责几何、材料、源项、事件和运行逻辑。

第二部分是配置输入。`source_models/` 保存材料表、实验配置和 X 射线能谱。

第三部分是 Python 分析。`analysis/classify_absorption_groups.py` 读取仿真输出，构造样本并运行基础分类方法。

```mermaid
flowchart LR
    A["材料和源项配置"] --> B["Geant4 仿真"]
    B --> C["事件 CSV"]
    C --> D["Python 特征"]
    D --> E["基础分类"]
    E --> F["结果表和图"]
```

## 4. 数据和特征

仿真输出的事件数据包含探测器能量沉积、探测器命中、主射线透射等信息。Python 分析阶段将这些原始字段整理为样本级特征。

主要特征包括：

- `primary_transmission_rate`：主射线透射率。
- `mean_detector_edep_keV`：平均探测器能量沉积。
- `detector_gamma_rate`：探测器 gamma 命中率。

这些特征有明确物理含义，适合本科阶段解释和展示。

## 5. 分类方法

项目使用了两类基础方法。

第一类是阈值法。它根据透射率设置一个分界值，判断样本属于低吸收组还是高吸收组。

第二类是 Logistic Regression。它可以使用单个特征，也可以使用多个特征组合，适合与阈值法对比。

## 6. 实验结果

结果汇总文件为：

```text
results/absorption_group_classification_summary.csv
```

当前结果显示，在公开仓库的仿真数据和粗粒度吸收组任务中，最佳 accuracy 为 `0.98`。

核心图表包括：

- `figures/elementary_system_flow.png`
- `figures/elementary_xray_spectrum.png`
- `figures/elementary_direct_scatter_ratio.png`
- `figures/elementary_absorption_accuracy.png`

## 7. 复现方式

在安装 Geant4、CMake、C++ 编译器和 Python 依赖后，可以运行：

```bash
cmake -S . -B build
cmake --build build
export XRT_EXPERIMENT_CONFIG=source_models/config/experiment_config.txt
cd build
./xrt_sorter ../analysis/configs/run_research.mac
cd ..
python analysis/classify_absorption_groups.py
```

如果暂时没有 Geant4 环境，可以直接阅读 `figures/`、`results/` 和 `docs/` 中的整理成果。

## 8. 项目边界

本项目公开版本只覆盖本科级仿真系统和基础分类验证。当前结果不能被解释为所有矿物、所有设备条件或所有现场流程下都成立。

## 9. 总结

本项目完成了一条从 Geant4 物理仿真到 Python 数据分析再到本科论文材料的完整链路。它的价值不只在于一个分类数字，而在于把仿真、数据、特征、模型、结果和文档组织成了可学习、可运行、可展示的工程化项目。
