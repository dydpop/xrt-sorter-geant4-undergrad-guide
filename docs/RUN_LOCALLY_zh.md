# 本地运行与复现指南

这份文档说明如何在本机重新跑出公开证据包。它的目标不是让你一次性理解所有 C++ 细节，而是让你知道环境要准备什么、命令按什么顺序执行、每一步会产生什么文件、最后如何判断结果是否合理。

## 1. 运行前需要什么

你需要准备以下环境：

- Geant4，且 CMake 能找到 `Geant4Config.cmake`。
- CMake。
- 支持 C++17 的编译器。
- Python 3。
- Python 包：`pandas`、`scikit-learn`。

如果你在 WSL 或 Linux 环境中运行，通常还需要先加载 Geant4 环境脚本，例如：

```bash
source /path/to/geant4-install/bin/geant4.sh
```

如果 CMake 找不到 Geant4，可以在本机创建不提交 Git 的 `CMakeUserPresets.json`，或设置 `Geant4_DIR` / `CMAKE_PREFIX_PATH`。这些本机绝对路径不应提交到 GitHub。

## 2. 构建程序

从仓库根目录运行：

```bash
cmake -S . -B build
cmake --build build
```

构建成功后，`build/` 目录下应出现 `xrt_sorter` 可执行程序。`build/` 是本机生成目录，不作为 Git 跟踪内容。

## 3. 运行十种材料仿真

公开证据包使用 `source_models/config/undergrad_batch/` 下的十个材料配置。每个配置文件指定材料、厚度、源项、探测器位置和输出前缀。仿真事件数由 `analysis/configs/run_research.mac` 中的 `/run/beamOn 5000` 控制。

```bash
cd build

for material in quartz calcite orthoclase albite dolomite pyrite hematite magnetite chalcopyrite galena; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
```

每个材料运行后会在 `build/` 下生成事件级 CSV、命中级 CSV 和 metadata。事件级 CSV 形如 `xrt_real_source_<material>_events.csv`，是后续分类验证的主要输入。

## 4. 安装 Python 依赖

如果尚未安装依赖，运行：

```bash
pip install pandas scikit-learn
```

本项目的 Python 分析脚本使用 pandas 读取和聚合 CSV，使用 scikit-learn 中的 `StandardScaler`、`LogisticRegression`、`accuracy_score` 和 `confusion_matrix` 完成基础分类验证。

## 5. 生成验证证据包

在仓库根目录运行：

```bash
python analysis/classify_absorption_groups.py
```

脚本会读取 `source_models/materials/material_catalog.csv` 中启用的材料，再到 `build/` 查找每种材料对应的事件文件。随后脚本会完成以下步骤：

1. 检查材料目录、配置文件和事件文件是否存在。
2. 读取事件级 CSV，并检查事件行数、唯一事件数、重复事件数和事件编号连续性。
3. 每 100 个 event 聚合为 1 个虚拟样本。
4. 构造 `primary_transmission_rate`、`mean_detector_edep_keV` 和 `detector_gamma_rate` 等样本级特征。
5. 按每种材料前 25 个样本训练、后 25 个样本测试进行拆分。
6. 运行阈值法、单特征 Logistic Regression 和三特征 Logistic Regression。
7. 输出摘要表、混淆矩阵、预测表和 manifest。

输出目录为：

```text
results/undergrad_validation/
```

## 6. 运行后怎样检查

至少检查下面四个文件：

| 文件 | 应该看到什么 |
| --- | --- |
| `event_row_summary.csv` | 每种材料 `event_count=5000`，`duplicate_event_count=0`，`ignored_tail_events=0` |
| `train_test_split_samples.csv` | 总训练样本 250，总测试样本 250 |
| `absorption_group_classification_summary.csv` | 三种方法测试结果为 246/250、248/250、249/250 |
| `validation_manifest.json` | 材料列表、样本政策、软件版本和结论边界 |

当前公开证据包的核心结果是：

```text
threshold baseline: 246/250 = 0.9840
single-feature Logistic Regression: 248/250 = 0.9920
three-feature Logistic Regression: 249/250 = 0.9960
```

重新运行 Geant4 时，如果没有固定随机种子，个别数字可能小幅波动。只要数据规模、拆分规则和结果级别一致，就说明链路复现成功；如果结果差异很大，应先检查 Geant4 环境、配置文件和事件 CSV 是否完整。

## 7. 常见问题

### CMake 找不到 Geant4

先确认本机已经安装 Geant4，并能找到 `Geant4Config.cmake`。可以用本机环境变量或本机专用的 `CMakeUserPresets.json` 指向 Geant4 安装路径。不要把包含本机绝对路径的配置提交到 GitHub。

### Python 提示找不到事件文件

说明 `analysis/classify_absorption_groups.py` 在 `build/` 下找不到材料目录中声明的 `event_file`。先确认十种材料仿真都运行过，再检查 `source_models/materials/material_catalog.csv` 中的 `event_file` 是否与实际文件名一致。

### accuracy 和公开结果不完全一致

Geant4 仿真具有随机性。如果重新生成事件 CSV 且没有固定随机种子，accuracy 可能有小幅波动。当前公开结果应理解为一次可复查证据包，而不是数学上永远固定的常数。

### 能不能只新增一个材料

可以，但流程不是只改材料目录。你需要确认 `src/DetectorConstruction.cc` 中有该材料的 Geant4 定义，新建 `source_models/config/undergrad_batch/<material>.txt`，补充 `material_catalog.csv`，运行 Geant4 生成该材料事件 CSV，再重新运行 Python 脚本生成新的证据包。新增材料后，论文和 README 中的结论也要同步改写。

## 8. 复现边界

本地复现能验证的是：当前代码、配置和 Python 脚本能重新生成仿真事件数据，并得到同一类低/高吸收组分类结果。它不能验证真实设备性能、复杂矿流泛化能力、工业部署能力或所有矿物覆盖能力。

如果导师要求进一步提高证据等级，下一步应增加固定随机种子、多种子重复、不同厚度和几何、混合材料、独立 run-level 测试以及真实设备或真实样品数据对照。
