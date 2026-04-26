# 本机运行指南

这份文档面向组员本机复现。它解释如何配置 Geant4/CMake/Python，如何运行六材料仿真，以及如何生成 `results/undergrad_validation/` 证据包。

## 1. 需要安装什么

你需要：

- Geant4
- CMake
- 支持 C++17 的编译器
- Python 3.10 或更高版本
- Python 包：`pandas`、`scikit-learn`

VS Code 的 CMake 插件只是界面工具，不等于安装了 CMake，也不等于安装了 Geant4。真正让项目能编译的是：系统里有 `cmake` 命令，且 CMake 能找到 Geant4 安装目录中的 `Geant4Config.cmake`。

## 2. 让 CMake 找到 Geant4

如果 CMake 报错说找不到 Geant4，通常需要先加载 Geant4 环境脚本：

```bash
source /path/to/geant4-install/bin/geant4.sh
```

也可以在本机创建 `CMakeUserPresets.json`，但不要提交到 GitHub，因为里面会包含个人机器路径：

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "local-geant4",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "Geant4_DIR": "/path/to/geant4-install/lib/cmake/Geant4",
        "CMAKE_PREFIX_PATH": "/path/to/geant4-install"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "local-geant4",
      "configurePreset": "local-geant4"
    }
  ]
}
```

这个本地配置的好处是：每个人可以使用自己的 Geant4 安装路径，仓库不会泄露个人路径，也不会让别人的机器被你的路径卡住。

## 3. 编译项目

在仓库根目录运行：

```bash
cmake -S . -B build
cmake --build build
```

如果使用本地 preset：

```bash
cmake --preset local-geant4
cmake --build --preset local-geant4
```

编译成功后，`build/` 下会出现可执行程序 `xrt_sorter`。`build/` 是本机生成目录，不提交到 GitHub。

## 4. 运行六材料仿真

公开分类脚本需要六种材料的事件文件，所以不要只跑默认配置。进入 `build/` 后运行：

```bash
cd build

for material in quartz orthoclase calcite pyrite hematite magnetite; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
```

每个材料配置都会运行 `/run/beamOn 5000`，生成 5000 个仿真事件。输出文件包括：

- `build/xrt_real_source_<material>_events.csv`
- `build/xrt_real_source_<material>_hits.csv`
- `build/xrt_real_source_<material>_metadata.json`

如果运行时报 `libG4*.so` 找不到，说明当前 shell 没加载 Geant4 运行时库路径。先执行 `source /path/to/geant4-install/bin/geant4.sh`，再重新运行。

## 5. 生成验证证据包

安装 Python 依赖后运行：

```bash
pip install pandas scikit-learn
python analysis/classify_absorption_groups.py
```

脚本会读取 `build/` 下六个 `*_events.csv`，并生成：

- `results/undergrad_validation/event_row_summary.csv`
- `results/undergrad_validation/absorption_group_virtual_samples.csv`
- `results/undergrad_validation/train_test_split_samples.csv`
- `results/undergrad_validation/feature_group_summary.csv`
- `results/undergrad_validation/absorption_group_classification_summary.csv`
- `results/undergrad_validation/absorption_group_confusion_threshold.csv`
- `results/undergrad_validation/absorption_group_confusion_logistic_1f.csv`
- `results/undergrad_validation/absorption_group_confusion_logistic_3f.csv`
- `results/undergrad_validation/test_predictions.csv`
- `results/undergrad_validation/validation_manifest.json`

同时，脚本会刷新 `results/absorption_group_classification_summary.csv` 和 `results/*confusion*.csv` 作为兼容入口。

## 6. 如何检查结果是否合理

先打开 `results/undergrad_validation/validation_manifest.json`，确认以下内容：

- 材料数为 6。
- 每种材料 `event_count` 为 5000。
- 每种材料 `complete_virtual_samples` 为 50。
- `total_train_samples` 为 150。
- `total_test_samples` 为 150。

再打开 `results/undergrad_validation/absorption_group_classification_summary.csv`。当前提交证据包中，阈值法、单特征 Logistic Regression、三特征 Logistic Regression 的测试 accuracy 分别为 `0.9800`、`0.9867`、`0.9933`。

最后看混淆矩阵。行是真实标签，列是预测标签。它能告诉你错误发生在低吸收组还是高吸收组，比单独看 accuracy 更可靠。

## 7. 常见问题

**CMake 找不到 Geant4。**
先确认 Geant4 已安装，再设置 `Geant4_DIR` 或 `CMAKE_PREFIX_PATH`。VS Code 插件不能替代 Geant4 安装。

**编译成功但运行时报动态库找不到。**
执行 `source /path/to/geant4-install/bin/geant4.sh`，让当前 shell 加载 Geant4 的运行时库路径。

**Python 提示没有 pandas 或 scikit-learn。**
运行 `pip install pandas scikit-learn`，或者在自己的虚拟环境里安装。

**Python 提示缺少某个 `xrt_real_source_*_events.csv`。**
说明六材料仿真没有跑完整。回到第 4 节，把六个材料都运行一遍。

**复跑结果和 GitHub 上数字略有不同。**
Geant4 仿真有随机性。如果没有固定随机种子，重新生成事件 CSV 后结果可能小幅波动。论文和讲解应以当前提交的 `results/undergrad_validation/` 证据包为准，并说明这个数字不是普适常数。

## 8. 重要边界

本仓库结果只说明当前六材料、固定几何、仿真数据、粗粒度吸收组二分类任务下的表现。它不是真实设备指标，不证明能覆盖所有矿物，也不证明能直接用于工业在线控制。
