# 本机运行说明

这份说明面向组员。目标是让你知道需要什么环境、怎样从 `clone` 跑到第一批 CSV、怎样得到分类结果，以及出错时先检查哪里。

## 1. 先判断你要做哪件事

如果你只想看懂项目，不需要安装 Geant4。直接读 `README.md`、`docs/TEAM_GUIDE_zh.md`、`figures/` 和 `results/`。

如果你想重新跑仿真，需要安装 Git、CMake、支持 C++17 的编译器、Geant4、Python 3.10 或更高版本。

本公开包按 Ubuntu/WSL 风格命令整理。打包校验环境可正常执行 CMake、g++ 和 Python 静态检查；Geant4 需要在你的电脑上自行安装并正确加载环境。

## 2. 安装或加载依赖

Python 包：

```bash
python -m pip install pandas scikit-learn
```

检查基础命令：

```bash
git --version
cmake --version
g++ --version
python --version
```

Geant4 安装后通常需要先加载环境脚本。下面是示例，路径要换成你自己电脑上的 Geant4 安装位置：

```bash
source /path/to/geant4-install/bin/geant4.sh
```

如果 CMake 找不到 Geant4，可以显式设置其中一种路径：

```bash
export CMAKE_PREFIX_PATH=/path/to/geant4-install:$CMAKE_PREFIX_PATH
```

或在配置时写：

```bash
cmake -DGeant4_DIR=/path/to/geant4-install/lib/cmake/Geant4 -S . -B build
```

## 3. VS Code 和本机配置建议

VS Code 的 CMake 插件不是 Geant4 本身。插件只是调用 CMake；真正让项目能配置成功的是让 CMake 找到 Geant4 的 `Geant4Config.cmake`。

推荐做法是把个人电脑上的 Geant4 路径放在本机配置里，而不是直接写进公开仓库。这样有几个好处：

- 本机以后不用每次都手写 `-DGeant4_DIR=...`，VS Code 和命令行都能直接配置项目。
- 公开仓库不会暴露个人电脑路径，也不会让其他组员误用你的路径。
- 每个组员可以按自己的安装位置设置 Geant4，仓库本身保持通用。
- 后续调试时更稳定，问题会集中在代码或依赖版本上，而不是每次都卡在“找不到 Geant4”。

如果你使用 VS Code CMake Tools，可以在自己电脑上创建本地文件 `CMakeUserPresets.json`。这个文件建议只保留在本机，不提交到 Git：

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "local-geant4",
      "displayName": "Local Geant4",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "Geant4_DIR": "/path/to/geant4-install/lib/cmake/Geant4"
      },
      "environment": {
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

然后运行：

```bash
cmake --preset local-geant4
cmake --build --preset local-geant4
```

如果你不想让本机配置进入 Git，可以把下面内容加入 `.git/info/exclude`：

```text
.vscode/
CMakeUserPresets.json
```

## 4. 获取代码

```bash
git clone https://github.com/adgjlqetuozcbm/xrt-sorter-geant4-undergrad-guide.git
cd xrt-sorter-geant4-undergrad-guide
```

## 5. 编译 C++ 仿真程序

```bash
cmake -S . -B build
cmake --build build
```

成功标志：

```bash
test -x build/xrt_sorter && echo "xrt_sorter build ok"
```

如果你在 Windows 原生命令行运行，生成文件名可能带 `.exe`。新手优先建议使用 WSL 或 Linux 环境。

## 6. 先跑一个材料做烟雾测试

这一步只检查程序能否启动，不用于生成完整分类结果：

```bash
cd build
XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/quartz.txt \
  ./xrt_sorter ../analysis/configs/run_research.mac
cd ..
```

成功后应能看到：

```bash
ls build/xrt_real_source_quartz_events.csv
ls build/xrt_real_source_quartz_hits.csv
ls build/xrt_real_source_quartz_metadata.json
```

其中 `*_events.csv` 是分类脚本需要的事件表，`*_hits.csv` 是命中明细，`*_metadata.json` 是该次运行的配置摘要。

## 7. 跑完整六材料公开复现配置

分类脚本需要六种材料的事件 CSV。请从仓库根目录执行：

```bash
cmake --build build
cd build

for material in quartz orthoclase calcite pyrite hematite magnetite; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
```

成功后应能看到这六个事件文件：

```bash
ls build/xrt_real_source_*_events.csv
```

文件名应包括：

- `xrt_real_source_quartz_events.csv`
- `xrt_real_source_orthoclase_events.csv`
- `xrt_real_source_calcite_events.csv`
- `xrt_real_source_pyrite_events.csv`
- `xrt_real_source_hematite_events.csv`
- `xrt_real_source_magnetite_events.csv`

## 8. 运行基础分类

```bash
python analysis/classify_absorption_groups.py
```

脚本会读取 `build/` 下的六个事件 CSV，并生成或更新：

- `analysis/results/absorption_group_virtual_samples.csv`
- `analysis/results/absorption_group_classification_summary.csv`
- `analysis/results/absorption_group_confusion_threshold.csv`
- `analysis/results/absorption_group_confusion_logistic_1f.csv`
- `analysis/results/absorption_group_confusion_logistic_3f.csv`

公开仓库已经提供整理好的结果表在 `results/`。如果你暂时没有跑通 Geant4，也可以先读 `results/absorption_group_classification_summary.csv` 和 `figures/elementary_absorption_accuracy.png`。

## 9. 常见错误

### 找不到 Geant4

表现：

```text
Could not find a package configuration file provided by "Geant4"
```

处理：

- 确认 Geant4 已安装。
- 确认已经执行 `source /path/to/geant4-install/bin/geant4.sh`。
- 尝试设置 `CMAKE_PREFIX_PATH` 或 `Geant4_DIR`。
- 删除旧构建目录后重新配置：`rm -rf build && cmake -S . -B build`。

### 找不到事件 CSV

表现：

```text
FileNotFoundError: xrt_real_source_quartz_events.csv
```

处理：

- 先跑第 7 节的六材料仿真。
- 确认当前目录是仓库根目录，而不是 `build/`。
- 确认 `build/xrt_real_source_*_events.csv` 至少有六个文件。

### Python 缺少 pandas 或 scikit-learn

表现：

```text
ModuleNotFoundError
```

处理：

```bash
python -m pip install pandas scikit-learn
```

### 输出文件为空或数量太少

处理：

- 打开 `analysis/configs/run_research.mac`，确认 `/run/beamOn 5000` 没被改小。
- 查看终端中的 `XRT Research Run Summary`。
- 检查对应的 `build/*_metadata.json` 是否记录了正确材料和事件数。

## 10. 不想跑也可以怎么学

如果你的电脑暂时没有 Geant4，按这个顺序学习：

1. 用 2 分钟扫一遍 `README.md`。
2. 读 `docs/TEAM_GUIDE_zh.md`。
3. 遇到术语查 `docs/GLOSSARY_BY_FIRST_APPEARANCE.md`。
4. 看 `figures/` 里的图。
5. 打开 `results/absorption_group_classification_summary.csv`。
6. 阅读 `analysis/classify_absorption_groups.py`。
7. 最后再看 C++ 代码。
