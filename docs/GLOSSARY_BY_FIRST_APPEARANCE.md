# 术语表：按首次出现顺序排列

本术语表按 `README.md`、`docs/TEAM_GUIDE_zh.md`、`docs/RUN_LOCALLY_zh.md` 中术语首次出现的大致顺序排列，不按拼音或字母排序。读文档时遇到新词，可以从上往下顺着查。

| 顺序 | 术语 | 英文/缩写 | 首次出现位置 | 文件命名中常见写法 | 大白话解释 |
| --- | --- | --- | --- | --- | --- |
| 1 | Geant4 | Geant4 | `README.md` 标题 | `G4`, `geant4`, `CMakeLists` | 用 C++ 做粒子和材料相互作用仿真的工具包。 |
| 2 | XRT | X-ray Transmission | `README.md` 标题 | `xrt` | X 射线透射，关注射线穿过材料后信号怎样变化。 |
| 3 | 矿物分选 | mineral sorting | `README.md` 标题 | `sorter`, `sorting` | 把不同性质的矿石或矿物分开。 |
| 4 | 仿真 | simulation | `README.md` 标题 | `simulation`, `run` | 在电脑里搭建实验过程，而不是直接做实物实验。 |
| 5 | X 射线 | X-ray | `README.md` 第 1 段 | `xray`, `spectrum` | 一种能穿透物体的射线，穿过不同材料时衰减不同。 |
| 6 | 矿石样本 | ore sample | `README.md` 第 1 段 | `ore`, `material` | 仿真中被 X 射线照射的材料块。 |
| 7 | 探测器 | detector | `README.md` 第 1 段 | `detector`, `hit` | 接收穿过矿石后的信号，类似虚拟传感器。 |
| 8 | C++ | C++ | `README.md` 第 1 段 | `.cc`, `.hh`, `src/`, `include/` | Geant4 项目主要使用的编程语言。 |
| 9 | 事件数据 | event data | `README.md` 第 1 段 | `events.csv`, `event_id` | 每一次模拟粒子发射和探测记录形成的一行或一组数据。 |
| 10 | Python | Python | `README.md` 第 1 段 | `analysis/`, `.py` | 这里用来读取 CSV、提取特征、做基础分类。 |
| 11 | 特征提取 | feature extraction | `README.md` 第 1 段 | `feature`, `rate`, `edep` | 从原始数据里整理出可用于判断的数值。 |
| 12 | 基础分类 | baseline classification | `README.md` 第 1 段 | `classification`, `classifier` | 用简单、可解释的方法判断样本属于哪一类。 |
| 13 | 图表 | figures/tables | `README.md` 第 1 段 | `figures/`, `results/` | 用图片和表格把结果展示出来。 |
| 14 | CSV | Comma-Separated Values | `README.md` 项目亮点 | `.csv` | 表格数据文件，仿真输出和 Python 分析都常用。 |
| 15 | 源项 | source term | `README.md` 项目亮点 | `source`, `source_models` | 仿真里描述 X 射线从哪里来、能量如何分布的设置。 |
| 16 | W 靶 | tungsten target | `README.md` 图文导览 | `w_target` | 用钨靶产生 X 射线的一种源模型描述。 |
| 17 | 120 kV | 120 kilovolt | `README.md` 图文导览 | `120kV` | X 射线管电压参数，用来限定能谱范围。 |
| 18 | 能谱 | spectrum | `README.md` 图文导览 | `spectrum`, `w_target_120kV` | 不同能量的 X 射线占多少比例。 |
| 19 | 透射率 | transmission rate | `README.md` 项目亮点 | `transmission_rate` | 有多少射线信号穿过样本并到达探测器。 |
| 20 | 能量沉积 | energy deposition | `README.md` 项目亮点 | `edep`, `detector_edep` | 射线在探测器或材料中留下的能量。 |
| 21 | 直接命中 | direct hit | `README.md` 项目亮点 | `direct`, `directscatter` | 没有明显偏转、直接到达探测器的命中。 |
| 22 | 散射命中 | scatter hit | `README.md` 项目亮点 | `scatter`, `directscatter` | 路径发生偏转后到达探测器的命中。 |
| 23 | 粗粒度吸收组 | coarse absorption group | `README.md` 项目亮点 | `low_absorption`, `high_absorption` | 先把材料大致分成低吸收和高吸收两类。 |
| 24 | accuracy | accuracy | `README.md` 图文导览 | `accuracy` | 分类正确比例；`0.98` 表示当前仿真任务中测试样本有 98% 分对。 |
| 25 | 样本构造 | sample construction | `README.md` 图文导览 | `virtual_samples`, `PHOTONS_PER_SAMPLE` | 把多个事件合成一个可用于分类的样本。 |
| 26 | 划分方式 | train/test split | `README.md` 图文导览 | `train_df`, `test_df` | 把样本分成训练部分和测试部分。 |
| 27 | 实物设备测试指标 | physical equipment metric | `README.md` 图文导览 | 无固定文件名 | 真实机器测试得到的指标；本项目公开结果不属于这一类。 |
| 28 | Mermaid | Mermaid | `README.md` 项目流程 | `mermaid` | Markdown 里画流程图的一种语法。 |
| 29 | CMake | CMake | `README.md` 快速运行 | `CMakeLists.txt`, `cmake` | C++ 项目的构建工具，用来生成编译配置。 |
| 30 | C++17 编译器 | C++17 compiler | `docs/RUN_LOCALLY_zh.md` | `g++`, `clang++`, `MSVC` | 能编译 C++17 标准代码的编译器。 |
| 31 | 运行宏 | Geant4 macro | `README.md` 快速运行 | `.mac`, `run_research.mac` | Geant4 的命令脚本，用来设置输出和发射事件数。 |
| 32 | 六材料配置 | six-material configs | `README.md` 快速运行 | `undergrad_batch/` | Quartz、Orthoclase、Calcite、Pyrite、Hematite、Magnetite 六个公开复现配置。 |
| 33 | 虚拟样本 | virtual sample | `docs/RUN_LOCALLY_zh.md` | `absorption_group_virtual_samples.csv` | 由一组仿真事件汇总成的分类样本。 |
| 34 | 混淆矩阵 | confusion matrix | `docs/RUN_LOCALLY_zh.md` | `confusion`, `confusion_matrix` | 显示哪些类别被分对、哪些类别被分错的表。 |
| 35 | 环境脚本 | environment setup script | `docs/RUN_LOCALLY_zh.md` | `geant4.sh` | 安装 Geant4 后用来设置环境变量的脚本。 |
| 36 | `CMAKE_PREFIX_PATH` | CMAKE_PREFIX_PATH | `docs/RUN_LOCALLY_zh.md` | `CMAKE_PREFIX_PATH` | 告诉 CMake 去哪里找已安装库。 |
| 37 | `Geant4_DIR` | Geant4_DIR | `docs/RUN_LOCALLY_zh.md` | `Geant4_DIR` | 直接告诉 CMake Geant4 配置文件所在目录。 |
| 38 | Pull Request | Pull Request / PR | `docs/TEAM_GUIDE_zh.md` | `PR` | 在 GitHub 上提交改动建议的协作方式。 |
| 39 | 主分支 | main branch | `docs/TEAM_GUIDE_zh.md` | `main` | 仓库默认稳定分支，不建议直接乱改。 |
