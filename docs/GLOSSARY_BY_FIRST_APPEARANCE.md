# 术语表：按首次出现顺序排列

本术语表按 `README.md`、`docs/TEAM_GUIDE_zh.md`、`docs/RUN_LOCALLY_zh.md` 中术语首次出现的大致顺序排列，不按拼音或字母排序。它重点解释术语、英文名、文件命名和项目里的实际变量名。

| 顺序 | 术语 | 英文/缩写 | 常见文件或变量名 | 大白话解释 |
| --- | --- | --- | --- | --- |
| 1 | Geant4 | Geant4 | `G4`, `CMakeLists.txt`, `src/` | 用 C++ 做粒子和材料相互作用仿真的工具包。 |
| 2 | XRT | X-ray Transmission | `xrt`, `XRT` | X 射线透射，关注射线穿过材料后信号怎样变化。 |
| 3 | 矿物分选 | mineral sorting | `sorter`, `sorting` | 根据物理信号把不同性质的矿物或矿石分开。 |
| 4 | 仿真 | simulation | `run`, `simulation` | 在电脑里模拟实验过程，不是直接做真实设备实验。 |
| 5 | X 射线 | X-ray | `spectrum`, `w_target_120kV` | 能穿透物体的射线，不同材料会造成不同衰减。 |
| 6 | 矿石样本 | ore sample | `ore`, `ore_primary_material` | 仿真中被 X 射线照射的材料块。 |
| 7 | 探测器 | detector | `detector`, `hit` | 接收穿过样本后的信号，相当于虚拟传感器。 |
| 8 | 事件 | event | `event_id`, `events.csv` | 一次粒子发射和传播过程，事件表中通常对应一行。 |
| 9 | 命中 | hit | `hits.csv`, `RecordDetectorHit` | 粒子进入探测器边界时记录的一次命中。 |
| 10 | CSV | Comma-Separated Values | `.csv` | 表格文本文件，仿真输出和 Python 分析都使用它。 |
| 11 | 源项 | source term | `source_mode`, `source_models/` | 描述 X 射线从哪里来、能量如何分布、方向如何设定。 |
| 12 | W 靶 | tungsten target | `w_target_120kV_1mmAl.csv` | 钨靶 X 射线源的能谱描述。 |
| 13 | 120 kV | 120 kilovolt | `120kV` | X 射线管电压参数，用来限定能谱范围。 |
| 14 | 能谱 | spectrum | `spectrum_file` | 不同能量的 X 射线占多少比例。 |
| 15 | 材料目录 | material catalog | `material_catalog.csv` | 记录材料名称、化学式、密度、分组标签、配置文件、事件文件和证据状态的表格。 |
| 16 | 十材料配置 | ten-material configs | `undergrad_batch/*.txt` | 当前公开复现配置，包含 5 个低吸收材料和 5 个高吸收材料。 |
| 17 | `event_id` | event identifier | `event_id` | 仿真事件编号，本次每种材料为 0 到 4999。 |
| 18 | `detector_edep_keV` | detector energy deposition | `detector_edep_keV` | 当前事件在探测器中沉积的能量，单位 keV。 |
| 19 | `detector_gamma_entries` | detector gamma entries | `detector_gamma_entries` | 当前事件中 gamma 进入探测器的计数。 |
| 20 | `primary_gamma_entries` | primary gamma entries | `primary_gamma_entries` | 当前事件中 primary gamma 到达探测器的计数。 |
| 21 | 虚拟样本 | virtual sample | `sample_id`, `virtual_samples` | 由多个 event 聚合成的分类样本，本项目每 100 个 event 形成一个。 |
| 22 | `PHOTONS_PER_SAMPLE` | photons per virtual sample | `PHOTONS_PER_SAMPLE = 100` | Python 脚本中的聚合规则，决定每个虚拟样本包含多少 event。 |
| 23 | `sample_id` | sample identifier | `sample_id` | 虚拟样本编号，由 `event_id // 100` 得到。 |
| 24 | 吸收组 | absorption group | `low_absorption`, `high_absorption` | 本科任务中的粗粒度标签，分为低吸收组和高吸收组。 |
| 25 | `group_label` | group label | `group_label` | Python 中的类别标签字段。 |
| 26 | 主 gamma 透射率 | primary transmission rate | `primary_transmission_rate` | 每个虚拟样本中 primary gamma 到达探测器的比例。 |
| 27 | 平均探测器能量沉积 | mean detector energy deposition | `mean_detector_edep_keV` | 每个虚拟样本平均每次 event 的探测器能量沉积。 |
| 28 | 探测器 gamma 命中率 | detector gamma rate | `detector_gamma_rate` | 每个虚拟样本中探测器 gamma 命中的平均比例。 |
| 29 | 训练集 | training set | `train_df`, `split=train` | 用来确定阈值或拟合模型的样本。本项目每种材料前 25 个样本训练。 |
| 30 | 测试集 | test set | `test_df`, `split=test` | 用来报告结果的样本。本项目每种材料后 25 个样本测试。 |
| 31 | 阈值法 | threshold classifier | `A_threshold_transmission_only`, `threshold` | 用训练集计算一个透射率分界值，再按高低判断类别。 |
| 32 | `threshold` | decision threshold | `threshold` | 低吸收组和高吸收组训练均值的中点。 |
| 33 | 标准化 | standardization | `StandardScaler` | 把特征缩放到更适合线性模型处理的尺度。 |
| 34 | Logistic Regression | logistic regression | `LogisticRegression` | 经典二分类方法，用特征学习线性分类边界。 |
| 35 | accuracy | accuracy | `accuracy_score`, `accuracy` | 测试样本中预测正确的比例。 |
| 36 | 混淆矩阵 | confusion matrix | `confusion_matrix`, `confusion*.csv` | 显示真实类别和预测类别如何对应的表。 |
| 37 | 测试集分母 | test denominator | `test_samples` | 当前证据包为 250 个测试虚拟样本，不是 event 数。 |
| 38 | 近似直接透射 | direct primary | `is_direct_primary` | primary gamma 到达探测器且偏转角小于 1 度的工程近似。 |
| 39 | 散射后透射 | scattered primary | `is_scattered_primary` | primary gamma 到达探测器但偏转角不小于 1 度的工程近似。 |
| 40 | CMake | CMake | `cmake`, `CMakeLists.txt` | C++ 项目构建工具，用来生成编译配置。 |
| 41 | `Geant4_DIR` | Geant4_DIR | `Geant4Config.cmake` | 告诉 CMake Geant4 配置文件在哪个目录。 |
| 42 | `CMAKE_PREFIX_PATH` | CMAKE_PREFIX_PATH | `CMAKE_PREFIX_PATH` | 告诉 CMake 去哪些安装前缀里找依赖包。 |
| 43 | 运行宏 | Geant4 macro | `.mac`, `run_research.mac` | Geant4 命令脚本，本项目用它执行 `/run/beamOn 5000`。 |
| 44 | 验证证据包 | validation package | `results/undergrad_validation/` | 本轮整理出的紧凑证据目录，用于追溯数据规模、拆分和结果。 |
| 45 | manifest | manifest | `validation_manifest.json` | 记录证据包生成方式、样本政策、软件版本和结论边界的 JSON 文件。 |
| 46 | 本科级边界 | undergraduate boundary | `claim_boundary` | 说明本项目是仿真和基础分类验证，不是工业设备验证。 |
| 47 | 物理/化学特征 | physical/chemical descriptors | `formula`, `density_g_cm3`, `primary_transmission_rate` | 用材料组成、密度和仿真响应描述材料，而不是只记住材料名字。 |
| 48 | 候选检索 | candidate retrieval | material dictionary lookup | 后续研究可以把仿真特征和矿物字典结合，给出候选材料或人工复核线索；当前公开仓库只做本科级验证。 |

## 常见误解

| 误解 | 正确理解 |
| --- | --- |
| 一个 event 就是一个机器学习样本 | 当前项目每 100 个 event 聚合为 1 个虚拟样本。 |
| `0.9960` 表示所有矿物都能分对 | 它只表示当前十材料、固定几何、仿真数据、粗粒度吸收组测试集上的结果。 |
| 仿真数据等于真实设备数据 | 仿真能降低探索成本，但不能替代真实设备、真实矿流和现场验证。 |
| 新增材料只要改词典就够 | 当前至少还需要 C++ 材料定义、配置文件、Geant4 运行和新的证据包。 |
