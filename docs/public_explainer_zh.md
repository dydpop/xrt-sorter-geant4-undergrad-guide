# 通俗讲解：这个 XRT 仿真项目厉害在哪里

这个项目可以理解成一个“电脑里的 X 射线矿物分选小实验台”。真实设备要有 X 射线源、矿石样本、探测器、安全屏蔽和控制系统；我们当前做的是本科级仿真版本：用 Geant4 在电脑里模拟 X 射线穿过不同材料，再把探测器信号导出成表格，用 Python 做基础分析。

项目的完整性体现在它不是只写了一段分类代码。仓库里有 C++ 仿真程序，有十种材料配置，有 W 靶 120 kV 能谱，有事件级 CSV 输出，有 Python 特征构造，有训练/测试拆分，有混淆矩阵，也有论文和组员指南。它的价值是把“物理仿真 - 数据生成 - 特征提取 - 分类验证 - 结果呈现”串成了一个闭环。

当前公开验证使用十种材料：Quartz、Calcite、Orthoclase、Albite、Dolomite 属于低吸收组，Pyrite、Hematite、Magnetite、Chalcopyrite、Galena 属于高吸收组。每种材料运行 5000 个仿真事件，每 100 个事件聚合成 1 个虚拟样本，因此每种材料有 50 个虚拟样本。每种材料前 25 个样本用于训练，后 25 个样本用于测试。也就是说，结果不是直接拿训练集报准确率。

当前证据包中，三特征 Logistic Regression 在 250 个测试虚拟样本上正确 249 个，测试 accuracy 为 0.9960。这个数字说明当前十材料仿真配置中，XRT 透射相关特征具有明显区分度。但它不能被理解成真实设备准确率，也不能说明世界上所有矿物都能被同样准确地区分。

如果只想快速理解项目，先看 `README.md` 和 `docs/TEAM_GUIDE_zh.md`。如果想复查数字从哪里来，看 `results/undergrad_validation/validation_manifest.json`、`train_test_split_samples.csv` 和 `absorption_group_classification_summary.csv`。如果想自己运行，看 `docs/RUN_LOCALLY_zh.md`。
