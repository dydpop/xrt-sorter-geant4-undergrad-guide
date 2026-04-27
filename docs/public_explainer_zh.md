# 通俗讲解：这个 XRT 仿真项目到底做了什么

可以把这个项目理解成一个“电脑里的 X 射线矿物分选实验台”。真实设备需要 X 射线源、矿石样本、探测器、安全屏蔽、输送机构和控制系统；我们当前做的是本科级仿真版本：用 Geant4 在电脑里模拟 X 射线穿过不同矿物材料，再把探测器信号导出成表格，用 Python 做数据整理、特征提取和基础分类验证。

项目的价值不在于某一个材料数量，也不在于单独一个 accuracy 数字，而在于链路完整。仓库里有 C++ 仿真程序，有材料目录和配置文件，有 W 靶 120 kV 能谱，有事件级 CSV 和命中级 CSV 输出，有 Python 数据质量检查，有每 100 个事件聚合成一个虚拟样本的特征工程，有训练/测试拆分，有阈值法和 Logistic Regression 的测试结果，也有论文和组员指南。也就是说，它把“物理仿真-数据生成-特征提取-分类验证-结果呈现”串成了一个可以复查的闭环。

当前公开验证使用十种单一材料。低吸收组包括 Quartz、Calcite、Orthoclase、Albite 和 Dolomite，高吸收组包括 Pyrite、Hematite、Magnetite、Chalcopyrite 和 Galena。每种材料运行 5000 个仿真事件，每 100 个事件聚合成 1 个虚拟样本，因此每种材料有 50 个虚拟样本。每种材料前 25 个样本用于训练，后 25 个样本用于测试，所以最终训练集和测试集各有 250 个样本。

机器学习部分不是复杂黑箱模型。项目先做一个只看主 gamma 透射率的阈值法 baseline，再做两个 `StandardScaler + LogisticRegression` 线性分类模型：一个只用主 gamma 透射率，一个使用主 gamma 透射率、平均探测器能量沉积和 gamma 命中率三个特征。当前证据包中，三特征 Logistic Regression 在 250 个测试虚拟样本上正确 249 个，测试 accuracy 为 0.9960。这个结果说明当前仿真配置中，XRT 透射相关特征对低/高吸收组有明显区分度。

这个结果不能被理解成真实设备准确率，也不能说明所有矿物都能被同样准确地区分。原因很简单：当前数据来自仿真，不是真实 XRT 设备；样本是单一材料 slab，不是真实复杂矿石；训练集和测试集来自同一类仿真配置，不是跨设备、跨厚度、跨矿区或跨随机种子的外部验证。因此，最严谨的说法是：本项目完成了本科级 XRT 仿真与分析闭环，并在当前十材料仿真证据包上验证了粗粒度吸收组分类能力。

我们还额外做了十材料物种级诊断。结果显示，直接预测十个材料名时主方法 top-1 accuracy 只有 `0.464`，没有达到验收标准。这说明项目目前不能宣称“已经能识别十种矿物”，只能把材料级分选作为下一阶段实验方向。

如果只想快速理解项目，先看 `README.md` 和 `docs/TEAM_GUIDE_zh.md`。如果想复查数字从哪里来，看 `results/undergrad_validation/validation_manifest.json`、`train_test_split_samples.csv` 和 `absorption_group_classification_summary.csv`。如果想自己运行，看 `docs/RUN_LOCALLY_zh.md`。如果要写论文或答辩，看 `paper/main_thesis_HIT_revised_zh.md`。
