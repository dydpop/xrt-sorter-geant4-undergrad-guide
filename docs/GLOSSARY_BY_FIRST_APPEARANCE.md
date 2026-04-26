# 术语表

这份术语表按项目阅读中常见的概念组织，不追求教科书式完整定义，而是帮助没参与过项目的组员快速看懂仓库、论文和结果表。

## XRT

XRT 是 X-ray Transmission 的缩写，中文可理解为 X 射线透射。它关注 X 射线穿过材料后还剩多少信号。不同材料对 X 射线吸收能力不同，因此透射信号可以用来辅助区分材料。

## Geant4

Geant4 是用于模拟粒子与物质相互作用的工具包。本项目用它模拟 X 射线光子穿过矿物样本，并记录探测器响应。它负责生成仿真数据，不负责自动训练机器学习模型。

## event

event 是一次仿真事件。可以把它理解为一次模拟发射和探测过程。当前每种材料运行 5000 个 event。

## hits.csv

命中级 CSV，记录探测器命中的更细信息，例如命中位置、能量、是否为 primary、偏转角等。它适合用来理解探测器响应细节。

## events.csv

事件级 CSV，每一行对应一个 event。当前机器学习验证主要读取这个文件，因为它更适合做稳定的样本级统计。

## primary gamma

primary gamma 指由源项直接产生的原始 gamma 光子。`primary_gamma_entries` 表示 primary gamma 到达探测器的次数。这个字段与透射率关系很直接，是当前分类验证最重要的数据来源之一。

## detector energy deposition

探测器能量沉积，文件字段为 `detector_edep_keV`。它表示某个事件中探测器吸收到的能量，单位是 keV。Python 会把 100 个 event 的能量沉积求平均，形成 `mean_detector_edep_keV`。

## virtual sample

虚拟样本是 Python 分析中的样本单位。当前规则是每 100 个 event 聚合为 1 个 virtual sample。每种材料 5000 个 event，因此形成 50 个虚拟样本。

## feature

feature 是机器学习使用的特征。本项目的核心特征包括 `primary_transmission_rate`、`mean_detector_edep_keV` 和 `detector_gamma_rate`。这些特征不是黑箱变量，而是由探测器响应统计得到的物理相关量。

## primary_transmission_rate

主 gamma 透射率，计算方式是 `primary_gamma_entries_sum / n_events`。大白话说，就是一组 event 里有多少 primary gamma 成功到达探测器。材料越强吸收 X 射线，这个值通常越低。

## detector_gamma_rate

探测器 gamma 命中率，计算方式是 `detector_gamma_entries_sum / n_events`。它表示每个虚拟样本中探测器平均接收到多少 gamma 命中。

## low_absorption / high_absorption

低吸收组和高吸收组是当前本科验证的分类标签。它们不是具体矿物名称，而是粗粒度吸收组。当前任务是二分类：判断虚拟样本属于低吸收组还是高吸收组。

## train/test split

训练/测试拆分。当前每种材料前 25 个虚拟样本作为训练集，后 25 个作为测试集。十种材料合计后，训练集 250 个样本，测试集 250 个样本。

## threshold baseline

阈值法 baseline。它只使用主 gamma 透射率，在训练集上计算低吸收组和高吸收组的平均值，再取中点作为阈值。它简单但可解释，用来判断单一透射率特征是否已经有效。

## StandardScaler

scikit-learn 中的标准化工具。它把特征转换到更适合线性模型学习的尺度。当前 Logistic Regression 模型前面都接了 `StandardScaler`。

## Logistic Regression

逻辑回归，是一种经典线性分类模型。本项目用它做低/高吸收组二分类。它不是决策树，也不是深度学习模型；它的优势是基础、稳定、容易解释。

## accuracy

Accuracy 是测试样本中预测正确的比例。当前三特征 Logistic Regression 的 accuracy 是 0.9960，意思是 250 个测试虚拟样本中正确 249 个。

## confusion matrix

混淆矩阵展示真实标签和预测标签的对应关系。它能告诉我们错误发生在哪一类。当前三特征 Logistic Regression 中，低吸收组错 1 个，高吸收组错 0 个。

## validation manifest

`validation_manifest.json` 是证据包的总说明，记录材料列表、样本政策、软件版本、生成时间和结论边界。复查结果时应优先看它。

## same-distribution simulation split

同分布仿真切分。意思是训练集和测试集来自同一套仿真条件，只是样本编号不同。这能验证当前仿真链路下的分类能力，但不能等同于真实设备验证或跨工况泛化验证。

## product coverage

产品覆盖指系统能否在真实复杂场景中覆盖足够多矿物、工况和错误类型。当前项目没有证明产品覆盖；它证明的是本科级仿真和分析闭环。
