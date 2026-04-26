# 本科项目演示报告

## 项目目标

本项目目标是构建一个基于 Geant4 的 XRT 矿物分选仿真原型，并完成从仿真输出到基础分类结果的闭环展示。公开仓库只展示本科级仿真系统和基础验证，不声称真实设备部署。

## 已完成链路

```mermaid
flowchart LR
    A["X 射线源配置"] --> B["Geant4 仿真"]
    B --> C["探测器事件数据"]
    C --> D["Python 虚拟样本"]
    D --> E["训练/测试拆分"]
    E --> F["粗粒度吸收组分类"]
    F --> G["结果表和图表"]
```

## 关键结果

| 文件 | 说明 |
| --- | --- |
| `undergrad_validation/validation_manifest.json` | 当前证据包总说明 |
| `undergrad_validation/event_row_summary.csv` | 六材料事件行数检查 |
| `undergrad_validation/train_test_split_samples.csv` | 训练/测试拆分证据 |
| `undergrad_validation/absorption_group_classification_summary.csv` | 分类方法和 accuracy 汇总 |
| `undergrad_validation/absorption_group_confusion_threshold.csv` | 阈值法混淆矩阵 |
| `undergrad_validation/absorption_group_confusion_logistic_1f.csv` | 单特征 Logistic Regression 混淆矩阵 |
| `undergrad_validation/absorption_group_confusion_logistic_3f.csv` | 三特征 Logistic Regression 混淆矩阵 |
| `directscatter_feature_comparison.csv` | 直接/散射特征对比 |

当前证据包显示：每种材料 5000 个 events、50 个虚拟样本；训练集 150 个样本、测试集 150 个样本。三特征 Logistic Regression 在测试集上正确 149 个样本，accuracy 为 `0.9933`。

## 图表

![系统流程](../figures/elementary_system_flow.png)

![X 射线能谱](../figures/elementary_xray_spectrum.png)

![直接与散射命中比例](../figures/elementary_direct_scatter_ratio.png)

![基础分类精度](../figures/elementary_absorption_accuracy.png)

## 结果边界

本报告支持“Geant4 XRT 仿真原型系统与基础分类验证”结论。当前结果只对应公开仓库内的六材料仿真任务和粗粒度分类目标，不代表所有材料、所有设备条件或所有现场流程。
