# 本科项目演示报告

## 项目目标

本项目目标是构建一个基于 Geant4 的 XRT 矿物分选仿真原型，并完成从仿真输出到基础分类结果的闭环展示。

## 已完成链路

```mermaid
flowchart LR
    A["X 射线源配置"] --> B["Geant4 仿真"]
    B --> C["探测器事件数据"]
    C --> D["Python 特征处理"]
    D --> E["粗粒度吸收组分类"]
    E --> F["结果表和图表"]
```

## 关键结果

| 文件 | 说明 |
| --- | --- |
| `absorption_group_classification_summary.csv` | 分类方法和 accuracy 汇总 |
| `absorption_group_confusion_threshold.csv` | 阈值法混淆矩阵 |
| `absorption_group_confusion_logistic_1f.csv` | 单特征 Logistic Regression 混淆矩阵 |
| `absorption_group_confusion_logistic_3f.csv` | 三特征 Logistic Regression 混淆矩阵 |
| `directscatter_feature_comparison.csv` | 直接/散射特征对比 |

当前最重要的结果是：在当前仿真数据和粗粒度吸收组任务中，基础分类最高 accuracy 为 `0.98`。

## 图表

![系统流程](../figures/elementary_system_flow.png)

![X 射线能谱](../figures/elementary_xray_spectrum.png)

![直接与散射命中比例](../figures/elementary_direct_scatter_ratio.png)

![基础分类精度](../figures/elementary_absorption_accuracy.png)

## 结果边界

本报告支持“Geant4 XRT 仿真原型系统与基础分类验证”结论。当前结果只对应公开仓库内的仿真任务和粗粒度分类目标，不代表所有材料、所有设备条件或所有现场流程。
