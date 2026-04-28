# 十材料分选严格泛化复核

本文记录 v2 失败后的重做、GPU 候选模型、独立新 seed 测试和扩展能量扫描。结论边界很重要：当前仓库仍不能声称十材料自动分选成功。

## 验收标准

自动十材料分选必须同时满足：

| 指标 | 门槛 |
| --- | ---: |
| Top-1 accuracy | >= 0.85 |
| macro-F1 | >= 0.80 |
| 最低单类召回 | >= 0.70 |
| 每类 final test support | >= 30 |

模型、源组合、photon budget 和阈值必须只用训练/验证 seed 决定。final test seed 打开后即视为烧掉，后续只能作为开发诊断，不能继续当作成功测试。

## 已完成阶段

| 阶段 | 数据/协议 | 结论 |
| --- | --- | --- |
| v2 full matrix | `270` material + `9` calibration，train `101` / validation `202` / test `303` | Top-1 `0.3813`，macro-F1 `0.3153`，min recall `0.0`，失败 |
| selected rebuild 旧测试 | `40/50/120 keV`，train `101/202/303`，validation `404`，test `505` | p5000 Top-1 `0.9`，macro-F1 `0.8993`，但 min recall `0.5`，且每类 support `6`，不能声明成功 |
| strict generalization `sr2` | 旧 seed 作为开发训练池，新 validation `606`，新 final test `707/808/909/1001/1102` | Top-1 `0.88`，macro-F1 `0.8789`，每类 support `30`，但 min recall `0.5`，失败 |
| post-hoc p10000 | 同一 `sr2` test，p10000 诊断 | Top-1 `0.90`，macro-F1 `0.8996`，min recall `0.4667`，更高单样本光子数没有解决 Hematite/Magnetite |
| `es2` 扩展能量扫描 | `30/40/50/60/70/80/90/100/110/120/150/200 keV`，只用 101/202 开发筛源 | 最好三能量和四能量 validation min recall 都只有 `0.6`，不足以进入新的成功声明轮 |

## 关键证据

- `results/material_sorting_strict_generalization_sr2_locked/strict_generalization_manifest.json`
- `results/material_sorting_strict_generalization_sr2_locked/final_test_summary.csv`
- `results/material_sorting_strict_generalization_sr2_locked/per_class_recall_final_test.csv`
- `results/material_sorting_energy_scan_es2/material_sorting_energy_scan_manifest.json`
- `results/material_sorting_energy_scan_es2_candidates/material_sorting_energy_scan_manifest.json`

`sr2` locked final test 的最低召回来自：

| 材料 | support | recall |
| --- | ---: | ---: |
| Hematite | 30 | 0.5000 |
| Magnetite | 30 | 0.6333 |

这不是单次小样本偶然失败。此前 v2 separability 诊断已显示 Hematite/Magnetite 的最近邻 separability ratio 长期远低于 `1.0`，说明在当前几何、源项、标量/表格指纹和 photon budget 下，类内噪声半径远大于两类中心距离。

## 编辑/导师判断

作为论文或答辩材料，应表述为：

> 当前系统已经具备可复现的 Geant4 XRT 仿真、校准、多 seed 留出、GPU 候选模型和 Top-K 候选检索证据链；但在严格独立 seed final test 上，十材料自动 Top-1 分选仍未通过最低单类召回门槛。当前最稳妥的产品形态是“吸收组分选 + Top-3 材料候选 + 复核”，不是全自动十材料识别。

不能表述为：

> 十种矿物已经被模型可靠分清。

## 下一步边界

如果继续攻关，不建议再在当前 `40/50/120` 或 `40/110/120/200` 表格特征上盲目调参。应优先改变输入物理或任务定义：

- 增加真正能区分 Fe 氧化物的测量维度，例如更明确的能谱分辨、K-edge 附近信息、双角度散射或更真实的探测器响应。
- 把自动分选目标降级为高置信自动分选 + H/M 等硬对复核。
- 对 Hematite/Magnetite 单独做可分性上界实验，再决定是否值得继续生成大规模矩阵。

本轮没有把任何已烧掉 test 结果包装成成功结论。
