# 本科十材料分选重做诊断

本报告由 `analysis/material_sorting_rebuild_diagnostics.py` 生成，用于解释 v2 full matrix 失败后应先改输入协议还是先升级模型。

## 结论

- 当前最好 photon budget 为 `1000` photons/sample，final Top-1 为 `0.8033`，macro-F1 为 `0.7962`，min recall 为 `0.3000`。
- 这些结果仍必须按十材料负结果处理，除非同时达到 Top-1 >= 0.85、macro-F1 >= 0.80、min recall >= 0.70。
- 如果 photon budget 提升不能显著改善结果，下一步应优先做 mono energy scan 和多能输入筛选，而不是直接把当前标量表交给 CNN/Transformer。

## 最难分材料

- `Magnetite` 最近邻为 `Hematite`，separability ratio `0.0134`。
- `Hematite` 最近邻为 `Magnetite`，separability ratio `0.0134`。
- `Magnetite` 最近邻为 `Hematite`，separability ratio `0.0185`。
- `Hematite` 最近邻为 `Magnetite`，separability ratio `0.0186`。
- `Magnetite` 最近邻为 `Hematite`，separability ratio `0.0288`。
- `Hematite` 最近邻为 `Magnetite`，separability ratio `0.0290`。

## 当前源组合筛选

- `mono_60kev;mono_100kev;spectrum_120kv`: validation Top-1 `0.7100`, macro-F1 `0.7078`。
- `mono_100kev;spectrum_120kv`: validation Top-1 `0.6767`, macro-F1 `0.6728`。
- `mono_60kev;spectrum_120kv`: validation Top-1 `0.6700`, macro-F1 `0.6683`。
- `mono_60kev;mono_100kev`: validation Top-1 `0.6683`, macro-F1 `0.6636`。
- `spectrum_120kv`: validation Top-1 `0.6117`, macro-F1 `0.6087`。

## 输出文件

- `photon_budget_curve.csv`：不同 photon 聚合预算下的 validation 选择和 final test 结果。
- `photon_budget_model_comparison.csv`：各模型在 validation seed 上的对比。
- `per_material_separability.csv`：每种材料的最近邻、类内半径和 separability ratio。
- `seed_variance.csv`：leave-one-seed-out 稳定性诊断。
- `source_pair_screening.csv`：现有 60 keV、100 keV、120 kV spectrum 的单源/双源/三源筛选。
- `confusion_pair_distance.csv`：final test 误分对、score margin 和开发集 centroid 距离。

本报告不包含 V3 或人工复核高级线；那些内容应留到导师/高级仓库。

Manifest: `analysis/material_sorting_rebuild_diagnostics.py` at `2026-04-27T15:54:37+00:00`.
