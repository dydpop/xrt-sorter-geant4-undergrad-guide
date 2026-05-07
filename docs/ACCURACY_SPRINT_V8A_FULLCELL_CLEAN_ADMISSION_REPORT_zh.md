# v8A full-cell clean admission report

Generated: 2026-05-07

## 一句话结论

这一次终于不是“看起来干净”，而是 clean admission gate 真的通过了。
当前可用数据是 `v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature`：
它只解锁 development-only 模型训练诊断，不解锁 shadow/final、不解锁 full ten-material matrix、不解锁产品准确率或硬件验证 claim。

## 这轮修掉了什么

上一轮最大的结构问题是：每个最细 nuisance cell
`seed_block × thickness × pose × count_bin` 里只有 1 对 H/M。这样 null 标签虽然在较粗层级看似平衡，
但在最细层级不可能做到一半翻转、一半不翻转，模型仍可能抓到打乱协议本身的方向。

本轮从源头重采样，而不是继续从旧数据硬洗：

- 新 profile：`v8a_hm_clean_fullcell_nullrep_cif_lit`
- rows：`2016`
- Geant4：`2016/2016` completed，`0` failed
- strict pairs：train `432`，validation `288`，stress-holdout `288`
- 每个最细 nuisance cell 有 `2` 对 H/M，支持 strict full-cell null orientation
- 主可训练候选关闭 residualization，只保留 train-fit source z-score 后的 diffraction features

## 门禁结果

Preflight 通过：

- `gate_passed=true`
- rows `2016`
- strict pairs：train `432`，validation `288`，stress-holdout `288`

Event-to-feature 通过：

- samples `2016`
- sidecar long rows `358736`
- schema gate `true`

Non-material shortcut gate 通过：

- max non-material balanced accuracy `0.3681`
- decision `feature_shortcut_structure_clean`

Strict full-cell paired-null gate 通过：

- primary mode `seed_block_thickness_pose_count_strict_balanced_orientation`
- effective shuffle fraction min/max `0.5 / 0.5`
- full-cell orientation max abs sum `0.0`
- fixed H/M min recall p95 `0.5248`
- fixed H/M min recall max `0.5451`

Threshold-free null gate 通过：

- primary oriented AUC p95 `0.5540`
- primary oriented AUC max `0.6090`
- rank-overlap p05 `0.8920`
- threshold inflation p95 `0.0`

Admission 通过：

- decision `crystal_clean_view_training_diagnostics_unlocked`
- `training_unlocked=true`

## 学到的方法学规则

这轮按以下同类数据处理原则重做：

- 数据泄漏默认存在，直到被审计打掉；参考 Kaufman/Rosset/Perlich 的 leakage taxonomy。
- 所有 scaling/residualization/feature choice 都必须 train-only fit，再 apply 到 validation/holdout；参考 scikit-learn data leakage guidance。
- null/permutation test 必须保持 group/paired 结构，而不是简单 row shuffle；参考 Ojala & Garriga permutation-test framework。
- grouped/structured split 要按最强依赖单元隔离；参考 Roberts 等 structured cross-validation。
- 光谱/衍射预处理本身就是模型选择，不能在全数据上先做；参考 Rinnan 等 NIR preprocessing review。
- XRD 深度模型可以用作后续 feature sufficiency probe，但只能在 null/admission 通过后使用；参考 Oviedo 等 XRD augmentation/CNN validation work。

## 仍然不能说什么

不能说：

- ordinary XRT 已经解决 H/M
- 产品准确率已经成立
- 硬件验证已经成立
- shadow/final 已通过
- full ten-material v8A matrix 可以开跑
- manuscript-grade powder XRD claim 已成立

可以说：

- 在 development-only、CIF/literature peak-table、source-on/default、strict full-cell balanced clean design 下，
  source-scaled no-residualization diffraction sidecar view 已经通过 clean admission，
  可以进入下一阶段 development-only 模型训练诊断。

## 下一步

下一步可以开始 development-only 训练诊断，但必须按顺序：

1. Logistic / ExtraTrees baseline。
2. calibration、threshold sweep、by-thickness、by-pose、by-seed-block recall。
3. total-count-only、source/origin-only、shuffled-label controls 继续保留。
4. advanced MLP/FT-Transformer/1D CNN 只能作为 feature sufficiency probe，必须同步跑 null controls。
5. 任一 null/control 反弹，立刻回到数据/表示，不升级 shadow/final。

## Key artifacts

- Matrix config: `analysis/configs/v8a_clean_hm_fullcell_nullrep_matrix_config.json`
- Parallel runner: `analysis/run_material_sorting_matrix_parallel.py`
- Rework view builder: `analysis/build_v8a_residualization_protocol_rework_v1_views.py`
- Full-cell source features: `results/accuracy_v3/v8a_clean_hm_fullcell_nullrep_event_to_feature/`
- Training-admitted view: `results/accuracy_v3/v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature/`
- Admission gate: `results/accuracy_v3/v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_admission/v8a_crystal_clean_admission_gate.json`

## References

- Kaufman, Rosset & Perlich, leakage in data mining: https://doi.org/10.1145/2382577.2382579
- scikit-learn common pitfalls/data leakage: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- Ojala & Garriga, classifier permutation tests: https://jmlr.org/papers/v11/ojala10a.html
- Varma & Simon, cross-validation/model-selection bias: https://doi.org/10.1186/1471-2105-7-91
- Rinnan, van den Berg & Engelsen, spectral preprocessing review: https://doi.org/10.1016/j.trac.2009.07.007
- Oviedo et al., small XRD data augmentation/deep learning: https://doi.org/10.1038/s41524-019-0196-x
