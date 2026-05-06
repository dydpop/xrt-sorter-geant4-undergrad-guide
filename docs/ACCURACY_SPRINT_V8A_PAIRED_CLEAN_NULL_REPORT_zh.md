# v8A Paired-Clean Null Report

## 一句话结论

本阶段把旧的 row/pair shuffle null 改成了适合 strict paired clean design 的 paired-clean null：在 `clean_match_pair_id` 层生成假标签，并在 train seed block、pose、count-bin 等 nuisance 维度上强制平衡。

这一步的目的不是训练，而是确认“假标签是否真的学不起来”。只有 paired-clean null 和 shortcut gate 同时通过，admission 才能解锁下一步 development-only baseline。

## 新 Null Protocol

新增脚本：

- `analysis/diagnose_v8a_paired_clean_null_behavior.py`

输入：

- `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_event_to_feature/`

输出：

- `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_paired_null/`

核心设计：

- 只使用 development-only clean-design-cell view。
- 不读 shadow/final。
- 不读 existing XRT cube。
- 不训练真实标签模型。
- 主特征仍只来自 `diffraction_*`。
- 在 `clean_match_pair_id` 内生成 pseudo label orientation。
- primary null mode 是 `paired_nuisance_balanced_orientation`。
- 每个 train seed block 内 orientation 严格 50/50。
- pose 与 count-bin 方向也严格平衡。
- thickness 因为每个 seed block 内每档有 9 对，只能做到最小不平衡。

准入阈值：

- shuffle seeds `>=60`
- effective shuffle fraction 在 `0.45-0.55`
- primary fixed/selected p95 `<=0.55`
- primary fixed/selected single-seed max `<=0.65`

## Admission 规则更新

`analysis/audit_v8a_crystal_clean_admission.py` 现在可以识别：

- 旧 null gate：`v8A_shuffled_label_null_behavior_diagnosis`
- 新 paired-clean null gate：`v8A_paired_clean_null_behavior_diagnosis`

如果传入 paired-clean null gate，admission 使用 paired null 的 p95 与 single-seed max，而不是旧 row-level max 字段。

本轮同时修正了 admission JSON 的字段语义：

- `fixed_threshold_null_hm_gate_value` / `selected_threshold_null_hm_gate_value` 表示 admission 实际使用的门控值；
- paired-clean null 下，这个门控值是 p95；
- `fixed_threshold_null_hm_max` / `selected_threshold_null_hm_max` 始终表示真正的 single-seed max。

这样后续报告不会把 p95 误读成 max，也不会把 max 误读成准入主统计量。

## 本轮运行结果

本轮 paired-clean null 已经真正运行，但没有通过准入：

- paired null gate：`gate_passed=false`
- admission：`training_unlocked=false`
- primary effective shuffle fraction：`0.50-0.50`
- train seed-block orientation max abs sum：`0.0`
- primary fixed-threshold p95：`0.5972`
- primary selected-threshold p95：`0.5972`
- primary fixed/selected single-seed max：`0.6250`
- all-mode fixed/selected p95：`0.5972`
- all-mode fixed/selected single-seed max：`0.6250`
- paired-null stop reasons：`fixed_threshold_null_p95_exceeded_ceiling`，`selected_threshold_null_p95_exceeded_ceiling`，`all_modes_fixed_threshold_null_p95_exceeded_ceiling`，`all_modes_selected_threshold_null_p95_exceeded_ceiling`
- admission stop reasons：`null_gate_failed`，`fixed_threshold_null_p95_exceeded_ceiling`，`selected_threshold_null_p95_exceeded_ceiling`

解释：这次不再是明显的“没有打乱干净”。effective shuffle fraction 正好是 0.5，seed-block 方向也平衡；但 60 个 pseudo-label seed 的尾部仍然偏高，p95 超过 `0.55` ceiling。单次最大值 `0.6250` 没有超过 `0.65` ceiling，说明问题比旧 null gate 小很多，但还没小到可以解锁训练。

gate 也已经从“只看 primary null mode”加严为“primary 与所有 paired-clean null modes 都必须低于阈值”。当前 secondary mode 不是旁路，后续不能出现 primary 过了但 secondary 暴露问题却照样解锁训练的情况。

因此当前结论是：旧 null protocol 的确放大了问题，但 paired-clean null 仍未把假标签风险压到可接受范围。不能启动 development-only baseline training，更不能启动高级模型、shadow/final 或大矩阵。

## Claim Boundary

本阶段即使通过，也只说明假标签审计方式更适合 clean paired data。它最多解锁 development-only baseline diagnostics，不解锁 shadow/final、不解锁 full ten-material、不构成 product accuracy、hardware validation 或 manuscript-grade powder XRD claim。

## 大白话

旧的假标签测试像是把一张成对试卷随便打乱，结果有些格子里其实没有真正乱开，或者偶然把半张卷子打成了一个方向，模型就可能被这种随机偏差带偏。

新的测试换成“按对子打乱，而且每个 seed block、姿态、count-bin 里正反方向都尽量平衡”。如果这样假标签还学得起来，那就是真有更深问题；如果学不起来，说明上一轮主要是 null 方法和成对数据结构不匹配。
