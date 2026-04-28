# 十材料分辨 v2 协议：物化指纹与字典底座

这份说明记录下一阶段十材料分辨的执行协议。它不是把旧的 `0.9960` 二分类结果包装成十材料结果，而是把材料级任务升级为可校准、可分批运行、可复核的实验链路。

## 1. 目标和验收

主目标是 seed 留出测试：`seed=101` 用于开发训练，`seed=202` 用于验证和模型选择，`seed=303` 作为最终测试。最终测试只能在模型、阈值和字典原型冻结后运行一次。

| 指标 | 门槛 |
| --- | ---: |
| Top-1 accuracy | >= 0.85 |
| macro-F1 | >= 0.80 |
| 最低单类召回 | >= 0.70 |

如果这些条件不满足，不能声称“十材料完全分清”。如果 Top-3 很高但 Top-1 未达标，应表述为候选检索和复核系统。

## 2. 数据矩阵

`analysis/generate_material_sorting_matrix.py` 生成两类 run：

| run 类型 | 数量 | 作用 |
| --- | ---: | --- |
| material | 270 | `10材料 x 3厚度 x 3源 x 3 seeds` |
| calibration | 9 | `3源 x 3 seeds` 的 air-path 空束校准 |

材料 run 用于监督训练和测试；calibration run 只用于计算 `I0_bin`，不能进入十材料训练集。

```bash
python analysis/generate_material_sorting_matrix.py --profile pilot
python analysis/generate_material_sorting_matrix.py --profile full
```

## 3. 分批运行

```bash
python analysis/run_material_sorting_matrix.py --profile full --role calibration
python analysis/run_material_sorting_matrix.py --profile full --role material --start 0 --limit 30
python analysis/run_material_sorting_matrix.py --profile full --role material --status-only
```

runner 支持 `--start`、`--limit`、`--role`、`--rerun-existing` 和 `--status-only`，用于分批恢复 full matrix。

## 4. v2 分析

完整 full 矩阵跑完后运行：

```bash
python analysis/material_sorting_v2.py
```

脚本默认读取 `build/material_sorting_runs/full`，输出到 `results/material_sorting_v2/`。如果 full 矩阵不完整，脚本默认只写 incomplete manifest 并停止，不会回退旧数据，也不会输出十材料成功指标。

调试 smoke 可显式使用：

```bash
python analysis/material_sorting_v2.py \
  --raw-dir build/material_sorting_runs/pilot \
  --output-dir build/material_sorting_v2_smoke \
  --allow-incomplete
```

## 5. 物化指纹和模型族

v2 固定能量 bin：

`[0,40,50,60,70,80,90,100,110,120,inf] keV`

主要特征按 family 标注：raw counts、calibrated transmission、attenuation、thickness-normalized attenuation、spectral shape、scatter/direct、source-fusion、dictionary-distance。具体包括 `I_bin`、`T_bin=(I+0.5)/(I0+0.5)`、`A_bin=-log(T_bin)`、`A_bin/thickness`、低/高能透射 ratio、谱质心、谱标准差、稳定 direct/scatter 比值、字典最近距离、Top-k 距离和 margin。

原来的不稳定比值 `scattered/(direct+1e-6)` 不再作为 v2 特征使用。

模型选择只使用 `seed=202` validation。候选包括 `PhysicsOnly`、`DictionaryOnly`、`PhysicsPlusDictionary`、centroid baseline、Logistic/SVM/RandomForest/ExtraTrees/HistGB/MLP。最终 `seed=303` test 只在 validation 选定模型和 review 阈值后评估。

## 6. 防泄漏

以下信息不得作为模型特征：材料名、化学式、密度、吸收组标签、run id、config path、output prefix、sample id、random seed 和 split 标记。`run_role=calibration` 只能用于过滤校准行。

v2 主要输出：

- `material_raw_inventory_v2.csv`
- `material_feature_columns_v2.csv`
- `material_feature_families_v2.csv`
- `material_excluded_columns_v2.csv`
- `material_seed_split_assignments_v2.csv`
- `material_leakage_report_v2.json`
- `model_selection_validation.csv`
- `feature_family_ablation.csv`
- `threshold_selection_validation.csv`
- `validation_decisions.csv`
- `candidate_retrieval_validation.csv`
- `material_confusion_graph.csv`
- `per_class_recall_validation.csv`
- `material_dictionary.json`
- `material_dictionary.csv`
- `material_dictionary_enriched.json`
- `material_dictionary_enriched.csv`
- `final_test_summary.csv`
- `final_test_decisions.csv`
- `candidate_retrieval_final_test.csv`
- `per_class_recall_final_test.csv`
- `material_sorting_v2_manifest.json`

## 7. 当前状态

本仓库已经完成 full matrix：`270` 个 material run 和 `9` 个 calibration run 全部 returncode `0`。正式 v2 评估结果已经写入 `results/material_sorting_v2/`。

当前结论是负结果诊断，不是十材料自动分选成功：

| 项 | 数值 |
| --- | ---: |
| validation 选中模型 | `LogisticRegression` |
| validation Top-1 / Top-3 | `0.5187` / `0.9233` |
| final test Top-1 / Top-3 | `0.3813` / `0.8117` |
| final test macro-F1 | `0.3153` |
| final test 最低单类召回 | `0.0000` |
| final stage conclusion | `diagnostic_only_not_ready` |

validation 上选出的 review 阈值为 probability `0.75`、margin `0.0`，validation auto-sort precision `0.9049`，但 final test auto-sort precision 只有 `0.5647`、review rate `0.6057`。这说明当前 v2 特征和模型在 seed holdout 下泛化不足，不能包装成成功结论。

主要混淆来自低吸收硅酸盐/碳酸盐内部（如 Quartz-Albite-Orthoclase-Dolomite）和高吸收硫化物/氧化物内部（如 Hematite-Magnetite-Chalcopyrite-Pyrite）。旧的 `0.9960` 仍只能代表低/高吸收组二分类，不能外推到十材料分辨。

## 8. v2 后续严格泛化复核

v2 失败后，本仓库继续完成了 selected rebuild、GPU 候选模型和独立新 seed 复核。新增脚本：

- `analysis/material_sorting_energy_scan.py`：开发集能量组合筛选，可限制候选 source subset，避免无意义全组合搜索。
- `analysis/material_sorting_selected_rebuild.py`：selected rebuild 多 seed 评估，包含 GPU XGBoost 候选和层级 ExtraTrees。
- `analysis/strict_generalization_audit.py`：严格 train/validation/test seed 复核，支持合并多个 raw profile，并标记已烧掉 test seed。

关键结果：

| 阶段 | 结果 |
| --- | --- |
| selected rebuild p5000旧测试 | Top-1 `0.9000`，macro-F1 `0.8993`，min recall `0.5000`，每类 support `6`，不 claim-safe |
| `sr2` locked unseen final test | Top-1 `0.8800`，macro-F1 `0.8789`，min recall `0.5000`，每类 support `30`，失败 |
| `es2` 扩展能量扫描三能量 | 推荐 `40/110/200 keV`，validation min recall `0.6000` |
| `es2` 候选四能量 | 推荐 `40/110/120/200 keV`，validation min recall `0.6000` |

`sr2` 的 final test seeds 为 `707/808/909/1001/1102`，不与旧 v2 test `303` 或 selected rebuild test `505` 重复。它仍未达标，主要因为 Hematite recall `0.5000`、Magnetite recall `0.6333`。

因此当前项目结论应维持为：可复现候选检索和诊断链路已经增强，但十材料自动分选没有通过严格泛化验收。最新复核见 `docs/STRICT_GENERALIZATION_REVIEW_zh.md`。
