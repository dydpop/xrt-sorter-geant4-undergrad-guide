# G4 Accuracy Sprint v6 执行说明

## 目标

v6 的目标是提高严格泛化准确率，同时避免 validation 反复刷榜、shadow 被调参污染、final 被提前打开。

新的 final 口径：

| 指标 | 门槛 |
| --- | --- |
| Top-1 | `>=0.88` |
| macro-F1 | `>=0.84` |
| min class recall | `>=0.75` |
| per-class final support | `>=40` |

H/M 阶段额外要求：

| 指标 | 门槛 |
| --- | --- |
| H/M min recall | `>=0.80` |
| H/M pairwise min recall | `>=0.75` |
| overall min recall | `>=0.75` |

Burned seeds `303/505/707/808/909/1001/1102` 禁止调参使用。

## 阶段 1：跑完 v5_hm_lowwide

先跑 calibration，再跑 material。runner 会 checkpoint/dedupe，重复运行不会重跑已成功的行。

```bash
python3 analysis/run_material_sorting_matrix.py --profile v5_hm_lowwide --role calibration
python3 analysis/run_material_sorting_matrix.py --profile v5_hm_lowwide --role material
python3 analysis/run_material_sorting_matrix.py --profile v5_hm_lowwide --status-only
```

进入 audit 前必须看到：

```text
selected_rows=6760 completed=6760 failed=0 pending=0
```

## 阶段 2：H/M development gate

使用 bundled Windows Python 运行 audit，因为 WSL Python 缺少 pandas/sklearn/xgboost。

```powershell
$repo='\\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
$py='C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py "$repo\analysis\strict_generalization_audit.py" `
  --project-root $repo `
  --raw-dir 'build/material_sorting_runs/v5_hm_lowwide' `
  --output-dir 'results/accuracy_v3/v5_hm_lowwide' `
  --photon-budget 5000 `
  --train-seeds '1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520' `
  --validation-seeds '1601,1602,1603,1604,1605,1606,1607,1608,1609,1610' `
  --burned-test-seeds '303,505,707,808,909,1001,1102' `
  --development-only `
  --protocol-name 'v6_hm_lowwide_development_gate'
```

Gate 检查：

```powershell
& $py "$repo\analysis\accuracy_v6_gate.py" `
  --project-root $repo `
  --audit-dir 'results/accuracy_v3/v5_hm_lowwide' `
  --stage hm_development `
  --status-profile v5_hm_lowwide `
  --output-json 'results/accuracy_v3/v5_hm_lowwide/v6_gate_report.json'
```

若 gate 失败，停止并记录失败，不进入 GPU 长训练。

## 阶段 3：GPU 长训练

只有 H/M development gate 通过后，才运行预注册 XGBoost GPU 网格。

```powershell
& $py "$repo\analysis\xgboost_gpu_grid_v6.py" `
  --project-root $repo `
  --raw-dir 'build/material_sorting_runs/v5_hm_lowwide' `
  --output-dir 'results/accuracy_v3/v6_gpu_search'
```

该脚本固定使用：

- train `1501-1516`
- inner early-stop `1517-1520`
- external validation `1601-1610`
- `max_depth=[3,4,5]`
- `learning_rate=[0.015,0.03,0.06]`
- `subsample=[0.85,1.0]`
- `colsample_bytree=[0.75,0.9]`
- `reg_lambda=[1.0,3.0]`
- `n_estimators=5000`
- `early_stopping_rounds=100`
- `device=cuda`

输出：

- `gpu_grid_search_candidates.csv`
- `gpu_grid_selected_summary.csv`
- `gpu_grid_manifest.json`
- `validation_decisions.csv`
- `failure_analysis.csv`
- `hm_pairwise_audit.csv`

## 阶段 4：Shadow gate

Shadow seeds `1701-1710` 只能用于一次确认。若失败，不允许反复用 shadow 调参。

```powershell
& $py "$repo\analysis\strict_generalization_audit.py" `
  --project-root $repo `
  --raw-dir 'build/material_sorting_runs/v5_hm_lowwide' `
  --output-dir 'results/accuracy_v3/v5_hm_shadow' `
  --photon-budget 5000 `
  --train-seeds '1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520,1601,1602,1603,1604,1605,1606,1607,1608,1609,1610' `
  --validation-seeds '1701,1702,1703,1704,1705,1706,1707,1708,1709,1710' `
  --burned-test-seeds '303,505,707,808,909,1001,1102' `
  --development-only `
  --protocol-name 'v6_hm_shadow_gate'
```

```powershell
& $py "$repo\analysis\accuracy_v6_gate.py" `
  --project-root $repo `
  --audit-dir 'results/accuracy_v3/v5_hm_shadow' `
  --stage hm_shadow `
  --output-json 'results/accuracy_v3/v5_hm_shadow/v6_gate_report.json'
```

只有 development 与 shadow 都通过，才允许生成十材料 `v5_full_trainval`。`v5_full_final_locked` 仍不得打开。
