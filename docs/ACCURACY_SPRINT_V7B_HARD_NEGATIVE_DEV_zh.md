# Accuracy Sprint v7B: hard-negative Geant4 development preregistration

Date: 2026-04-30

## 1. Scientific position

v7B is triggered by the v7A failure. v7A proved that reusing the v6c raw observations as a measurement cube is not enough for stable Hematite/Magnetite separation. v7B therefore changes the physical sampling matrix before any further claim-facing evaluation.

This is still a development phase. It may select models and tune only against v7B validation. It must not use shadow or final seeds.

## 2. Locked matrix

Profile: `v7b_hard_negative_dev`

- materials: `Quartz, Calcite, Orthoclase, Albite, Dolomite, Pyrite, Hematite, Magnetite, Chalcopyrite, Galena`
- thickness: `5, 10, 15, 20, 30 mm`
- energies: `70, 90, 120, 150, 200 keV`
- source variants: `normal_narrow, normal_wide, oblique_10deg, oblique_20deg`
- train seeds: `4101-4112`
- validation seeds: `4201-4206`
- reserved shadow seeds: `4301-4306`, recorded only; not generated or used in v7B development
- events/run: `20000`
- measurement aggregation: `photon_budget=5000`, giving 4 complete samples/run

Expected full development rows:

- material rows: `10 x 5 x 20 x 18 = 18000`
- calibration rows: `20 x 18 = 360`
- total rows: `18360`

## 3. New code surfaces

- `analysis/generate_material_sorting_matrix.py`
  - adds profile `v7b_hard_negative_dev`
  - adds `oblique_20deg`
  - records source/layout metadata in each generated config
- `analysis/configs/run_material_sorting_v7b.mac`
  - runs `20000` events per row
- `analysis/run_material_sorting_matrix.py`
  - maps `v7b_*` profiles to the v7B macro
- `analysis/export_measurement_cube_v7b.py`
  - exports ten-material measurement cubes
  - writes calibration-normalized channels
  - excludes shadow seeds unless explicitly overridden
- `analysis/train_v7b.py`
  - trains fixed candidate models
  - writes model selection, per-class recall, hard-pair audit, view ablation, failure analysis, and gate JSON

## 4. Model candidates

Fixed v7B development candidates:

- `ExtraTrees`
- `HistGradientBoosting`
- `XGBoost`
- `HardNegativeExtraTrees`
- `HardNegativeXGBoost`
- `GroupExpertExtraTrees`
- `HMPairwiseRerankExtraTrees`

Selection priority is H/M first:

1. H/M min recall
2. H/M pairwise min recall
3. key hard-negative pair min recall
4. ten-material macro-F1
5. ten-material top-1 accuracy
6. min class recall
7. simpler model rank as tie-breaker

## 5. Development gate

The v7B gate passes only if all checks pass:

- ten-material top-1 accuracy `>=0.85`
- ten-material macro-F1 `>=0.82`
- ten-material min class recall `>=0.70`
- H/M min recall `>=0.80`
- H/M pairwise min recall `>=0.78`
- key hard-negative pair min recall `>=0.75`
- validation support/class `>=100`
- runner failures `0`
- runner pending `0`
- shadow/final used `false`

Key hard-negative pairs:

- Hematite/Magnetite
- Pyrite/Chalcopyrite
- Calcite/Dolomite
- Orthoclase/Albite
- Pyrite/Galena
- Chalcopyrite/Galena

## 6. Stop rule

Run at most three development-training rounds on the same v7B validation evidence. If the gate still fails after that, stop tuning this validation set and move to v7B2 physical-matrix redesign.

If ten-material macro-F1 passes but H/M fails, v7B2 should strengthen H/M geometry, oblique/scatter views, and photon budget.

If H/M passes but ten-material macro-F1 fails, v7B2 should focus on hierarchy/group expert design and class balance.

If both fail, the physical matrix is insufficient and should be redesigned rather than model-shopped.

## 7. Commands

Generate smoke matrix:

```bash
python3 analysis/generate_material_sorting_matrix.py \
  --profile v7b_hard_negative_dev \
  --profile-alias v7b_hard_negative_dev_smoke \
  --material-list Hematite,Magnetite \
  --energy-list-kev 70,200 \
  --thickness-list 5 \
  --source-variant-list normal_narrow \
  --seed-list 4101,4201
```

Generate ten-material micro-smoke:

```bash
python3 analysis/generate_material_sorting_matrix.py \
  --profile v7b_hard_negative_dev \
  --profile-alias v7b_hard_negative_dev_ten_material_micro_smoke \
  --energy-list-kev 120 \
  --thickness-list 10 \
  --source-variant-list normal_narrow \
  --seed-list 4101,4201
```

Full matrix generation:

```bash
python3 analysis/generate_material_sorting_matrix.py --profile v7b_hard_negative_dev
```

Runner status:

```bash
python3 analysis/run_material_sorting_matrix.py --profile v7b_hard_negative_dev --status-only
```

Export cube:

```powershell
$py = 'C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\export_measurement_cube_v7b.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v7b_hard_negative_dev `
  --output-dir results/accuracy_v3/v7b_hard_negative_dev
```

Train/gate:

```powershell
$py = 'C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\train_v7b.py" `
  --project-root $root `
  --cube-dir results/accuracy_v3/v7b_hard_negative_dev
```

## 8. Claim discipline

v7B development validation is for model selection only. Even if the v7B gate passes, the result is not final manuscript evidence until shadow/final evaluation is run under the sealed protocol.

## 9. Smoke checkpoint

Completed on 2026-04-30.

H/M pipeline smoke:

- profile alias: `v7b_hard_negative_dev_smoke`
- matrix rows: `12`
- Geant4 status: `completed=12 failed=0 pending=0`
- schema check: metadata records `source_variant`, `detector_layout`, incidence/source/side-detector fields; hits include `detector_id,x_mm,y_mm,z_mm,photon_energy_keV,is_primary,theta_deg,is_direct_primary,is_scattered_primary`
- detectors observed in hit sample: `transmission`, `side_scatter`
- cube shape: `(16, 2, 2, 8, 8, 12)`
- training smoke: `ExtraTrees,HMPairwiseRerankExtraTrees`, one round; gate output written and failed as expected because support/material coverage is smoke-only

Ten-material micro-smoke:

- profile alias: `v7b_hard_negative_dev_ten_material_micro_smoke`
- matrix rows: `22`
- Geant4 status: `completed=22 failed=0 pending=0`
- cube shape: `(80, 1, 2, 8, 8, 12)`
- training smoke: `ExtraTrees,GroupExpertExtraTrees,HMPairwiseRerankExtraTrees`, one round
- selected smoke method: `HMPairwiseRerankExtraTrees`
- observed smoke metrics: top-1 `0.70`, macro-F1 `0.6901`, min class recall `0.25`, H/M min recall `0.50`, validation support/class `4`
- gate failed as expected; this verifies pipeline wiring, not accuracy

Full development matrix:

- profile: `v7b_hard_negative_dev`
- matrix rows: `18360`
- status at generation checkpoint: `completed=0 failed=0 pending=18360`
- full simulation has not been started in this checkpoint
