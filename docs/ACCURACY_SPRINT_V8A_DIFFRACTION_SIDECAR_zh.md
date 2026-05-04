# Accuracy Sprint v8A：diffraction-aware sidecar Pilot 预注册

日期：2026-05-04

## 1. 科学定位

v8A 由 H/M information-boundary 证据包触发。v7B 已经证明当前系统具备有用的十材料粗分类能力，但 v7B2 formal Pilot 和 Phase 1 compact sidecar 共同说明，ordinary attenuation-derived XRT features 仍然不能稳定区分 Hematite/Magnetite。下一步因此不是 GPU/deep model 搜索，也不是继续增加 ordinary XRT view，而是测试显式加入 crystal-structure-aware diffraction sidecar 后，是否出现物理上有意义的 H/M 信号。

本阶段仍然是 development-only。不得读取、写入、生成或评估 shadow/final 数据。在 tiny synthetic observability gate 通过前，不启动大型 Geant4 batch。

## 2. 最小 v8A sidecar 概念

v8A sidecar 是 XRD-like，而不是 ordinary XRT-like。目标信号是 Hematite 和 Magnetite 的 phase-specific powder peaks 与 peak ratios。第一个 prototype 使用项目文献扫描得到的 tabulated 2theta peak anchors：

- Hematite: `24.1`, `33.2`, `35.6`, `40.9`, `49.5`, `54.1`, `57.5`, `62.5`, `64.0` degrees.
- Magnetite: `18.3`, `30.1`, `35.5`, `37.0`, `43.1`, `53.4`, `57.0`, `62.6`, `74.0` degrees.

prototype 会故意包含 `35.5/35.6`、`57.0/57.5`、`62.5/62.6` 等 overlapping regions。只有当 non-overlapping peak families 和 peak-ratio features 在 angle jitter、peak broadening、background slope、thickness attenuation、count noise、grain-orientation intensity perturbation 后仍可分时，才允许通过。

主模型特征集必须排除这些 overlap windows。overlap regions 只作为 guardrail model 保留；如果 overlap-only features 能区分 H/M，也不能算通过依据，因为现实中的 peak broadening 与 calibration drift 很容易吞掉这种细微信号。

这个 tiny prototype 假设固定 powder-pattern 2theta axis。若后续 v8A 引入多波长或多 source energies，特征轴必须先转换为 `q` 或 d-spacing，再做 source fusion；不能跨波长复用固定 2theta bins。

## 3. Prototype implementation

Script:

- `analysis/v8a_diffraction_observability.py`

Default output directory:

- `results/accuracy_v3/v8a_diffraction_sidecar_pilot/`

Expected outputs:

- `v8a_synthetic_powder_features.csv`
- `v8a_observability_metrics.csv`
- `v8a_model_selection.csv`
- `v8a_validation_decisions.csv`
- `v8a_synthetic_manifest.json`
- `v8a_gate.json`
- `v8a_observability_report.md`

Default command:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_diffraction_observability.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426
```

Recommended stress command:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_diffraction_observability.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426 \
  --output-dir results/accuracy_v3/v8a_diffraction_sidecar_pilot_stress \
  --peak-width-deg 0.35 \
  --angle-jitter-deg 0.12 \
  --orientation-sigma 0.80 \
  --background-level 0.15 \
  --background-slope-sigma 0.35 \
  --counts-scale 700 \
  --read-noise-sigma 0.010 \
  --attenuation-strength 0.025
```

脚本只能使用 prototype 内部 hardcoded powder-peak table。不得读取 v7B/v7B2 cubes、ordinary XRT result artifacts、shadow data 或 final data。输出目录默认必须为空；只有显式传入 `--overwrite` 时才允许替换已有 prototype artifacts，防止旧结果被静默复用。

## 4. Gate

synthetic observability gate 只有在全部检查通过时才通过：

- pre-ML physical observability：最佳单一 peak/ratio feature 的 oriented AUC `>=0.95` 或 absolute `d-prime >=3.0`；
- small development classifier：H/M min recall `>=0.80`；
- thickness-blind/worst-thickness robustness：H/M min recall `>=0.78`；
- control guard：只使用 thickness、total intensity、background、scale-like features 的 control-only model，H/M min recall 必须 `<0.75`；
- shuffled-label guard：同一 peak-shape model 只打乱 training labels 后，H/M min recall 必须 `<0.65`；
- overlap-only guard：只使用 `35.5/35.6`、`57.0/57.5`、`62.5/62.6` 重叠区域的模型，H/M min recall 必须 `<0.75`；
- manifest 必须确认 `development_only=true`、`shadow_or_final_used=false`、`reads_existing_xrt_cubes=false`。

Decision rules：

- 全部通过：`proceed_to_v8a_transport_preregistration`。
- H/M min recall `<0.75`：`no_go_refine_or_stop_diffraction_sidecar`。
- 其他情况：`gray_zone_strengthen_perturbations_before_transport`。

## 5. Claim discipline

v8A synthetic gate 通过不等于 publishable XRD result，不等于 Geant4 transport result，也不等于 hardware validation。它只能说明：在预注册扰动和 sanity guards 下，tabulated powder-peak sidecar 具备 synthetic H/M observability。通过后的下一步是预注册 Geant4 transport/detector integration design，并显式定义 custom diffraction generator/process 或 tabulated powder-pattern sidecar。

不得把 standard Geant4 Rayleigh scattering 写成 powder XRD。不得把 `G4XrayReflection` 写成 bulk H/M diffraction。不得在 v8A transport preregistration 明确定义 geometry、angular bins、background、detector resolution、throughput 和 no-go criteria 之前扩展 hard-negative material matrix。

## 6. Environment

固定使用项目 Python 环境：

- `/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python`

除非遇到真实 blocker 并先获得用户确认，不切换 Python 环境。
