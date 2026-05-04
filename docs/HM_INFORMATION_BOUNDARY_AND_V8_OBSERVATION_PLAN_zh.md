# H/M 信息边界证据包与 v8 观测方案

日期：2026-05-04

## 1. 编辑层判断

Hematite/Magnetite 的结果现在应该被处理为当前 ordinary-XRT 观测通道的科学边界发现，而不是继续扩大模型搜索的理由。v7B 最强的十材料粗分类 development 模型已经达到 top-1 accuracy `0.8675`、macro-F1 `0.8668`，所以论文不能说 XRT sorter 全局失败。真正的问题更窄，也更有科学价值：Hematite/Magnetite 这组细粒度铁氧化物 hard-negative pair，在 source design、measurement cube、physical-view features 和 compact sidecar 都扩展之后，仍然没有形成稳定分离。

这组结论仍然是 development-only evidence。shadow 和 final 数据保持封存。准确的论文表述应当是：在当前模拟的 ordinary-XRT attenuation、hit-rate、energy-response、angle-response、scatter/direct summary 下，项目尚未得到可靠的 Hematite/Magnetite phase discriminator。这并不证明所有 X-ray 方法都不能区分 H/M；它说明当前 sorter 里的信息通道缺少真实实验室常用的 crystal-structure 或 valence/local-structure 观测。

## 2. 证据包

| 阶段 | 测试的观测/建模变化 | 关键 development 结果 | 决策 |
| --- | --- | --- | --- |
| v6c source design | 高能 source variants 加简化 side-scatter layout | H/M thickness-aware min recall `0.6333`，thickness-blind `0.6500`，pairwise `0.6333` | gate failed；不跑 shadow、GPU grid、ten-material expansion、final |
| v7A measurement cube | 把 v6c 观测升级为 multi-source/multi-detector spatial cube | selected `HardNegativeXGBoost`；H/M min recall `0.5167`；Hematite `0.5667`，Magnetite `0.5167` | gate failed；representation alone 没有救回 H/M |
| v7B hard-negative dev | 新的十材料 hard-negative matrix，20 个 source views，3600 samples | selected `ExtraTrees`；top-1 `0.8675`，macro-F1 `0.8668`，H/M min recall `0.6083`；Hematite `0.6083`，Magnetite `0.6333` | 粗分类有价值，但 H/M 仍是瓶颈 |
| v7B R2/R3 decision | view-focused R2 models、top-k、pair expert diagnostics | selected `R2ExtraTreesTransmissionHighEnergy`；H/M min recall 仍为 `0.6083`；improvement `0.0`；H/M truth in top3 `1.0` | 转向物理观测 redesign，而不是再调模型 |
| v7B2 formal H/M Pilot | 30-source H/M physical matrix，3-40 mm 厚度，oblique 20/30/40 degree views，768 samples | H/M min recall `0.5078`；Hematite `0.5078`，Magnetite `0.6719`；相对 v7B improvement `-0.1005` | `stop_physics_expansion_write_limitation`；不扩 full ten-material v7B2 |
| Phase 1 compact XRT sidecar | 从现有 v7B/v7B2 cube 提取 development-only compact feature table | best compact pair expert `0.6008`；best sidecar policy 是 baseline/no sidecar，H/M min recall `0.6083`；improvement `0.0` | `stop_write_xrt_information_boundary` |

主要结果路径：

- `results/accuracy_v3/v6c_hm_source_design/gate_v6c.json`
- `results/accuracy_v3/v7a_hm_measurement_cube/v7a_gate.json`
- `results/accuracy_v3/v7b_hard_negative_dev/v7b_gate.json`
- `results/accuracy_v3/v7b_hard_negative_dev/v7b_r2_gate.json`
- `results/accuracy_v3/v7b_hard_negative_dev/v7b_hm_decision_report.md`
- `results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_gate.json`
- `results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_limitation_note.md`
- `results/accuracy_v3/hm_phase1_sidecar_dev/hm_phase1_gate.json`
- `results/accuracy_v3/hm_phase1_sidecar_dev/hm_phase1_report.md`

这些文件只能作为内部 development evidence 引用。所有 smoke 和 micro-smoke，包括 v7B2 smoke 中看起来较高的 H/M 分数，都只能证明链路可运行，不能写成 accuracy evidence。source ranking、top-k containment 和 pair-only expert 是诊断工具，不能替代 preregistered gate。

## 3. 论文可用解释

在真实实验室矿物学里，Hematite 和 Magnetite 通常不是靠元素组成单独分开的。二者都是铁氧化物，普通 XRF 主要给元素丰度而不是晶相；XRT 和 dual-energy XRT 主要观察 attenuation、density、effective-Z-like contrast，以及较粗的 scatter/transmission summary。这些通道可以支持有用的粗分选，但它们并不直接编码 H/M 的定义性差异：Hematite 的 corundum/rhombohedral crystal structure 与 Magnetite 的 inverse-spinel/cubic crystal structure，也不直接编码 Fe3+ 与 mixed Fe2+/Fe3+ 的局域电子结构差异。

项目结果与这个物理预期一致。v7B 说明当前系统能学习有价值的十材料 sorter；但一系列 H/M-specific 升级没有带来稳定相分辨。v7B2 尤其关键，因为它不是普通模型容量实验，而是一个 preregistered physical-observation Pilot：它扩展了厚度、oblique views 和 source set，却跌破 v7B 的 H/M baseline。Phase 1 进一步检验了更强 compact XRT-derived sidecar features 是否能从已有 cube 中恢复 H/M 分离，最终最佳策略仍然选择 v7B baseline/no-sidecar。

这段 negative result 应该正面书写。当前 sorter 的产品价值是 fast pre-screen 与 top-k risk ranking：它可以把 H/M ambiguity 暴露出来，并把不确定样本导向复核方法。正确的产品流程不是宣称 ordinary XRT 直接完成 H/M phase ID，而是 XRT 先做粗分选和候选检索，H/M-critical 样本进入 XRD/Rietveld 或 valence/local-structure 方法复核。

Nature 风格的讨论段可以这样组织：The failed gates do not simply mark a modelling failure; they identify the missing physical variable. Coarse XRT sorting remained useful, but the H/M pair required information about phase and valence that attenuation-derived channels do not directly observe. This turns the failure into a design rule for multimodal sorting: use XRT to triage and rank risk, then use diffraction or valence-sensitive assays when the decision depends on iron-oxide phase.

## 4. v8 观测概念

### v8A diffraction-aware sidecar

下一条主线建议设为 `v8A_diffraction_sidecar_pilot`。它的目标是加入 crystal-structure-aware observation，而不是再增加一个 ordinary attenuation view。最低可行设计是 XRD-like sidecar：窄能谱或准单色 beam、准直、样品 pose 或 grain-orientation averaging、离轴 angularly resolved detectors。导出的特征张量应围绕 `q` 或 `2theta` bins、detector sector、source energy、thickness 和 pose 组织。H/M 的目标特征必须是 phase-specific peak positions、multi-peak ratios、peak-background summaries，而不是 total Rayleigh scatter intensity。

Geant4 可以模拟 photon transport、absorption、Compton scattering、Rayleigh scattering、fluorescence 以及 detector/background effects。但标准 Rayleigh implementations 不等同于带 Bragg peaks 的 bulk powder diffraction，`G4XrayReflection` 也只是 small-angle surface specular reflection，不是 general powder-XRD model。因此 v8A 应明确写成 Geant4 transport 加 custom diffraction photon generator、custom process，或 tabulated powder-pattern sidecar；Hematite/Magnetite 的峰信息应来自 CIF 或 reference peak tables。gate 要检验 phase-specific peaks 在 background、有限 angular resolution、thickness、grain-orientation perturbation 和 noise 下是否仍然存在。

v8A go gate：

- pre-ML physical observability：扰动 angular resolution、peak width、background、thickness 和 orientation 后，H/M peak features 仍达到 `AUC >= 0.95` 或主特征 `d-prime >= 3`；
- small development matrix：H/M min recall `>=0.80`，pairwise H/M min recall `>=0.80`，thickness-blind H/M min recall `>=0.78`；
- robustness：模型不能靠 thickness、total counts 或 total Fe-like attenuation 过关；
- implementation：custom diffraction sidecar 必须显式记录，不能把 ordinary Geant4 Rayleigh scattering 写成 XRD。

v8A no-go：如果 diffraction Pilot 的 H/M min recall 低于 `0.75`，或信号只在无背景理想条件下出现，不扩展 full hard-negative matrix。

### v8B valence/edge-aware sidecar

第二条线建议设为 `v8B_edge_valence_pilot`。它的目标是检验 Fe K-edge 或 XANES-inspired features 是否能编码 Fe2+/Fe3+ 差异。最低可行输出是 Fe K-edge 附近的 energy-resolved table，包含 transmission 或 fluorescence-yield channels、pre-edge/post-edge normalization、derivative peaks、edge-position estimates 和 peak-area ratios。模型必须使用 Hematite/Magnetite material-specific reference spectra 或 cross-section tables；不能把 ordinary XRF 单独写成 H/M phase discriminator。

主要限制是：标准 Geant4 atomic relaxation 和 fluorescence 是 element-level processes，基于 atomic data libraries。它们可以模拟 geometry、attenuation、secondary photons、fluorescence 和 detector response，但不会自然生成 Fe2+/Fe3+ chemical shifts 或 XANES fine structure。后者需要 custom material-specific spectra、FEFF-like inputs 或实验 reference tables。Fe K-edge 路线还带来硬件负担：目标 contrast 在能量上很窄，厚的富铁样本也会受到 penetration 和 self-absorption 限制。

v8B go gate：

- 在提出仿真性能前，必须先有可信 Hematite/Magnetite material-specific edge 或 XANES reference functions；
- 所需 energy resolution 和 flux 必须作为 measurement concept 明确写出，不能默认为 ordinary XRT 硬件已经具备；
- small development matrix 达到 H/M min recall `>=0.80`，并且在 thickness 与 self-absorption perturbation 后仍稳定。

v8B no-go：如果没有可信 material-specific edge/XANES reference，或如果概念需要 eV-level resolution 而产品仍定位为 ordinary industrial XRT，则 v8B 只保留为论文讨论项，不进入主线实现。

## 5. 立即工作包

下一轮 implementation 不应直接启动大型 Geant4 run。应先创建 `v8A_diffraction_sidecar_pilot` preregistration document 和一个 tiny synthetic observability prototype。prototype 可以从 H/M tabulated powder peaks 出发，生成带 controlled background 与 noise 的 `2theta`-binned sidecar features。只有 observability gate 通过后，才值得投资 Geant4 transport integration、detector geometry 和 matrix generation。

论文包应把 v7B 写成正向 coarse-sorting result，把 v7B2 和 Phase 1 写成 negative H/M boundary evidence。作为 Nature 编辑视角，这不是削弱项目，而是把 failure 转化成 design rule：ordinary XRT 可以负责快速筛选与风险排序；当问题依赖 iron-oxide phase 或 valence 时，系统必须引入 diffraction 或 valence-sensitive observation。

## 6. 固定约束

- 后续 export、feature、training、gate、prototype scripts 使用 `/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python`，除非有真实需要并先向用户说明、获得同意。
- 本分支不读取、不写入 shadow/final data。
- v7B2 Pilot 已失败，不强行扩 full v7B2。
- smoke score 不作为 accuracy evidence。
- 在没有新增 diffraction、edge 或 valence-aware observation 之前，不把 GPU/deep model 作为下一主线。

## 7. 写论文前需要核对的官方参考

- Geant4 Low Energy Livermore documentation: https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/introduction/livermore.html
- Geant4 Elastic Scattering documentation: https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/elastic/index.html
- Geant4 Atomic Relaxation documentation: https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/atomic_relaxation/relaxation.html
- Geant4 X-ray Reflection documentation: https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/xrayreflection/G4XrayReflection.html

这些参考只用于限定 Geant4 能模拟什么：Livermore low-energy EM 包含 photoelectric、Compton、Rayleigh、gamma conversion、bremsstrahlung、ionisation、fluorescence 和 Auger emission；gamma elastic scattering 可以包含 Rayleigh 等过程；atomic relaxation 基于原子退激数据而非材料特异性 Fe valence/XANES fine structure；`G4XrayReflection` 是 small-angle surface specular reflection，不能写成 bulk H/M powder diffraction。
