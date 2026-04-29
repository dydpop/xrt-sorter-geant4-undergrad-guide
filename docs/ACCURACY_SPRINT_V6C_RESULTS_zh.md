# Accuracy Sprint v6c results: H/M source-design gate

Date: 2026-04-29

## Summary

The `v6c_hm_source_design` Geant4 batch completed successfully, but the development gate failed. Shadow validation, GPU grid search, ten-material expansion, and final locked testing remain blocked.

This is a negative result for the current本科级 virtual XRT source/detector design. It should be recorded as evidence that the Hematite/Magnetite bottleneck is not solved by the added high-energy/source-variant/side-scatter feature set.

## Run Status

- profile: `v6c_hm_source_design`
- selected rows: 4752
- completed: 4752
- failed: 0
- pending: 0
- audit output: `results/accuracy_v3/v6c_hm_source_design/`
- gate report: `results/accuracy_v3/v6c_hm_source_design/gate_v6c.json`

## Gate Result

Observed development metrics:

- selected method: `ExtraTrees`
- thickness-aware H/M min recall: `0.6333333333333333`
- thickness-blind H/M min recall: `0.65`
- pairwise H/M min recall: `0.6333333333333333`
- validation support per class: `60`
- runner failures: `0`
- gate passed: `false`

Required thresholds:

- thickness-aware H/M min recall >= `0.80`
- thickness-blind H/M min recall >= `0.75`
- pairwise H/M min recall >= `0.75`
- validation support per class >= `120`

## Interpretation

v6c improved the measurement design and verified the pipeline, but it did not solve H/M separability. The failure pattern remains direct mutual confusion:

- Hematite validation recall: `0.6833333333333333`; misses: `19`, all to Magnetite.
- Magnetite validation recall: `0.6333333333333333`; misses: `22`, all to Hematite.

The thickness-blind score being slightly higher than thickness-aware does not rescue the result. Both are below gate, and support per class is below the preregistered `120` threshold because this H/M-only fused validation frame produces `60` samples per class.

## Decision

- Do not run shadow seeds `2301-2306`.
- Do not run GPU XGBoost grid on v6c.
- Do not generate ten-material expansion or locked final tests from v6c.
- Treat v6c as a negative development result and update the paper-facing limitation: this simplified virtual XRT modality still does not provide robust H/M separation under the current assumptions.

## Recommended Next Step

As Nature editor / project mentor / product reviewer, the correct next move is not another larger model search. The team should either:

1. Stop the accuracy sprint and write the H/M limitation honestly, using v6/v6b/v6c as staged negative evidence; or
2. Pre-register a genuinely new modality/design before more simulation, for example a higher-fidelity detector physics model, different scatter-angle geometry, true spectral detector response, or another measurement modality.

Do not tune on shadow or final seeds.
