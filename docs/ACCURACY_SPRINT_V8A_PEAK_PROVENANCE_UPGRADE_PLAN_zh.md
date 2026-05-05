# Accuracy Sprint v8A: H/M peak provenance upgrade plan

Date: 2026-05-05

## 1. Decision

The current H/M peak table `hm_powder_peaks_project_scan_v8a` remains a development-only anchor. It is good enough for smoke tests and schema checks, but not enough for a full v8A matrix, shadow/final validation, hardware claims, or manuscript-grade powder-XRD claims.

The 90-row v8A balanced event-feature smoke may proceed to development design review after its tiny gate, but any promotion beyond that tiny development gate requires upgraded peak provenance.

## 2. Current limitation

Current manifest:

- `source_models/config/diffraction_peak_tables/hm_powder_peaks_project_scan_v8a_manifest.json`

Known limitations:

- approximate project-scan anchors;
- prototype relative intensities;
- no documented CIF/Rietveld/literature derivation;
- no instrument response, preferred orientation, grain-size, impurity, mixture, fluorescence, or background provenance;
- valid only for development smokes and schema tests.

## 3. Upgrade target

Create a replacement or successor manifest before any full v8A matrix. The preferred target is:

- `hm_powder_peaks_cif_or_literature_v8a_manifest.json`

Required fields:

- `peak_table_id`
- `version`
- `status`
- `reference_type`
- `reference_citation`
- `reference_url_or_doi`
- `wavelength_a`
- `intensity_normalization`
- `materials`
- `material`
- `phase_name`
- `chemical_formula`
- `structure_note`
- `peak_id`
- `two_theta_deg`
- `q_a_inv`
- `d_a`
- `relative_intensity`
- `known_limitations`
- `upgrade_required_before`

## 4. Preferred provenance routes

Use one of these routes, in order of preference:

1. CIF-derived route:
   - identify Hematite and Magnetite CIF sources;
   - document database/source URL or DOI;
   - compute or extract peak positions and relative intensities using a reproducible script;
   - record wavelength and normalization.
2. Literature/Rietveld route:
   - cite a peer-reviewed table or reference pattern;
   - preserve phase, wavelength, and intensity normalization;
   - record any approximations used to convert 2theta to q/d.
3. Measured-reference route:
   - use instrument-specific reference scans;
   - record sample preparation, wavelength/energy, detector geometry, and normalization.

Do not use anonymous web snippets or untraceable peak lists as the final provenance source.

## 5. Acceptance gate

The upgraded manifest passes only if:

- JSON parses cleanly;
- every H/M peak has q/d values derived from a stated wavelength or source energy;
- relative intensities are explicitly normalized;
- at least one external citation or database source is recorded;
- limitations are explicit;
- status is still no stronger than `development_reference_candidate` until reviewed;
- the schema review doc is updated to point at the new manifest.

## 6. Claim boundary

Even with an upgraded peak manifest, a tiny v8A event-feature gate remains development evidence. It may support continuing toward a larger development matrix, but it still does not support:

- ordinary-XRT H/M solved;
- product accuracy;
- hardware validation;
- shadow/final claims;
- manuscript-grade powder-XRD validation without separate review.
