from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only residualization protocol rework v1 feature views for v8A H/M sidecar data; "
    "not real-label training evidence, product accuracy, hardware validation, shadow/final validation, "
    "full ten-material matrix evidence, or manuscript-grade powder XRD"
)

DEFAULT_SOURCE_DIR = "results/accuracy_v3/v8a_clean_hm_nullrep_event_to_feature"
DEFAULT_CLEAN_DIR = "results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_event_to_feature"
DEFAULT_OUTPUT_PREFIX = "results/accuracy_v3/v8a_residualization_protocol_rework_v1"


def ensure_clean_dir(path: Path, overwrite: bool, project_root: Path) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    if path.exists() and any(path.iterdir()) and overwrite:
        resolved_output = path.resolve()
        resolved_results = (project_root / "results" / "accuracy_v3").resolve()
        if not resolved_output.is_relative_to(resolved_results):
            raise RuntimeError(f"Refusing to clean unexpected output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def assert_manifest_clean(name: str, payload: dict[str, Any]) -> None:
    if bool(payload.get("shadow_or_final_used", False)):
        raise RuntimeError(f"Refusing rework v1 view build because {name} reports shadow/final use.")
    if bool(payload.get("reads_existing_xrt_cubes", False)):
        raise RuntimeError(f"Refusing rework v1 view build because {name} reports existing XRT cube reads.")
    if bool(payload.get("training_unlocked", False)):
        raise RuntimeError(f"Refusing rework v1 view build because {name} already unlocks training.")


def source_mapping(clean_manifest: dict[str, Any], clean_cols: list[str], source_cols: list[str]) -> dict[str, str]:
    residual_features = clean_manifest.get("residualization", {}).get("features", {})
    mapping: dict[str, str] = {}
    source_col_set = set(source_cols)
    for clean_col in clean_cols:
        manifest_source = residual_features.get(clean_col, {}).get("source_column")
        fallback_source = "diffraction_" + clean_col.removeprefix("diffraction_crystal_clean_")
        source_col = str(manifest_source or fallback_source)
        if source_col not in source_col_set:
            raise RuntimeError(f"Cannot map clean feature {clean_col} to source feature {source_col}.")
        mapping[clean_col] = source_col
    return mapping


def align_source_to_clean(source: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    source_indexed = source.set_index("sample_id", drop=False)
    missing = sorted(set(clean["sample_id"].astype(str)) - set(source_indexed.index.astype(str)))
    if missing:
        raise RuntimeError(f"Source frame is missing sample_id values: {missing[:5]}")
    return pd.DataFrame([source_indexed.loc[sample_id] for sample_id in clean["sample_id"].astype(str)]).reset_index(drop=True)


def zscore_train_apply(frame: pd.DataFrame, columns: list[str], split: pd.Series) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    train_mask = split.astype(str).eq("train").to_numpy()
    scaled = pd.DataFrame(index=frame.index)
    params: dict[str, dict[str, float]] = {}
    for col in columns:
        values = frame[col].fillna(0.0).to_numpy(dtype=np.float64)
        train_values = values[train_mask]
        center = float(np.mean(train_values)) if len(train_values) else 0.0
        scale = float(np.std(train_values)) if len(train_values) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        scaled[col] = (values - center) / scale
        params[col] = {"center": center, "scale": scale}
    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0), params


def pair_center(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if "clean_match_pair_id" not in frame.columns:
        raise RuntimeError("Pair-centered view requires clean_match_pair_id.")
    bad_pairs = []
    for pair_id, group in frame.groupby("clean_match_pair_id", sort=True):
        materials = sorted(group["material"].astype(str).tolist())
        if len(group) != 2 or materials != ["Hematite", "Magnetite"]:
            bad_pairs.append(str(pair_id))
    if bad_pairs:
        raise RuntimeError(f"Pair-centered view found malformed H/M pairs: {bad_pairs[:5]}")
    pair_means = frame.groupby("clean_match_pair_id", sort=False)[feature_cols].transform("mean")
    centered = frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float64) - pair_means.fillna(0.0).to_numpy(dtype=np.float64)
    return pd.DataFrame(centered, columns=feature_cols, index=frame.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_candidate(
    *,
    project_root: Path,
    clean: pd.DataFrame,
    source_aligned: pd.DataFrame,
    source_cols: list[str],
    mapping: dict[str, str],
    output_prefix: str,
    view_name: str,
    diagnostic_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = project_root / f"{output_prefix}_{view_name}_event_to_feature"
    ensure_clean_dir(output_dir, overwrite, project_root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lineage = clean[[col for col in clean.columns if not col.startswith("diffraction_")]].copy()
    source_for_clean = [mapping[clean_col] for clean_col in mapping]
    scaled_source, scaling = zscore_train_apply(source_aligned, source_for_clean, clean["split"])

    renamed_cols: dict[str, str] = {}
    if view_name == "source_scaled_no_residualization":
        for source_col in source_for_clean:
            suffix = source_col.removeprefix("diffraction_")
            renamed_cols[source_col] = f"diffraction_rework_source_scaled_{suffix}"
        features = scaled_source[source_for_clean].rename(columns=renamed_cols)
        transformation = {
            "kind": "train_fit_source_zscore_no_residualization",
            "fit_split": "train",
            "uses_material_labels": False,
            "uses_clean_match_pair_id_for_values": False,
            "residualization_disabled": True,
        }
    elif view_name == "within_pair_contrast":
        for source_col in source_for_clean:
            suffix = source_col.removeprefix("diffraction_")
            renamed_cols[source_col] = f"diffraction_rework_pair_centered_{suffix}"
        pair_frame = pd.concat([lineage[["clean_match_pair_id", "material"]], scaled_source[source_for_clean]], axis=1)
        features = pair_center(pair_frame, source_for_clean).rename(columns=renamed_cols)
        transformation = {
            "kind": "train_fit_source_zscore_then_pair_mean_center",
            "fit_split": "train for z-score only",
            "uses_material_labels": False,
            "uses_clean_match_pair_id_for_values": True,
            "pair_contrast_direction": "row_minus_pair_mean; never H-minus-M or M-minus-H",
            "diagnostic_only_reason": "Pair membership is a clean-design diagnostic dependency and is not a standalone deployment-time feature.",
            "residualization_disabled": True,
        }
    else:
        raise ValueError(f"Unknown rework view: {view_name}")

    output = pd.concat([lineage.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].copy()
    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    feature_cols = list(features.columns)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index", "stress", "origin", "count_bin"]
    lineage_like = [col for col in feature_cols if any(token in col.lower() for token in leak_tokens)]
    gate_passed = not lineage_like
    schema_gate = {
        "generated_by": "analysis/build_v8a_residualization_protocol_rework_v1_views.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_residualization_protocol_rework_v1_view_schema",
        "candidate_view": view_name,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "diagnostic_only_candidate": diagnostic_only,
        "claim_scope": CLAIM_SCOPE,
        "sample_count": int(len(output)),
        "main_feature_count": int(len(feature_cols)),
        "lineage_like_main_features": lineage_like,
        "gate_passed": gate_passed,
        "decision": "rework_v1_view_built_training_locked" if gate_passed else "stop_rework_v1_lineage_like_features",
    }
    manifest = {
        "generated_by": "analysis/build_v8a_residualization_protocol_rework_v1_views.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_residualization_protocol_rework_v1_view_manifest",
        "candidate_view": view_name,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "diagnostic_only_candidate": diagnostic_only,
        "claim_scope": CLAIM_SCOPE,
        "main_feature_columns": feature_cols,
        "source_feature_columns": source_for_clean,
        "source_to_output_feature_mapping": renamed_cols,
        "clean_to_source_feature_mapping": mapping,
        "train_fit_scaling": scaling,
        "transformation": transformation,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(schema_gate))
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(manifest))
    lines = [
        f"# v8A residualization protocol rework v1 view: {view_name}",
        "",
        f"Generated: {generated_at}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        f"- Gate passed: `{str(gate_passed).lower()}`",
        f"- Training unlocked: `false`",
        f"- Diagnostic-only candidate: `{str(diagnostic_only).lower()}`",
        f"- Samples: `{len(output)}`",
        f"- Main feature count: `{len(feature_cols)}`",
        f"- Transformation: `{transformation['kind']}`",
        "",
        "## Claim Boundary",
        "",
        "This materializes a development-only candidate view for null/admission testing. It does not run Geant4 and does not train real-label models.",
        "",
    ]
    (output_dir / "v8a_residualization_protocol_rework_v1_view_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "candidate_view": view_name,
        "output_dir": str(output_dir.relative_to(project_root)),
        "diagnostic_only_candidate": diagnostic_only,
        "sample_count": int(len(output)),
        "main_feature_count": int(len(feature_cols)),
        "gate_passed": gate_passed,
        "lineage_like_main_features": lineage_like,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residualization protocol rework v1 views without real-label training.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    source_dir = project_root / args.source_dir
    clean_dir = project_root / args.clean_dir
    source_manifest = load_json(source_dir / "v8a_event_feature_manifest.json")
    source_schema = load_json(source_dir / "v8a_event_schema_gate.json")
    clean_manifest = load_json(clean_dir / "v8a_event_feature_manifest.json")
    clean_schema = load_json(clean_dir / "v8a_event_schema_gate.json")
    for name, payload in {
        "source_manifest": source_manifest,
        "source_schema": source_schema,
        "clean_manifest": clean_manifest,
        "clean_schema": clean_schema,
    }.items():
        assert_manifest_clean(name, payload)
    source = pd.read_csv(source_dir / "v8a_event_sidecar_features.csv")
    clean = pd.read_csv(clean_dir / "v8a_event_sidecar_features.csv")
    if len(source) != len(clean):
        raise RuntimeError(f"Source/clean row count mismatch: {len(source)} != {len(clean)}")
    clean_cols, _, _, _, _ = feature_sets(clean)
    source_cols, _, _, _, _ = feature_sets(source)
    if not clean_cols or not source_cols:
        raise RuntimeError("Both source and clean views must contain diffraction_* features.")
    mapping = source_mapping(clean_manifest, clean_cols, source_cols)
    source_aligned = align_source_to_clean(source, clean)
    summaries = [
        build_candidate(
            project_root=project_root,
            clean=clean,
            source_aligned=source_aligned,
            source_cols=source_cols,
            mapping=mapping,
            output_prefix=args.output_prefix,
            view_name="source_scaled_no_residualization",
            diagnostic_only=False,
            overwrite=args.overwrite,
        ),
        build_candidate(
            project_root=project_root,
            clean=clean,
            source_aligned=source_aligned,
            source_cols=source_cols,
            mapping=mapping,
            output_prefix=args.output_prefix,
            view_name="within_pair_contrast",
            diagnostic_only=True,
            overwrite=args.overwrite,
        ),
    ]
    summary_dir = project_root / f"{args.output_prefix}_view_build_summary"
    ensure_clean_dir(summary_dir, args.overwrite, project_root)
    summary = {
        "generated_by": "analysis/build_v8a_residualization_protocol_rework_v1_views.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_residualization_protocol_rework_v1_view_build_summary",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "claim_scope": CLAIM_SCOPE,
        "source_dir": args.source_dir,
        "clean_dir": args.clean_dir,
        "candidate_views": summaries,
        "gate_passed": all(item["gate_passed"] for item in summaries),
        "decision": "rework_v1_candidate_views_built_training_locked",
    }
    write_json(summary_dir / "v8a_residualization_protocol_rework_v1_view_build_gate.json", json_clean(summary))
    pd.DataFrame(summaries).to_csv(summary_dir / "v8a_residualization_protocol_rework_v1_view_build_summary.csv", index=False, lineterminator="\n")
    print(
        "decision={decision} views={views} training_unlocked=false".format(
            decision=summary["decision"],
            views=",".join(item["candidate_view"] for item in summaries),
        )
    )


if __name__ == "__main__":
    main()
