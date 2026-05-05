from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v8a_count_balance_sensitivity import STRATEGIES, build_balanced_subset
from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_medium_count_matched_rework import THRESHOLDS as COUNT_REWORK_THRESHOLDS
from train_v8a_medium_development_model import (
    MAIN_METHODS,
    THRESHOLDS as MODEL_THRESHOLDS,
    build_models,
    evaluate_estimator,
    fit_estimator,
    integrity_summary,
    require_sklearn,
    selected_frame,
    value_for,
)


CLAIM_SCOPE = (
    "development-only count-balanced H/M retest over medium-plus-count-overlap features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def strategy_by_name(name: str) -> dict[str, Any]:
    for strategy in STRATEGIES:
        if str(strategy["strategy"]) == name:
            return strategy
    raise ValueError(f"Unknown count-balance strategy: {name}")


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "match_pair_id" not in frame.columns:
        return {"train": 0, "validation": 0, "stress_holdout": 0}
    values = frame.groupby("split")["match_pair_id"].nunique().to_dict()
    return {split: int(values.get(split, 0)) for split in ["train", "validation", "stress_holdout"]}


def standardized_gap(frame: pd.DataFrame, split: str) -> float:
    subset = frame[frame["split"].astype(str).eq(split)]
    hematite = subset[subset["material"].astype(str).eq("Hematite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
    magnetite = subset[subset["material"].astype(str).eq("Magnetite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
    if len(hematite) == 0 or len(magnetite) == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(hematite) + np.var(magnetite)) + 1e-12)
    return float(abs(float(np.mean(magnetite) - np.mean(hematite))) / pooled)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame, count_balance_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A count-balanced development retest gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Strategy: `{gate['count_balance_strategy']}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        f"- Selected main model: `{gate['selected_main_model']['method']}`",
        f"- Validation H/M min recall: `{gate['validation_main_hm_min_recall']:.4f}`",
        f"- Stress-holdout H/M min recall: `{gate['stress_holdout_main_hm_min_recall']:.4f}`",
        f"- Total-count-only H/M min recall: `{gate['total_count_only_hm_min_recall']:.4f}`",
        f"- Source-off H/M min recall: `{gate['source_off_hm_min_recall']:.4f}`",
        "",
        "## Strategy Evidence",
        "",
        markdown_table(
            count_balance_summary,
            [
                "strategy",
                "train_pairs",
                "validation_pairs",
                "stress_holdout_pairs",
                "best_main_validation_hm_min_recall",
                "best_main_stress_holdout_hm_min_recall",
                "total_count_validation_hm_min_recall",
                "total_count_stress_holdout_hm_min_recall",
                "strategy_passed",
            ],
        ),
        "",
        "## Model Summary",
        "",
        markdown_table(
            selection.sort_values(["eval_split", "family", "hm_min_recall", "accuracy"], ascending=[True, True, False, False]),
            [
                "method",
                "eval_split",
                "family",
                "threshold",
                "hm_min_recall",
                "hematite_recall",
                "magnetite_recall",
                "worst_thickness_hm_min_recall",
                "worst_pose_hm_min_recall",
                "worst_stress_label_hm_min_recall",
                "expected_calibration_error",
            ],
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This retest is development-only evidence that a count-balanced subset can support H/M observability checks. It does not unlock shadow/final or product/hardware/manuscript-grade claims.",
            "",
        ]
    )
    (output_dir / "v8a_count_balanced_retest_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a development-only count-balanced H/M retest gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature")
    parser.add_argument("--training-gate-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_training")
    parser.add_argument("--stress-gate-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_feature_stress_gate")
    parser.add_argument("--count-balance-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_balance_sensitivity")
    parser.add_argument("--extension-config", default="analysis/configs/v8a_count_overlap_extension_config.json")
    parser.add_argument("--strategy", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_balanced_retest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    config = load_json(as_project_path(project_root, args.extension_config))
    strategy_name = str(args.strategy or config["expected_combined_count_balance_support"]["strategy"])
    strategy = strategy_by_name(strategy_name)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    training_gate = load_json(as_project_path(project_root, args.training_gate_dir) / "v8a_event_training_gate.json")
    stress_gate = load_json(as_project_path(project_root, args.stress_gate_dir) / "v8a_event_feature_stress_gate.json")
    count_balance_gate = load_json(as_project_path(project_root, args.count_balance_dir) / "v8a_count_balance_sensitivity_gate.json")
    count_balance_summary = pd.read_csv(as_project_path(project_root, args.count_balance_dir) / "v8a_count_balance_sensitivity_summary.csv")
    selected_strategy_summary = count_balance_summary[count_balance_summary["strategy"].astype(str).eq(strategy_name)].copy()
    if selected_strategy_summary.empty:
        raise RuntimeError(f"Selected count-balance strategy is absent from sensitivity output: {strategy_name}")
    if not bool(selected_strategy_summary.iloc[0]["strategy_passed"]):
        raise RuntimeError(f"Selected count-balance strategy did not pass sensitivity audit: {strategy_name}")

    for name, payload in {
        "schema_gate": schema_gate,
        "feature_manifest": feature_manifest,
        "training_gate": training_gate,
        "stress_gate": stress_gate,
        "count_balance_gate": count_balance_gate,
    }.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing count-balanced retest because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing count-balanced retest because {name} reports existing XRT cube reads.")
    if not bool(schema_gate.get("gate_passed")) or not bool(schema_gate.get("tiny_training_gate_allowed")):
        raise RuntimeError(f"Input combined features are not training-allowed: {schema_gate.get('decision')}")
    if not bool(training_gate.get("gate_passed")):
        raise RuntimeError(f"Combined baseline training gate did not pass: {training_gate.get('decision')}")
    if not bool(stress_gate.get("gate_passed")):
        raise RuntimeError(f"Combined stress gate did not pass: {stress_gate.get('decision')}")
    if not bool(count_balance_gate.get("gate_passed")):
        raise RuntimeError(f"Count-balance sensitivity gate did not pass: {count_balance_gate.get('decision')}")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)
    balanced_source_on = build_balanced_subset(frame, strategy)
    source_off = frame[frame["source_mode"].astype(str).eq("custom_diffraction_off")].copy()
    source_off["count_balance_bin"] = "source_off_control"
    source_off["match_pair_id"] = "source_off_control"
    source_off["match_delta_total_count_norm"] = 0.0
    retest_frame = pd.concat([balanced_source_on, source_off], ignore_index=True, sort=False)
    if balanced_source_on.empty:
        raise RuntimeError(f"Selected count-balance strategy produced no source-on rows: {strategy_name}")

    sk = require_sklearn()
    integrity = integrity_summary(retest_frame, main_cols)
    if integrity["shadow_or_final_splits"]:
        raise RuntimeError(f"Refusing count-balanced retest because shadow/final splits are present: {integrity['shadow_or_final_splits']}")
    if integrity["lineage_like_main_features"]:
        raise RuntimeError(f"Refusing count-balanced retest because main features look lineage-like: {integrity['lineage_like_main_features']}")

    models = build_models(sk, main_cols, total_count_cols, overlap_cols, thickness_pose_cols)
    fitted: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    summary_rows: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    group_rows: list[pd.DataFrame] = []
    sweep_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    for model in models:
        train = selected_frame(retest_frame, "train", model["train_source_mode"])
        estimator = fit_estimator(model, train)
        fitted[model["method"]] = estimator
        validation = selected_frame(retest_frame, "validation", model["eval_source_mode"])
        summary, method_decisions, grouped, sweep, calibration = evaluate_estimator(
            model,
            estimator,
            validation,
            "validation",
            sk,
            selected_threshold=None,
        )
        selected_thresholds[model["method"]] = float(summary["threshold"])
        summary_rows.append(summary)
        for table, sink in [
            (method_decisions, decisions),
            (grouped, group_rows),
            (sweep, sweep_rows),
            (calibration, calibration_rows),
        ]:
            if not table.empty:
                sink.append(table)
        holdout = selected_frame(retest_frame, "stress_holdout", model["eval_source_mode"])
        holdout_summary, holdout_decisions, holdout_grouped, holdout_sweep, holdout_calibration = evaluate_estimator(
            model,
            estimator,
            holdout,
            "stress_holdout",
            sk,
            selected_threshold=selected_thresholds[model["method"]],
        )
        summary_rows.append(holdout_summary)
        for table, sink in [
            (holdout_decisions, decisions),
            (holdout_grouped, group_rows),
            (holdout_sweep, sweep_rows),
            (holdout_calibration, calibration_rows),
        ]:
            if not table.empty:
                sink.append(table)

    selection = pd.DataFrame(summary_rows)
    validation_main = selection[selection["method"].isin(MAIN_METHODS) & selection["eval_split"].eq("validation")]
    selected_main = validation_main.sort_values(
        ["hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall", "worst_stress_label_hm_min_recall", "accuracy"],
        ascending=False,
    ).iloc[0].to_dict()
    selected_method = str(selected_main["method"])
    validation_main_hm = value_for(selection, selected_method, "validation", "hm_min_recall")
    stress_holdout_main_hm = value_for(selection, selected_method, "stress_holdout", "hm_min_recall")
    worst_thickness = min(
        value_for(selection, selected_method, "validation", "worst_thickness_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_thickness_hm_min_recall"),
    )
    worst_pose = min(
        value_for(selection, selected_method, "validation", "worst_pose_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_pose_hm_min_recall"),
    )
    worst_stress_label = min(
        value_for(selection, selected_method, "validation", "worst_stress_label_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_stress_label_hm_min_recall"),
    )
    validation_ece = value_for(selection, selected_method, "validation", "expected_calibration_error")
    stress_holdout_ece = value_for(selection, selected_method, "stress_holdout", "expected_calibration_error")
    total_count_hm = max(
        value_for(selection, "ExtraTreesTotalCountOnly", "validation", "hm_min_recall"),
        value_for(selection, "ExtraTreesTotalCountOnly", "stress_holdout", "hm_min_recall"),
    )
    overlap_hm = value_for(selection, "ExtraTreesOverlapOnly", "validation", "hm_min_recall")
    thickness_pose_hm = value_for(selection, "ExtraTreesThicknessPoseOnly", "validation", "hm_min_recall")
    shuffled_hm = value_for(selection, "ExtraTreesShuffledTrainLabels", "validation", "hm_min_recall")
    source_off_hm = value_for(selection, "ExtraTreesSourceOffLeakage", "validation", "hm_min_recall")
    margin = validation_main_hm - source_off_hm
    pairs = pair_counts(balanced_source_on)

    pass_items = {
        "schema_gate_passed": bool(schema_gate.get("gate_passed")),
        "baseline_training_gate_passed": bool(training_gate.get("gate_passed")),
        "stress_gate_passed": bool(stress_gate.get("gate_passed")),
        "count_balance_sensitivity_gate_passed": bool(count_balance_gate.get("gate_passed")),
        "selected_strategy_passed": bool(selected_strategy_summary.iloc[0]["strategy_passed"]),
        "development_only_no_shadow_final": not integrity["shadow_or_final_splits"],
        "no_lineage_like_main_features": not integrity["lineage_like_main_features"],
        "train_pair_support": pairs["train"] >= COUNT_REWORK_THRESHOLDS["train_pairs_min"],
        "validation_pair_support": pairs["validation"] >= COUNT_REWORK_THRESHOLDS["validation_pairs_min"],
        "stress_holdout_pair_support": pairs["stress_holdout"] >= COUNT_REWORK_THRESHOLDS["stress_holdout_pairs_min"],
        "validation_main_hm_min_recall": validation_main_hm >= MODEL_THRESHOLDS["validation_main_hm_min_recall_min"],
        "stress_holdout_main_hm_min_recall": stress_holdout_main_hm >= MODEL_THRESHOLDS["stress_holdout_main_hm_min_recall_min"],
        "worst_thickness_hm_min_recall": worst_thickness >= MODEL_THRESHOLDS["worst_thickness_hm_min_recall_min"],
        "worst_pose_hm_min_recall": worst_pose >= MODEL_THRESHOLDS["worst_pose_hm_min_recall_min"],
        "worst_stress_label_hm_min_recall": worst_stress_label >= MODEL_THRESHOLDS["worst_stress_label_hm_min_recall_min"],
        "total_count_only_below_ceiling": total_count_hm < MODEL_THRESHOLDS["total_count_only_hm_min_recall_max"],
        "overlap_only_below_ceiling": overlap_hm < MODEL_THRESHOLDS["overlap_only_hm_min_recall_max"],
        "thickness_pose_below_ceiling": thickness_pose_hm < MODEL_THRESHOLDS["thickness_pose_hm_min_recall_max"],
        "shuffled_label_below_ceiling": shuffled_hm < MODEL_THRESHOLDS["shuffled_label_hm_min_recall_max"],
        "source_off_below_ceiling": source_off_hm < MODEL_THRESHOLDS["source_off_hm_min_recall_max"],
        "main_minus_source_off_margin": margin >= MODEL_THRESHOLDS["main_minus_source_off_hm_margin_min"],
        "validation_ece_below_ceiling": validation_ece <= MODEL_THRESHOLDS["validation_ece_max"],
        "stress_holdout_ece_below_ceiling": stress_holdout_ece <= MODEL_THRESHOLDS["stress_holdout_ece_max"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/train_v8a_count_balanced_retest.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_count_balanced_development_retest_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "count_balance_dir": args.count_balance_dir,
        "count_balance_strategy": strategy_name,
        "gate_passed": gate_passed,
        "decision": "count_balanced_development_retest_passed_keep_shadow_final_sealed" if gate_passed else "stop_or_rework_count_balanced_development_retest",
        "selected_main_model": {
            "method": selected_method,
            "threshold": float(selected_main["threshold"]),
            "feature_count": int(selected_main["feature_count"]),
        },
        "matched_pair_counts": pairs,
        "validation_count_gap_standardized": standardized_gap(balanced_source_on, "validation"),
        "stress_holdout_count_gap_standardized": standardized_gap(balanced_source_on, "stress_holdout"),
        "validation_main_hm_min_recall": validation_main_hm,
        "stress_holdout_main_hm_min_recall": stress_holdout_main_hm,
        "worst_thickness_hm_min_recall": worst_thickness,
        "worst_pose_hm_min_recall": worst_pose,
        "worst_stress_label_hm_min_recall": worst_stress_label,
        "validation_expected_calibration_error": validation_ece,
        "stress_holdout_expected_calibration_error": stress_holdout_ece,
        "total_count_only_hm_min_recall": total_count_hm,
        "overlap_only_hm_min_recall": overlap_hm,
        "thickness_pose_hm_min_recall": thickness_pose_hm,
        "shuffled_label_hm_min_recall": shuffled_hm,
        "source_off_hm_min_recall": source_off_hm,
        "main_minus_source_off_hm_margin": margin,
        "thresholds": {
            "model": MODEL_THRESHOLDS,
            "count_rework_support": {
                "train_pairs_min": COUNT_REWORK_THRESHOLDS["train_pairs_min"],
                "validation_pairs_min": COUNT_REWORK_THRESHOLDS["validation_pairs_min"],
                "stress_holdout_pairs_min": COUNT_REWORK_THRESHOLDS["stress_holdout_pairs_min"],
            },
        },
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "integrity_summary": integrity,
        "input_gate_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "baseline_training_gate": training_gate.get("decision"),
            "stress_gate": stress_gate.get("decision"),
            "count_balance_gate": count_balance_gate.get("decision"),
            "feature_peak_table_id": feature_manifest.get("peak_table_id"),
            "feature_source_peak_table_ids": feature_manifest.get("source_peak_table_ids"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest = {
        "generated_by": gate["generated_by"],
        "generated_at_utc": generated_at,
        "protocol_name": gate["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "count_balance_strategy": strategy_name,
        "sample_count": int(len(retest_frame)),
        "balanced_source_on_sample_count": int(len(balanced_source_on)),
        "source_off_control_sample_count": int(len(source_off)),
        "main_feature_count": int(len(main_cols)),
        "control_feature_count": int(len(control_cols)),
        "gate_file": "v8a_count_balanced_retest_gate.json",
    }

    validation_decisions = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    group_recalls = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    threshold_table = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    calibration_bins = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    retest_frame.to_csv(output_dir / "v8a_count_balanced_retest_features.csv", index=False, lineterminator="\n")
    selection.to_csv(output_dir / "v8a_count_balanced_retest_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_count_balanced_retest_decisions.csv", index=False, lineterminator="\n")
    group_recalls.to_csv(output_dir / "v8a_count_balanced_retest_group_recalls.csv", index=False, lineterminator="\n")
    threshold_table.to_csv(output_dir / "v8a_count_balanced_retest_threshold_sweep.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "v8a_count_balanced_retest_calibration_bins.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_count_balanced_retest_manifest.json", manifest)
    write_json(output_dir / "v8a_count_balanced_retest_gate.json", gate)
    write_report(output_dir, gate, selection, selected_strategy_summary)
    print(
        "decision={decision} gate_passed={passed} strategy={strategy} pairs={train}/{validation}/{holdout} main={main:.4f}/{holdout_main:.4f} total_count={total:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            strategy=strategy_name,
            train=pairs["train"],
            validation=pairs["validation"],
            holdout=pairs["stress_holdout"],
            main=validation_main_hm,
            holdout_main=stress_holdout_main_hm,
            total=total_count_hm,
        )
    )


if __name__ == "__main__":
    main()
