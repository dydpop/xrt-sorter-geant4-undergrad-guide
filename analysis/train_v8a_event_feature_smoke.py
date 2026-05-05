from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HM_PAIR = ("Hematite", "Magnetite")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/train_v8a_event_feature_smoke.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "roc_auc_score": roc_auc_score,
    }


def feature_sets(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    excluded_numeric_lineage = {
        "row_index",
        "random_seed",
        "thickness_mm",
        "pose_index",
        "development_only",
        "shadow_or_final_used",
    }
    main_cols = [col for col in numeric_cols if col.startswith("diffraction_") and col not in excluded_numeric_lineage]
    total_count_cols = [
        col
        for col in numeric_cols
        if col.startswith("control_total_count_")
        or col in {"control_high_angle_primary_norm", "control_direct_primary_norm", "control_scattered_primary_norm"}
    ]
    overlap_cols = [col for col in numeric_cols if col.startswith("control_overlap_only_")]
    thickness_pose_cols = [col for col in numeric_cols if col.startswith("control_thickness_pose_")]
    control_cols = sorted(set(total_count_cols + overlap_cols + thickness_pose_cols))
    return main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for material in HM_PAIR:
        mask = y_true == material
        recalls[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return recalls


def evaluate_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    method_name: str,
    estimator: Any,
    *,
    train_source_mode: str,
    validation_source_mode: str,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 9307,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_mask = frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq(train_source_mode)
    validation_mask = frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq(validation_source_mode)
    x_train = frame.loc[train_mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    x_validation = frame.loc[validation_mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = frame.loc[train_mask, "material"].astype(str).to_numpy()
    y_validation = frame.loc[validation_mask, "material"].astype(str).to_numpy()
    if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
        raise RuntimeError(
            f"{method_name} needs both H/M classes in train and validation; train={set(y_train)} validation={set(y_validation)}"
        )
    if shuffle_train_labels:
        rng = np.random.default_rng(shuffle_seed)
        y_train = rng.permutation(y_train)
    estimator.fit(x_train, y_train)
    predictions = np.asarray(estimator.predict(x_validation)).astype(str)
    recalls = pair_recalls(y_validation, predictions)
    decisions = frame.loc[
        validation_mask,
        ["sample_id", "split", "material", "source_mode", "source_id", "random_seed", "thickness_mm", "pose_index"],
    ].copy()
    decisions["method"] = method_name
    decisions["prediction"] = predictions
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == predictions
    by_thickness = []
    for _, group in decisions.groupby("thickness_mm", sort=True):
        thickness_recalls = pair_recalls(group["material"].astype(str).to_numpy(), group["prediction"].astype(str).to_numpy())
        by_thickness.append(min(thickness_recalls.values()))
    return (
        {
            "method": method_name,
            "train_source_mode": train_source_mode,
            "validation_source_mode": validation_source_mode,
            "feature_count": int(len(feature_cols)),
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "hematite_recall": recalls["Hematite"],
            "magnetite_recall": recalls["Magnetite"],
            "hm_min_recall": float(min(recalls.values())),
            "pairwise_hm_min_recall": float(min(recalls.values())),
            "worst_thickness_hm_min_recall": float(min(by_thickness)) if by_thickness else 0.0,
        },
        decisions,
    )


def observability_metrics(frame: pd.DataFrame, feature_cols: list[str], roc_auc_score: Any) -> pd.DataFrame:
    validation = frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    y = frame.loc[validation, "material"].astype(str).to_numpy()
    y_binary = (y == "Magnetite").astype(int)
    rows = []
    for col in feature_cols:
        values = frame.loc[validation, col].fillna(0.0).to_numpy(dtype=np.float64)
        h = values[y == "Hematite"]
        m = values[y == "Magnetite"]
        pooled = float(np.sqrt(0.5 * (np.var(h) + np.var(m)) + 1e-12))
        d_prime = abs(float(np.mean(h) - np.mean(m))) / pooled
        try:
            auc = float(roc_auc_score(y_binary, values))
            oriented_auc = max(auc, 1.0 - auc)
        except ValueError:
            auc = 0.5
            oriented_auc = 0.5
        rows.append(
            {
                "feature": col,
                "oriented_auc": oriented_auc,
                "raw_auc_magnetite_positive": auc,
                "d_prime_abs": d_prime,
                "hematite_mean": float(np.mean(h)),
                "magnetite_mean": float(np.mean(m)),
            }
        )
    return pd.DataFrame(rows).sort_values(["oriented_auc", "d_prime_abs"], ascending=[False, False])


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# v8A event-feature tiny training gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only tiny training/control gate using event-derived sidecar features from the completed v8A Geant4 boundary smoke. This is not product accuracy, not shadow/final validation, not ordinary XRT H/M sorting, and not manuscript-grade powder XRD evidence.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Best main method: `{gate['best_main_method']}`",
        f"- Main H/M min recall: `{gate['best_main_hm_min_recall']:.4f}`",
        f"- Worst-thickness H/M min recall: `{gate['best_main_worst_thickness_hm_min_recall']:.4f}`",
        f"- Total-count-only H/M min recall: `{gate['total_count_hm_min_recall']:.4f}`",
        f"- Overlap-only H/M min recall: `{gate['overlap_only_hm_min_recall']:.4f}`",
        f"- Thickness/pose-only H/M min recall: `{gate['thickness_pose_hm_min_recall']:.4f}`",
        f"- Shuffled-label H/M min recall: `{gate['shuffled_label_hm_min_recall']:.4f}`",
        f"- Source-off H/M min recall: `{gate['source_off_hm_min_recall']:.4f}`",
        "",
        "## Model Selection",
        "",
        markdown_table(
            selection.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False),
            ["method", "validation_source_mode", "hm_min_recall", "hematite_recall", "magnetite_recall", "worst_thickness_hm_min_recall"],
        ),
        "",
        "## Top Main Features",
        "",
        markdown_table(metrics.head(8), ["feature", "oriented_auc", "d_prime_abs"]),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        for reason in gate["stop_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_event_training_gate_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny training/control gate on v8A event-derived sidecar features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_event_to_feature_smoke")
    parser.add_argument("--schema-contract", default="analysis/configs/v8a_diffraction_output_schema_contract.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_event_training_smoke")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    sk = require_sklearn()

    schema = load_json(project_root / args.schema_contract)
    thresholds = schema["gates"]["tiny_event_to_feature_gate"]
    gate_in = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest_in = load_json(input_dir / "v8a_event_feature_manifest.json")
    if not bool(gate_in.get("gate_passed", False)):
        raise RuntimeError(f"Event schema gate did not pass: {gate_in.get('decision')}")
    if not bool(gate_in.get("tiny_training_gate_allowed", False)):
        raise RuntimeError(f"Tiny training gate is not allowed: {gate_in.get('stop_reasons')}")
    if bool(gate_in.get("shadow_or_final_used", False)) or bool(manifest_in.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing event-feature training because input reports shadow/final use.")
    if bool(gate_in.get("reads_existing_xrt_cubes", False)) or bool(manifest_in.get("reads_existing_xrt_cubes", False)):
        raise RuntimeError("Refusing event-feature training because input reports existing XRT cube reads.")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No main diffraction_* features available.")

    models: list[tuple[str, Any, list[str], str, str, bool]] = [
        (
            "LogisticEventMain",
            sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9401)),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesEventMain",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9402, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9403, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9404, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            overlap_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesThicknessPoseOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9405, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            thickness_pose_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9406, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            True,
        ),
        (
            "ExtraTreesSourceOffLeakage",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9407, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_off",
            "custom_diffraction_off",
            False,
        ),
    ]

    rows = []
    decisions = []
    for method_name, estimator, cols, train_mode, validation_mode, shuffle_labels in models:
        if not cols:
            rows.append(
                {
                    "method": method_name,
                    "train_source_mode": train_mode,
                    "validation_source_mode": validation_mode,
                    "feature_count": 0,
                    "train_samples": 0,
                    "validation_samples": 0,
                    "hematite_recall": 0.0,
                    "magnetite_recall": 0.0,
                    "hm_min_recall": 0.0,
                    "pairwise_hm_min_recall": 0.0,
                    "worst_thickness_hm_min_recall": 0.0,
                    "status": "not_evaluable_no_features",
                }
            )
            continue
        row, method_decisions = evaluate_model(
            frame,
            cols,
            method_name,
            estimator,
            train_source_mode=train_mode,
            validation_source_mode=validation_mode,
            shuffle_train_labels=shuffle_labels,
        )
        row["status"] = "evaluated"
        rows.append(row)
        decisions.append(method_decisions)

    selection = pd.DataFrame(rows)
    validation_decisions = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    metrics = observability_metrics(frame, main_cols, sk["roc_auc_score"])
    main_selection = selection[selection["method"].isin(["LogisticEventMain", "ExtraTreesEventMain"])]
    best_main = main_selection.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0].to_dict()

    def hm_value(method: str) -> float:
        values = selection.loc[selection["method"].eq(method), "hm_min_recall"]
        return float(values.iloc[0]) if not values.empty else 0.0

    total_count_hm = hm_value("ExtraTreesTotalCountOnly")
    overlap_hm = hm_value("ExtraTreesOverlapOnly")
    thickness_pose_hm = hm_value("ExtraTreesThicknessPoseOnly")
    shuffled_hm = hm_value("ExtraTreesShuffledTrainLabels")
    source_off_hm = hm_value("ExtraTreesSourceOffLeakage")
    main_hm = float(best_main["hm_min_recall"])
    worst_thickness_hm = float(best_main["worst_thickness_hm_min_recall"])
    guard_pass = bool(
        total_count_hm < float(thresholds["total_count_only_hm_min_recall_max"])
        and overlap_hm < float(thresholds["overlap_only_hm_min_recall_max"])
        and shuffled_hm < float(thresholds["shuffled_label_hm_min_recall_max"])
        and source_off_hm < float(thresholds["leakage_off_hm_min_recall_max"])
        and main_hm - source_off_hm >= float(thresholds["main_minus_leakage_margin_min"])
    )
    ml_pass = bool(
        main_hm >= float(thresholds["hm_min_recall_min"])
        and worst_thickness_hm >= float(thresholds["worst_thickness_hm_min_recall_min"])
    )
    manifest_pass = bool(
        manifest_in.get("development_only")
        and not manifest_in.get("shadow_or_final_used")
        and not manifest_in.get("reads_existing_xrt_cubes")
        and not manifest_in.get("runs_geant4")
        and manifest_in.get("bin_axis") in {"q_a_inv", "d_a"}
    )
    stop_reasons: list[str] = []
    if not ml_pass:
        stop_reasons.append("Main event-derived diffraction features did not meet H/M recall thresholds.")
    if not guard_pass:
        stop_reasons.append("At least one control/leakage guard exceeded its allowed threshold or margin.")
    if not manifest_pass:
        stop_reasons.append("Input manifest did not satisfy development-only/no-shadow/no-XRT-cube requirements.")
    gate_passed = bool(ml_pass and guard_pass and manifest_pass)
    gate = {
        "generated_by": "analysis/train_v8a_event_feature_smoke.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_event_feature_tiny_training_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": "development-only event-feature tiny training/control gate; not product accuracy or manuscript-grade powder XRD",
        "input_dir": args.input_dir,
        "gate_passed": gate_passed,
        "decision": "proceed_to_v8a_balanced_dev_design_review" if gate_passed else "stop_or_rework_v8a_event_feature_tiny_gate",
        "ml_pass": ml_pass,
        "guard_pass": guard_pass,
        "manifest_pass": manifest_pass,
        "best_main_method": str(best_main["method"]),
        "best_main_hm_min_recall": main_hm,
        "best_main_worst_thickness_hm_min_recall": worst_thickness_hm,
        "best_main_hematite_recall": float(best_main["hematite_recall"]),
        "best_main_magnetite_recall": float(best_main["magnetite_recall"]),
        "total_count_hm_min_recall": total_count_hm,
        "overlap_only_hm_min_recall": overlap_hm,
        "thickness_pose_hm_min_recall": thickness_pose_hm,
        "shuffled_label_hm_min_recall": shuffled_hm,
        "source_off_hm_min_recall": source_off_hm,
        "main_minus_source_off_hm_margin": main_hm - source_off_hm,
        "thresholds": thresholds,
        "feature_counts": {
            "main": len(main_cols),
            "control": len(control_cols),
            "total_count": len(total_count_cols),
            "overlap": len(overlap_cols),
            "thickness_pose": len(thickness_pose_cols),
        },
        "stop_reasons": stop_reasons,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest = {
        "generated_by": "analysis/train_v8a_event_feature_smoke.py",
        "generated_at_utc": gate["generated_at_utc"],
        "protocol_name": "v8A_event_feature_tiny_training_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "sample_count": int(len(frame)),
        "validation_sample_count": int(frame["split"].astype(str).eq("validation").sum()),
        "main_feature_count": len(main_cols),
        "control_feature_count": len(control_cols),
        "gate_file": "v8a_event_training_gate.json",
    }
    selection.to_csv(output_dir / "v8a_event_training_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_event_training_validation_decisions.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "v8a_event_training_observability_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_event_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_event_training_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, selection, metrics)
    print(
        "decision={decision} gate_passed={passed} main_hm={main:.4f} source_off_hm={source_off:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            main=main_hm,
            source_off=source_off_hm,
        )
    )


if __name__ == "__main__":
    main()
