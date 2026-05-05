from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json, pair_recalls
from train_v8a_medium_development_model import expected_calibration_error


MATCH_KEY_COLUMNS = ["split", "stress_label", "thickness_mm", "pose_index"]
MATCH_COLUMN = "control_total_count_norm"
PRIMARY_MATCH_TOLERANCE = 0.020
SENSITIVITY_TOLERANCES = [0.005, 0.010, 0.015, 0.020, 0.030]
THRESHOLDS = {
    "train_pairs_min": 100,
    "validation_pairs_min": 50,
    "stress_holdout_pairs_min": 50,
    "match_delta_total_count_norm_max": 0.020,
    "main_hm_min_recall_min": 0.95,
    "stress_holdout_main_hm_min_recall_min": 0.95,
    "worst_thickness_hm_min_recall_min": 0.90,
    "worst_pose_hm_min_recall_min": 0.90,
    "worst_stress_label_hm_min_recall_min": 0.90,
    "total_count_only_hm_min_recall_max": 0.60,
    "overlap_only_hm_min_recall_max": 0.60,
    "thickness_pose_hm_min_recall_max": 0.60,
    "shuffled_label_hm_min_recall_max": 0.55,
    "source_off_hm_min_recall_max": 0.60,
    "main_minus_total_count_hm_margin_min": 0.35,
    "main_minus_source_off_hm_margin_min": 0.35,
    "expected_calibration_error_max": 0.25,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/train_v8a_medium_count_matched_rework.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
    }


def build_count_matched_frame(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    source_on = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    source_on = source_on.reset_index(drop=False).rename(columns={"index": "input_feature_row_index"})
    rows: list[pd.Series] = []
    for keys, group in source_on.groupby(MATCH_KEY_COLUMNS, sort=True):
        hematite = group[group["material"].astype(str).eq("Hematite")].sort_values(MATCH_COLUMN)
        magnetite = group[group["material"].astype(str).eq("Magnetite")].sort_values(MATCH_COLUMN)
        used_m: set[int] = set()
        pair_number = 0
        for h_index, h_row in hematite.iterrows():
            available = magnetite[~magnetite.index.isin(used_m)]
            if available.empty:
                break
            diffs = (available[MATCH_COLUMN] - float(h_row[MATCH_COLUMN])).abs()
            m_index = int(diffs.idxmin())
            delta = float(diffs.loc[m_index])
            if delta > tolerance:
                continue
            used_m.add(m_index)
            pair_number += 1
            pair_id = "|".join(str(value) for value in keys) + f"|pair{pair_number:03d}"
            for row_index, role in [(h_index, "hematite"), (m_index, "magnetite")]:
                row = source_on.loc[row_index].copy()
                row["match_pair_id"] = pair_id
                row["match_role"] = role
                row["match_delta_total_count_norm"] = delta
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(frame.columns) + ["match_pair_id", "match_role", "match_delta_total_count_norm"])
    matched = pd.DataFrame(rows).sort_values(["split", "stress_label", "thickness_mm", "pose_index", "match_pair_id", "material"])
    return matched.reset_index(drop=True)


def split_source(frame: pd.DataFrame, split: str, source_mode: str) -> pd.DataFrame:
    return frame[
        frame["split"].astype(str).eq(split)
        & frame["source_mode"].astype(str).eq(source_mode)
    ].copy()


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = pair_recalls(y_true, y_pred)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "hematite_recall": recalls["Hematite"],
        "magnetite_recall": recalls["Magnetite"],
        "hm_min_recall": float(min(recalls.values())),
    }


def threshold_sweep(sample_frame: pd.DataFrame, probabilities: np.ndarray, method: str, eval_split: str) -> pd.DataFrame:
    y_true = sample_frame["material"].astype(str).to_numpy()
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite")
        rows.append(
            {
                "method": method,
                "eval_split": eval_split,
                "threshold": float(threshold),
                "samples": int(len(sample_frame)),
                **metrics_from_predictions(y_true, predictions.astype(str)),
            }
        )
    return pd.DataFrame(rows)


def magnetite_probability(estimator: Any, x_values: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x_values)
    classes = [str(item) for item in estimator.classes_]
    return probabilities[:, classes.index("Magnetite")].astype(np.float64) if "Magnetite" in classes else np.zeros(len(x_values))


def group_recall_rows(decisions: pd.DataFrame, method: str, eval_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name in ["thickness_mm", "pose_index", "stress_label"]:
        for value, group in decisions.groupby(group_name, sort=True):
            rows.append(
                {
                    "method": method,
                    "eval_split": eval_split,
                    "group": group_name,
                    "value": value,
                    "samples": int(len(group)),
                    **metrics_from_predictions(
                        group["material"].astype(str).to_numpy(),
                        group["prediction"].astype(str).to_numpy(),
                    ),
                }
            )
    return rows


def fit_residualizer(train_frame: pd.DataFrame, feature_cols: list[str], control_cols: list[str]) -> np.ndarray:
    design = np.c_[np.ones(len(train_frame)), train_frame[control_cols].fillna(0.0).to_numpy(dtype=np.float64)]
    response = train_frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    return np.linalg.pinv(design) @ response


def apply_residualizer(frame: pd.DataFrame, feature_cols: list[str], control_cols: list[str], coefficients: np.ndarray) -> np.ndarray:
    design = np.c_[np.ones(len(frame)), frame[control_cols].fillna(0.0).to_numpy(dtype=np.float64)]
    response = frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    return response - design @ coefficients


def evaluate_model(
    method: str,
    family: str,
    estimator: Any,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    eval_split: str,
    feature_builder: Callable[[pd.DataFrame], np.ndarray],
    sk: dict[str, Any],
    *,
    shuffle_labels: bool = False,
    selected_threshold: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, float]:
    x_train = feature_builder(train_frame)
    y_train = train_frame["material"].astype(str).to_numpy()
    if shuffle_labels:
        y_train = np.random.default_rng(9907).permutation(y_train)
    if len(set(y_train)) < 2 or len(eval_frame) == 0:
        empty = pd.DataFrame()
        return (
            {
                "method": method,
                "family": family,
                "eval_split": eval_split,
                "status": "not_evaluable",
                "samples": int(len(eval_frame)),
                "hm_min_recall": 0.0,
            },
            empty,
            empty,
            empty,
            empty,
            None,
            selected_threshold if selected_threshold is not None else 0.5,
        )
    fitted = deepcopy(estimator)
    fitted.fit(x_train, y_train)
    x_eval = feature_builder(eval_frame)
    probabilities = magnetite_probability(fitted, x_eval)
    sweep = threshold_sweep(eval_frame, probabilities, method, eval_split)
    if selected_threshold is None:
        ranked = sweep.assign(threshold_distance_to_0p5=(sweep["threshold"] - 0.5).abs())
        threshold = float(
            ranked.sort_values(["hm_min_recall", "accuracy", "threshold_distance_to_0p5"], ascending=[False, False, True])
            .iloc[0]["threshold"]
        )
    else:
        threshold = float(selected_threshold)
    predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite").astype(str)
    y_true = eval_frame["material"].astype(str).to_numpy()
    y_binary = (y_true == "Magnetite").astype(int)
    try:
        brier = float(sk["brier_score_loss"](y_binary, probabilities))
    except ValueError:
        brier = float("nan")
    try:
        logloss = float(sk["log_loss"](y_binary, probabilities, labels=[0, 1]))
    except ValueError:
        logloss = float("nan")
    ece, calibration_bins = expected_calibration_error(y_binary, probabilities)
    calibration_bins.insert(0, "eval_split", eval_split)
    calibration_bins.insert(0, "method", method)
    decisions = eval_frame[
        [
            "sample_id",
            "split",
            "material",
            "source_mode",
            "stress_label",
            "source_id",
            "random_seed",
            "thickness_mm",
            "pose_index",
            "match_pair_id",
            "match_delta_total_count_norm",
        ]
    ].copy()
    decisions["method"] = method
    decisions["threshold"] = threshold
    decisions["probability_magnetite"] = probabilities
    decisions["prediction"] = predictions
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == predictions
    group_recalls = pd.DataFrame(group_recall_rows(decisions, method, eval_split))
    metrics = metrics_from_predictions(y_true, predictions)

    def worst(group: str) -> float:
        values = group_recalls.loc[group_recalls["group"].eq(group), "hm_min_recall"] if not group_recalls.empty else pd.Series(dtype=float)
        return float(values.min()) if not values.empty else 0.0

    summary = {
        "method": method,
        "family": family,
        "eval_split": eval_split,
        "status": "evaluated",
        "samples": int(len(eval_frame)),
        "feature_count": int(x_train.shape[1]),
        "threshold": threshold,
        **metrics,
        "worst_thickness_hm_min_recall": worst("thickness_mm"),
        "worst_pose_hm_min_recall": worst("pose_index"),
        "worst_stress_label_hm_min_recall": worst("stress_label"),
        "brier_score": brier,
        "log_loss": logloss,
        "expected_calibration_error": ece,
    }
    return summary, decisions, group_recalls, sweep, calibration_bins, fitted, threshold


def evaluate_train_validation_holdout(
    method: str,
    family: str,
    estimator: Any,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], np.ndarray],
    sk: dict[str, Any],
    *,
    shuffle_labels: bool = False,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    validation_summary, validation_decisions, validation_groups, validation_sweep, validation_calibration, _, threshold = evaluate_model(
        method,
        family,
        estimator,
        train_frame,
        validation_frame,
        "validation",
        feature_builder,
        sk,
        shuffle_labels=shuffle_labels,
    )
    holdout_summary, holdout_decisions, holdout_groups, holdout_sweep, holdout_calibration, _, _ = evaluate_model(
        method,
        family,
        estimator,
        train_frame,
        holdout_frame,
        "stress_holdout",
        feature_builder,
        sk,
        shuffle_labels=shuffle_labels,
        selected_threshold=threshold,
    )
    return (
        [validation_summary, holdout_summary],
        [validation_decisions, holdout_decisions],
        [validation_groups, holdout_groups],
        [validation_sweep, holdout_sweep],
        [validation_calibration, holdout_calibration],
    )


def split_pairs(frame: pd.DataFrame) -> dict[str, int]:
    pairs = frame.groupby("split")["match_pair_id"].nunique().to_dict() if not frame.empty else {}
    return {split: int(pairs.get(split, 0)) for split in ["train", "validation", "stress_holdout"]}


def standardized_gap(left: pd.Series, right: pd.Series) -> float:
    pooled = np.sqrt(0.5 * (np.var(left.to_numpy(dtype=np.float64)) + np.var(right.to_numpy(dtype=np.float64))) + 1e-12)
    return float(abs(float(left.mean()) - float(right.mean())) / pooled)


def count_gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split", sort=True):
        hematite = group[group["material"].astype(str).eq("Hematite")]
        magnetite = group[group["material"].astype(str).eq("Magnetite")]
        rows.append(
            {
                "split": split,
                "hematite_mean_total_count_norm": float(hematite[MATCH_COLUMN].mean()),
                "magnetite_mean_total_count_norm": float(magnetite[MATCH_COLUMN].mean()),
                "mean_difference_magnetite_minus_hematite": float(magnetite[MATCH_COLUMN].mean() - hematite[MATCH_COLUMN].mean()),
                "standardized_gap_abs": standardized_gap(hematite[MATCH_COLUMN], magnetite[MATCH_COLUMN]),
                "max_pair_delta_total_count_norm": float(group["match_delta_total_count_norm"].max()),
                "mean_pair_delta_total_count_norm": float(group["match_delta_total_count_norm"].mean()),
                "pairs": int(group["match_pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def value_for(selection: pd.DataFrame, method: str, split: str, field: str) -> float:
    values = selection.loc[selection["method"].eq(method) & selection["eval_split"].eq(split), field]
    return float(values.iloc[0]) if not values.empty else 0.0


def max_value_for(selection: pd.DataFrame, method: str, field: str) -> float:
    values = selection.loc[selection["method"].eq(method), field]
    return float(values.max()) if not values.empty else 0.0


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


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame, count_gaps: pd.DataFrame, residualized: pd.DataFrame) -> None:
    lines = [
        "# v8A medium count-matched rework gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only total-count-confounding rework. This consumes already generated medium event features only; it is not product accuracy, not shadow/final validation, and not manuscript-grade powder XRD evidence.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Match tolerance on `{MATCH_COLUMN}`: `{gate['match_tolerance_total_count_norm']:.4f}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        f"- Selected main model: `{gate['selected_main_model']['method']}`",
        f"- Validation H/M min recall: `{gate['validation_main_hm_min_recall']:.4f}`",
        f"- Stress-holdout H/M min recall: `{gate['stress_holdout_main_hm_min_recall']:.4f}`",
        f"- Total-count-only max H/M min recall: `{gate['total_count_only_max_hm_min_recall']:.4f}`",
        f"- Main minus total-count margin: `{gate['main_minus_total_count_hm_margin']:.4f}`",
        "",
        "## Count Balance",
        "",
        markdown_table(
            count_gaps,
            [
                "split",
                "pairs",
                "hematite_mean_total_count_norm",
                "magnetite_mean_total_count_norm",
                "mean_difference_magnetite_minus_hematite",
                "standardized_gap_abs",
                "max_pair_delta_total_count_norm",
            ],
        ),
        "",
        "## Model Summary",
        "",
        markdown_table(
            selection.sort_values(["eval_split", "family", "hm_min_recall"], ascending=[True, True, False]),
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
        "## Residualized Sensitivity",
        "",
        markdown_table(
            residualized.sort_values(["residualization", "method", "eval_split"]),
            ["residualization", "method", "eval_split", "hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall"],
        ),
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
    (output_dir / "v8a_medium_count_matched_rework_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def load_required_inputs(input_dir: Path, phase4_dir: Path, total_count_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    phase4_gate = load_json(phase4_dir / "v8a_medium_development_model_gate.json")
    total_count_diagnostic = load_json(total_count_dir / "v8a_total_count_control_diagnostic.json")
    if not bool(schema_gate.get("gate_passed", False)):
        raise RuntimeError(f"Event schema gate did not pass: {schema_gate.get('decision')}")
    if bool(schema_gate.get("shadow_or_final_used", False)) or bool(phase4_gate.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing count-matched rework because an input reports shadow/final use.")
    if bool(schema_gate.get("reads_existing_xrt_cubes", False)) or bool(phase4_gate.get("reads_existing_xrt_cubes", False)):
        raise RuntimeError("Refusing count-matched rework because an input reports existing XRT cube reads.")
    return schema_gate, phase4_gate, total_count_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run count-matched total-count confounding rework gate for v8A medium development features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_event_to_feature")
    parser.add_argument("--phase4-model-dir", default="results/accuracy_v3/v8a_medium_development_model")
    parser.add_argument("--total-count-diagnostic-dir", default="results/accuracy_v3/v8a_medium_total_count_control_diagnostic")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_count_matched_rework")
    parser.add_argument("--match-tolerance-total-count-norm", type=float, default=PRIMARY_MATCH_TOLERANCE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    phase4_dir = project_root / args.phase4_model_dir
    total_count_dir = project_root / args.total_count_diagnostic_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate, phase4_gate, total_count_diagnostic = load_required_inputs(input_dir, phase4_dir, total_count_dir)
    sk = require_sklearn()
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)
    matched = build_count_matched_frame(frame, float(args.match_tolerance_total_count_norm))
    if matched.empty:
        raise RuntimeError("No count-matched samples were produced.")
    train = split_source(matched, "train", "custom_diffraction_on")
    validation = split_source(matched, "validation", "custom_diffraction_on")
    holdout = split_source(matched, "stress_holdout", "custom_diffraction_on")
    pairs = split_pairs(matched)
    count_gaps = count_gap_summary(matched)

    def raw_builder(cols: list[str]) -> Callable[[pd.DataFrame], np.ndarray]:
        return lambda table: table[cols].fillna(0.0).to_numpy(dtype=np.float64)

    models = [
        (
            "LogisticCountMatchedMain",
            "main",
            sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=9911)),
            raw_builder(main_cols),
            False,
        ),
        (
            "ExtraTreesCountMatchedMain",
            "main",
            sk["ExtraTreesClassifier"](n_estimators=500, random_state=9912, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            raw_builder(main_cols),
            False,
        ),
        (
            "ExtraTreesCountMatchedTotalCountOnly",
            "control",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9913, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            raw_builder(total_count_cols),
            False,
        ),
        (
            "ExtraTreesCountMatchedOverlapOnly",
            "control",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9914, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            raw_builder(overlap_cols),
            False,
        ),
        (
            "ExtraTreesCountMatchedThicknessPoseOnly",
            "control",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9915, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            raw_builder(thickness_pose_cols),
            False,
        ),
        (
            "ExtraTreesCountMatchedShuffledLabels",
            "control",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9916, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            raw_builder(main_cols),
            True,
        ),
    ]
    summary_rows: list[dict[str, Any]] = []
    decision_tables: list[pd.DataFrame] = []
    group_tables: list[pd.DataFrame] = []
    sweep_tables: list[pd.DataFrame] = []
    calibration_tables: list[pd.DataFrame] = []
    for method, family, estimator, builder, shuffle_labels in models:
        summaries, decisions, groups, sweeps, calibrations = evaluate_train_validation_holdout(
            method,
            family,
            estimator,
            train,
            validation,
            holdout,
            builder,
            sk,
            shuffle_labels=shuffle_labels,
        )
        summary_rows.extend(summaries)
        decision_tables.extend(table for table in decisions if not table.empty)
        group_tables.extend(table for table in groups if not table.empty)
        sweep_tables.extend(table for table in sweeps if not table.empty)
        calibration_tables.extend(table for table in calibrations if not table.empty)

    # Source-off leakage remains a separate control because source-off rows do not share the
    # source-on stress labels used for count matching.
    source_off_train = split_source(frame, "train", "custom_diffraction_off")
    source_off_validation = split_source(frame, "validation", "custom_diffraction_off")
    source_off_holdout = split_source(frame, "stress_holdout", "custom_diffraction_off")
    source_off_train = source_off_train.assign(match_pair_id="source_off_control", match_delta_total_count_norm=0.0)
    source_off_validation = source_off_validation.assign(match_pair_id="source_off_control", match_delta_total_count_norm=0.0)
    source_off_holdout = source_off_holdout.assign(match_pair_id="source_off_control", match_delta_total_count_norm=0.0)
    source_summaries, source_decisions, source_groups, source_sweeps, source_calibrations = evaluate_train_validation_holdout(
        "ExtraTreesSourceOffLeakage",
        "control",
        sk["ExtraTreesClassifier"](n_estimators=300, random_state=9917, class_weight="balanced", max_features="sqrt", n_jobs=-1),
        source_off_train,
        source_off_validation,
        source_off_holdout,
        raw_builder(main_cols),
        sk,
    )
    summary_rows.extend(source_summaries)
    decision_tables.extend(table for table in source_decisions if not table.empty)
    group_tables.extend(table for table in source_groups if not table.empty)
    sweep_tables.extend(table for table in source_sweeps if not table.empty)
    calibration_tables.extend(table for table in source_calibrations if not table.empty)

    selection = pd.DataFrame(summary_rows)
    decisions = pd.concat(decision_tables, ignore_index=True) if decision_tables else pd.DataFrame()
    group_recalls = pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame()
    threshold_sweeps = pd.concat(sweep_tables, ignore_index=True) if sweep_tables else pd.DataFrame()
    calibration_bins = pd.concat(calibration_tables, ignore_index=True) if calibration_tables else pd.DataFrame()

    residual_rows: list[dict[str, Any]] = []
    residual_specs = [
        ("residualized_total_count_norm", [MATCH_COLUMN]),
        ("residualized_all_total_count_controls", total_count_cols),
    ]
    for residual_name, controls in residual_specs:
        coefficients = fit_residualizer(train, main_cols, controls)
        builder = lambda table, controls=controls, coefficients=coefficients: apply_residualizer(table, main_cols, controls, coefficients)
        for method, estimator in [
            (
                "LogisticResidualizedMain",
                sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=9921)),
            ),
            (
                "ExtraTreesResidualizedMain",
                sk["ExtraTreesClassifier"](n_estimators=500, random_state=9922, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            ),
        ]:
            summaries, _, _, _, _ = evaluate_train_validation_holdout(
                method,
                "residualized_sensitivity",
                estimator,
                train,
                validation,
                holdout,
                builder,
                sk,
            )
            for row in summaries:
                row["residualization"] = residual_name
                residual_rows.append(row)
    residualized = pd.DataFrame(residual_rows)

    tolerance_rows = []
    for tolerance in SENSITIVITY_TOLERANCES:
        tolerance_frame = build_count_matched_frame(frame, tolerance)
        tolerance_pairs = split_pairs(tolerance_frame)
        tolerance_rows.append(
            {
                "match_tolerance_total_count_norm": tolerance,
                "train_pairs": tolerance_pairs["train"],
                "validation_pairs": tolerance_pairs["validation"],
                "stress_holdout_pairs": tolerance_pairs["stress_holdout"],
                "max_pair_delta_total_count_norm": float(tolerance_frame["match_delta_total_count_norm"].max()) if not tolerance_frame.empty else 0.0,
            }
        )
    tolerance_curve = pd.DataFrame(tolerance_rows)

    main_selection = selection[selection["method"].isin(["LogisticCountMatchedMain", "ExtraTreesCountMatchedMain"]) & selection["eval_split"].eq("validation")]
    selected_main = main_selection.sort_values(
        ["hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall", "worst_stress_label_hm_min_recall", "accuracy"],
        ascending=False,
    ).iloc[0].to_dict()
    selected_method = str(selected_main["method"])
    validation_main = value_for(selection, selected_method, "validation", "hm_min_recall")
    holdout_main = value_for(selection, selected_method, "stress_holdout", "hm_min_recall")
    worst_thickness = min(
        value_for(selection, selected_method, "validation", "worst_thickness_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_thickness_hm_min_recall"),
    )
    worst_pose = min(
        value_for(selection, selected_method, "validation", "worst_pose_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_pose_hm_min_recall"),
    )
    worst_stress = min(
        value_for(selection, selected_method, "validation", "worst_stress_label_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_stress_label_hm_min_recall"),
    )
    main_ece_max = max_value_for(selection, selected_method, "expected_calibration_error")
    total_count_max = max_value_for(selection, "ExtraTreesCountMatchedTotalCountOnly", "hm_min_recall")
    overlap_max = max_value_for(selection, "ExtraTreesCountMatchedOverlapOnly", "hm_min_recall")
    thickness_pose_max = max_value_for(selection, "ExtraTreesCountMatchedThicknessPoseOnly", "hm_min_recall")
    shuffled_max = max_value_for(selection, "ExtraTreesCountMatchedShuffledLabels", "hm_min_recall")
    source_off_max = max_value_for(selection, "ExtraTreesSourceOffLeakage", "hm_min_recall")
    max_delta = float(matched["match_delta_total_count_norm"].max())
    pass_items = {
        "train_pair_support": pairs["train"] >= THRESHOLDS["train_pairs_min"],
        "validation_pair_support": pairs["validation"] >= THRESHOLDS["validation_pairs_min"],
        "stress_holdout_pair_support": pairs["stress_holdout"] >= THRESHOLDS["stress_holdout_pairs_min"],
        "match_delta_below_tolerance": max_delta <= THRESHOLDS["match_delta_total_count_norm_max"],
        "validation_main_hm_min_recall": validation_main >= THRESHOLDS["main_hm_min_recall_min"],
        "stress_holdout_main_hm_min_recall": holdout_main >= THRESHOLDS["stress_holdout_main_hm_min_recall_min"],
        "worst_thickness_hm_min_recall": worst_thickness >= THRESHOLDS["worst_thickness_hm_min_recall_min"],
        "worst_pose_hm_min_recall": worst_pose >= THRESHOLDS["worst_pose_hm_min_recall_min"],
        "worst_stress_label_hm_min_recall": worst_stress >= THRESHOLDS["worst_stress_label_hm_min_recall_min"],
        "total_count_only_below_ceiling": total_count_max < THRESHOLDS["total_count_only_hm_min_recall_max"],
        "overlap_only_below_ceiling": overlap_max < THRESHOLDS["overlap_only_hm_min_recall_max"],
        "thickness_pose_below_ceiling": thickness_pose_max < THRESHOLDS["thickness_pose_hm_min_recall_max"],
        "shuffled_label_below_ceiling": shuffled_max < THRESHOLDS["shuffled_label_hm_min_recall_max"],
        "source_off_below_ceiling": source_off_max < THRESHOLDS["source_off_hm_min_recall_max"],
        "main_minus_total_count_margin": validation_main - total_count_max >= THRESHOLDS["main_minus_total_count_hm_margin_min"],
        "main_minus_source_off_margin": validation_main - source_off_max >= THRESHOLDS["main_minus_source_off_hm_margin_min"],
        "calibration_ece_below_ceiling": main_ece_max <= THRESHOLDS["expected_calibration_error_max"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/train_v8a_medium_count_matched_rework.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_medium_count_matched_total_count_rework_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": "development-only count-matched total-count-confounding rework; not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD",
        "input_dir": args.input_dir,
        "phase4_model_dir": args.phase4_model_dir,
        "total_count_diagnostic_dir": args.total_count_diagnostic_dir,
        "match_column": MATCH_COLUMN,
        "match_tolerance_total_count_norm": float(args.match_tolerance_total_count_norm),
        "matched_pair_counts": pairs,
        "gate_passed": gate_passed,
        "decision": "count_matched_total_count_rework_passed_continue_development_only_review" if gate_passed else "count_matched_total_count_rework_still_blocked",
        "selected_main_model": {
            "method": selected_method,
            "threshold": float(selected_main["threshold"]),
            "feature_count": int(selected_main["feature_count"]),
        },
        "validation_main_hm_min_recall": validation_main,
        "stress_holdout_main_hm_min_recall": holdout_main,
        "worst_thickness_hm_min_recall": worst_thickness,
        "worst_pose_hm_min_recall": worst_pose,
        "worst_stress_label_hm_min_recall": worst_stress,
        "main_expected_calibration_error_max": main_ece_max,
        "total_count_only_max_hm_min_recall": total_count_max,
        "overlap_only_max_hm_min_recall": overlap_max,
        "thickness_pose_max_hm_min_recall": thickness_pose_max,
        "shuffled_label_max_hm_min_recall": shuffled_max,
        "source_off_max_hm_min_recall": source_off_max,
        "main_minus_total_count_hm_margin": validation_main - total_count_max,
        "main_minus_source_off_hm_margin": validation_main - source_off_max,
        "max_pair_delta_total_count_norm": max_delta,
        "count_gap_summary": count_gaps.to_dict(orient="records"),
        "residualized_sensitivity_best": residualized.sort_values("hm_min_recall", ascending=False).head(8).to_dict(orient="records") if not residualized.empty else [],
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "input_gate_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "phase4_gate": phase4_gate.get("decision"),
            "total_count_diagnostic": total_count_diagnostic.get("decision"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest = {
        "generated_by": gate["generated_by"],
        "generated_at_utc": gate["generated_at_utc"],
        "protocol_name": gate["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "match_column": MATCH_COLUMN,
        "match_tolerance_total_count_norm": float(args.match_tolerance_total_count_norm),
        "matched_sample_count": int(len(matched)),
        "matched_pair_counts": pairs,
        "gate_file": "v8a_medium_count_matched_rework_gate.json",
    }

    matched.to_csv(output_dir / "v8a_medium_count_matched_features.csv", index=False, lineterminator="\n")
    selection.to_csv(output_dir / "v8a_medium_count_matched_model_selection.csv", index=False, lineterminator="\n")
    decisions.to_csv(output_dir / "v8a_medium_count_matched_decisions.csv", index=False, lineterminator="\n")
    group_recalls.to_csv(output_dir / "v8a_medium_count_matched_group_recalls.csv", index=False, lineterminator="\n")
    threshold_sweeps.to_csv(output_dir / "v8a_medium_count_matched_threshold_sweep.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "v8a_medium_count_matched_calibration_bins.csv", index=False, lineterminator="\n")
    count_gaps.to_csv(output_dir / "v8a_medium_count_matched_count_gap_summary.csv", index=False, lineterminator="\n")
    tolerance_curve.to_csv(output_dir / "v8a_medium_count_matched_tolerance_curve.csv", index=False, lineterminator="\n")
    residualized.to_csv(output_dir / "v8a_medium_count_matched_residualized_sensitivity.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_medium_count_matched_rework_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_medium_count_matched_rework_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, selection, count_gaps, residualized)
    print(
        "decision={decision} gate_passed={passed} pairs={train}/{validation}/{holdout} main={main:.4f}/{holdout_main:.4f} total_count_max={total:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            train=pairs["train"],
            validation=pairs["validation"],
            holdout=pairs["stress_holdout"],
            main=validation_main,
            holdout_main=holdout_main,
            total=total_count_max,
        )
    )


if __name__ == "__main__":
    main()
