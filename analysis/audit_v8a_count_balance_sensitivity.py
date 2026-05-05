from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_medium_count_matched_rework import (
    MATCH_COLUMN,
    THRESHOLDS,
    expected_calibration_error,
    magnetite_probability,
    metrics_from_predictions,
    threshold_sweep,
)


STRATEGIES = [
    {"strategy": "fixed_bin_width_0p001", "kind": "fixed_bin", "bin_width": 0.001, "exact_columns": []},
    {"strategy": "fixed_bin_width_0p002", "kind": "fixed_bin", "bin_width": 0.002, "exact_columns": []},
    {"strategy": "fixed_bin_width_0p003", "kind": "fixed_bin", "bin_width": 0.003, "exact_columns": []},
    {"strategy": "quantile_bins_12", "kind": "quantile_bin", "bin_count": 12, "exact_columns": []},
    {"strategy": "quantile_bins_16", "kind": "quantile_bin", "bin_count": 16, "exact_columns": []},
    {"strategy": "quantile_bins_20", "kind": "quantile_bin", "bin_count": 20, "exact_columns": []},
    {"strategy": "fixed_bin_width_0p005_by_thickness_pose", "kind": "fixed_bin", "bin_width": 0.005, "exact_columns": ["thickness_mm", "pose_index"]},
]


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (dict, list, tuple, set, np.ndarray)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    return value


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/audit_v8a_count_balance_sensitivity.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def add_count_bins(frame: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    if strategy["kind"] == "fixed_bin":
        width = float(strategy["bin_width"])
        result["count_balance_bin"] = np.floor(result[MATCH_COLUMN] / width).astype(int).astype(str)
        return result
    if strategy["kind"] == "quantile_bin":
        frames = []
        for split, split_frame in result.groupby("split", sort=False):
            split_frame = split_frame.copy()
            quantiles = np.quantile(split_frame[MATCH_COLUMN].to_numpy(dtype=np.float64), np.linspace(0.0, 1.0, int(strategy["bin_count"]) + 1))
            bins = np.unique(quantiles)
            if len(bins) < 2:
                split_frame["count_balance_bin"] = "single"
            else:
                split_frame["count_balance_bin"] = pd.cut(split_frame[MATCH_COLUMN], bins=bins, include_lowest=True, duplicates="drop").astype(str)
            frames.append(split_frame)
        return pd.concat(frames, ignore_index=True)
    raise ValueError(f"Unknown strategy kind: {strategy['kind']}")


def build_balanced_subset(frame: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    source_on = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    source_on = add_count_bins(source_on, strategy)
    rows = []
    exact_columns = list(strategy.get("exact_columns", []))
    group_columns = ["split", "stress_label", "count_balance_bin"] + exact_columns
    for keys, group in source_on.groupby(group_columns, sort=True, observed=True):
        hematite = group[group["material"].astype(str).eq("Hematite")].sort_values(MATCH_COLUMN)
        magnetite = group[group["material"].astype(str).eq("Magnetite")].sort_values(MATCH_COLUMN)
        pair_count = min(len(hematite), len(magnetite))
        if pair_count <= 0:
            continue
        for pair_index, (_, h_row) in enumerate(hematite.head(pair_count).iterrows(), start=1):
            row = h_row.copy()
            row["match_pair_id"] = f"{strategy['strategy']}|{keys}|pair{pair_index:03d}"
            row["match_delta_total_count_norm"] = 0.0
            rows.append(row)
        for pair_index, (_, m_row) in enumerate(magnetite.head(pair_count).iterrows(), start=1):
            row = m_row.copy()
            row["match_pair_id"] = f"{strategy['strategy']}|{keys}|pair{pair_index:03d}"
            row["match_delta_total_count_norm"] = 0.0
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[frame["split"].astype(str).eq(split)].copy()


def evaluate_estimator(
    method: str,
    estimator: Any,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    eval_split: str,
    feature_cols: list[str],
    selected_threshold: float | None = None,
) -> tuple[dict[str, Any], float]:
    if train.empty or evaluate.empty or not feature_cols:
        return {"method": method, "eval_split": eval_split, "hm_min_recall": 0.0, "samples": int(len(evaluate)), "status": "not_evaluable"}, 0.5
    model = deepcopy(estimator)
    model.fit(train[feature_cols].fillna(0.0).to_numpy(dtype=np.float64), train["material"].astype(str).to_numpy())
    probabilities = magnetite_probability(model, evaluate[feature_cols].fillna(0.0).to_numpy(dtype=np.float64))
    sweep = threshold_sweep(evaluate, probabilities, method, eval_split)
    if selected_threshold is None:
        ranked = sweep.assign(threshold_distance_to_0p5=(sweep["threshold"] - 0.5).abs())
        threshold = float(
            ranked.sort_values(["hm_min_recall", "accuracy", "threshold_distance_to_0p5"], ascending=[False, False, True])
            .iloc[0]["threshold"]
        )
    else:
        threshold = float(selected_threshold)
    predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite").astype(str)
    y_true = evaluate["material"].astype(str).to_numpy()
    y_binary = (y_true == "Magnetite").astype(int)
    ece, _ = expected_calibration_error(y_binary, probabilities)
    summary = {
        "method": method,
        "eval_split": eval_split,
        "threshold": threshold,
        "samples": int(len(evaluate)),
        "status": "evaluated",
        **metrics_from_predictions(y_true, predictions),
        "expected_calibration_error": ece,
    }
    return summary, threshold


def standardized_gap(frame: pd.DataFrame, split: str) -> float:
    subset = frame[frame["split"].astype(str).eq(split)]
    hematite = subset[subset["material"].astype(str).eq("Hematite")][MATCH_COLUMN].to_numpy(dtype=np.float64)
    magnetite = subset[subset["material"].astype(str).eq("Magnetite")][MATCH_COLUMN].to_numpy(dtype=np.float64)
    if len(hematite) == 0 or len(magnetite) == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(hematite) + np.var(magnetite)) + 1e-12)
    return float(abs(float(np.mean(magnetite) - np.mean(hematite))) / pooled)


def evaluate_strategy(frame: pd.DataFrame, strategy: dict[str, Any], sk: dict[str, Any], main_cols: list[str], total_count_cols: list[str]) -> dict[str, Any]:
    balanced = build_balanced_subset(frame, strategy)
    train = split_frame(balanced, "train")
    validation = split_frame(balanced, "validation")
    holdout = split_frame(balanced, "stress_holdout")
    pair_counts = {
        "train_pairs": int(train["match_pair_id"].nunique()) if not train.empty else 0,
        "validation_pairs": int(validation["match_pair_id"].nunique()) if not validation.empty else 0,
        "stress_holdout_pairs": int(holdout["match_pair_id"].nunique()) if not holdout.empty else 0,
    }
    model_specs = [
        (
            "LogisticMain",
            sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=9931)),
            main_cols,
        ),
        (
            "ExtraTreesMain",
            sk["ExtraTreesClassifier"](n_estimators=350, random_state=9932, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=350, random_state=9933, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
        ),
    ]
    summaries: list[dict[str, Any]] = []
    for method, estimator, cols in model_specs:
        validation_summary, threshold = evaluate_estimator(method, estimator, train, validation, "validation", cols)
        holdout_summary, _ = evaluate_estimator(method, estimator, train, holdout, "stress_holdout", cols, selected_threshold=threshold)
        summaries.extend([validation_summary, holdout_summary])
    summary_frame = pd.DataFrame(summaries)
    main_frame = summary_frame[summary_frame["method"].isin(["LogisticMain", "ExtraTreesMain"])]
    total_frame = summary_frame[summary_frame["method"].eq("ExtraTreesTotalCountOnly")]
    result = {
        "strategy": strategy["strategy"],
        "kind": strategy["kind"],
        "bin_width": strategy.get("bin_width"),
        "bin_count": strategy.get("bin_count"),
        "exact_columns": ",".join(strategy.get("exact_columns", [])),
        **pair_counts,
        "validation_count_gap_standardized": standardized_gap(balanced, "validation"),
        "stress_holdout_count_gap_standardized": standardized_gap(balanced, "stress_holdout"),
        "best_main_validation_hm_min_recall": float(main_frame[main_frame["eval_split"].eq("validation")]["hm_min_recall"].max()) if not main_frame.empty else 0.0,
        "best_main_stress_holdout_hm_min_recall": float(main_frame[main_frame["eval_split"].eq("stress_holdout")]["hm_min_recall"].max()) if not main_frame.empty else 0.0,
        "total_count_validation_hm_min_recall": float(total_frame[total_frame["eval_split"].eq("validation")]["hm_min_recall"].max()) if not total_frame.empty else 0.0,
        "total_count_stress_holdout_hm_min_recall": float(total_frame[total_frame["eval_split"].eq("stress_holdout")]["hm_min_recall"].max()) if not total_frame.empty else 0.0,
    }
    result["support_pass"] = bool(
        result["train_pairs"] >= THRESHOLDS["train_pairs_min"]
        and result["validation_pairs"] >= THRESHOLDS["validation_pairs_min"]
        and result["stress_holdout_pairs"] >= THRESHOLDS["stress_holdout_pairs_min"]
    )
    result["main_signal_pass"] = bool(
        result["best_main_validation_hm_min_recall"] >= THRESHOLDS["main_hm_min_recall_min"]
        and result["best_main_stress_holdout_hm_min_recall"] >= THRESHOLDS["stress_holdout_main_hm_min_recall_min"]
    )
    result["total_count_control_pass"] = bool(
        max(result["total_count_validation_hm_min_recall"], result["total_count_stress_holdout_hm_min_recall"])
        < THRESHOLDS["total_count_only_hm_min_recall_max"]
    )
    result["strategy_passed"] = bool(result["support_pass"] and result["main_signal_pass"] and result["total_count_control_pass"])
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit count-balance sensitivity for v8A medium development features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_event_to_feature")
    parser.add_argument("--count-matched-dir", default="results/accuracy_v3/v8a_medium_count_matched_rework")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_count_balance_sensitivity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    count_matched_gate = load_json(project_root / args.count_matched_dir / "v8a_medium_count_matched_rework_gate.json")
    if not bool(schema_gate.get("gate_passed", False)):
        raise RuntimeError(f"Event schema gate did not pass: {schema_gate.get('decision')}")
    if bool(schema_gate.get("shadow_or_final_used", False)) or bool(count_matched_gate.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing count-balance sensitivity because an input reports shadow/final use.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, total_count_cols, _, _ = feature_sets(frame)
    sk = require_sklearn()
    rows = [evaluate_strategy(frame, strategy, sk, main_cols, total_count_cols) for strategy in STRATEGIES]
    result_frame = pd.DataFrame(rows).sort_values(["strategy_passed", "support_pass", "total_count_control_pass", "validation_pairs"], ascending=[False, False, False, False])
    any_passed = bool(result_frame["strategy_passed"].any())
    any_signal_without_support = bool((result_frame["main_signal_pass"] & result_frame["total_count_control_pass"] & ~result_frame["support_pass"]).any())
    diagnostic = {
        "generated_by": "analysis/audit_v8a_count_balance_sensitivity.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": "development-only count-balance sensitivity audit; not product accuracy or shadow/final validation",
        "input_dir": args.input_dir,
        "count_matched_dir": args.count_matched_dir,
        "gate_passed": any_passed,
        "decision": "count_balance_strategy_available_for_development_only_retest" if any_passed else "existing_medium_outputs_need_count_overlap_extension",
        "any_signal_without_support": any_signal_without_support,
        "thresholds": {
            "train_pairs_min": THRESHOLDS["train_pairs_min"],
            "validation_pairs_min": THRESHOLDS["validation_pairs_min"],
            "stress_holdout_pairs_min": THRESHOLDS["stress_holdout_pairs_min"],
            "main_hm_min_recall_min": THRESHOLDS["main_hm_min_recall_min"],
            "total_count_only_hm_min_recall_max": THRESHOLDS["total_count_only_hm_min_recall_max"],
        },
        "strategy_count": int(len(result_frame)),
        "passed_strategy_count": int(result_frame["strategy_passed"].sum()),
        "best_rows": json_clean(result_frame.head(5).to_dict(orient="records")),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    result_frame.to_csv(output_dir / "v8a_count_balance_sensitivity_summary.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_count_balance_sensitivity_gate.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# v8A count-balance sensitivity audit",
        "",
        f"Generated: {diagnostic['generated_at_utc']}",
        "",
        "Scope: development-only sensitivity audit over already generated medium features. This does not run Geant4 and does not open shadow/final.",
        "",
        f"- Decision: `{diagnostic['decision']}`",
        f"- Gate passed: `{str(diagnostic['gate_passed']).lower()}`",
        f"- Any signal without support: `{str(any_signal_without_support).lower()}`",
        "",
        markdown_table(
            result_frame,
            [
                "strategy",
                "train_pairs",
                "validation_pairs",
                "stress_holdout_pairs",
                "best_main_validation_hm_min_recall",
                "best_main_stress_holdout_hm_min_recall",
                "total_count_validation_hm_min_recall",
                "total_count_stress_holdout_hm_min_recall",
                "support_pass",
                "total_count_control_pass",
                "strategy_passed",
            ],
        ),
        "",
    ]
    (output_dir / "v8a_count_balance_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        "decision={decision} gate_passed={passed} passed_strategies={count}".format(
            decision=diagnostic["decision"],
            passed=str(any_passed).lower(),
            count=diagnostic["passed_strategy_count"],
        )
    )


if __name__ == "__main__":
    main()
