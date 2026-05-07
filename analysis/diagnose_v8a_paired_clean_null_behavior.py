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

from train_v8a_event_feature_smoke import feature_sets, load_json, pair_recalls


CLAIM_SCOPE = (
    "development-only paired-clean shuffled-label/null diagnosis for v8A H/M sidecar features; "
    "not product accuracy, hardware validation, shadow/final validation, training evidence, or manuscript-grade powder XRD"
)

PRIMARY_MODE = "paired_nuisance_balanced_orientation"
SECONDARY_MODE = "seed_block_random_balanced_orientation"
STRICT_POSE_COUNT_MODE = "seed_block_pose_count_strict_balanced_orientation"
STRICT_FULL_CELL_MODE = "seed_block_thickness_pose_count_strict_balanced_orientation"
NULL_MODES = (PRIMARY_MODE, SECONDARY_MODE)
SUPPORTED_NULL_MODES = (PRIMARY_MODE, SECONDARY_MODE, STRICT_POSE_COUNT_MODE, STRICT_FULL_CELL_MODE)

THRESHOLDS = {
    "shuffle_seed_count_min": 60,
    "effective_shuffle_fraction_min": 0.45,
    "effective_shuffle_fraction_max": 0.55,
    "null_hm_min_recall_p95_ceiling": 0.55,
    "null_hm_min_recall_single_seed_max": 0.65,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
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


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite").astype(str)
    recalls = pair_recalls(y_true.astype(str), predictions)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_true == predictions)) if len(y_true) else 0.0,
        "hematite_recall": recalls["Hematite"],
        "magnetite_recall": recalls["Magnetite"],
        "hm_min_recall": float(min(recalls.values())),
    }


def selected_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        metric = threshold_metrics(y_true, probabilities, float(threshold))
        metric["threshold_distance_to_0p5"] = abs(float(threshold) - 0.5)
        rows.append(metric)
    selected = sorted(rows, key=lambda item: (item["hm_min_recall"], item["accuracy"], -item["threshold_distance_to_0p5"]), reverse=True)[0]
    return float(selected["threshold"]), selected


def model_specs(sk: dict[str, Any], seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "Logistic",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ),
        (
            "ExtraTrees",
            sk["ExtraTreesClassifier"](
                n_estimators=160,
                random_state=seed,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
        ),
    ]


def magnetite_probability(estimator: Any, x_values: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x_values)
    classes = [str(item) for item in estimator.classes_]
    if "Magnetite" not in classes:
        return np.zeros(len(x_values), dtype=np.float64)
    return probabilities[:, classes.index("Magnetite")].astype(np.float64)


def pair_table(train: pd.DataFrame) -> pd.DataFrame:
    required = ["clean_match_pair_id", "seed_block", "thickness_mm", "pose_index", "count_target_bin"]
    missing = [col for col in required if col not in train.columns]
    if missing:
        raise RuntimeError(f"Paired-clean null requires columns: {missing}")
    rows: list[dict[str, Any]] = []
    for pair_id, group in train.groupby("clean_match_pair_id", sort=True):
        materials = sorted(group["material"].astype(str).tolist())
        if len(group) != 2 or materials != ["Hematite", "Magnetite"]:
            raise RuntimeError(f"Pair {pair_id} is not exactly one H/M row.")
        first = group.iloc[0]
        rows.append(
            {
                "clean_match_pair_id": str(pair_id),
                "seed_block": str(first["seed_block"]),
                "thickness_mm": float(first["thickness_mm"]),
                "pose_index": int(first["pose_index"]),
                "count_target_bin": str(first["count_target_bin"]),
            }
        )
    return pd.DataFrame(rows)


def orientation_map_for_mode(pairs: pd.DataFrame, seed: int, mode: str) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    orientations: dict[str, int] = {}
    if mode == "seed_block_random_balanced_orientation":
        for _, group in pairs.groupby("seed_block", sort=True):
            pair_ids = group["clean_match_pair_id"].astype(str).to_numpy()
            if len(pair_ids) % 2 != 0:
                raise RuntimeError("Seed-block balanced orientation requires an even pair count per seed block.")
            swap_count = len(pair_ids) // 2
            swap_ids = set(rng.choice(pair_ids, size=swap_count, replace=False).tolist())
            for pair_id in pair_ids:
                orientations[str(pair_id)] = -1 if pair_id in swap_ids else 1
        return orientations
    if mode == STRICT_POSE_COUNT_MODE:
        for _, group in pairs.groupby(["seed_block", "pose_index", "count_target_bin"], sort=True):
            pair_ids = group["clean_match_pair_id"].astype(str).to_numpy()
            if len(pair_ids) % 2 != 0:
                raise RuntimeError(
                    "Strict seed-block/pose/count-bin balanced orientation requires an even pair count per cell."
                )
            swap_count = len(pair_ids) // 2
            swap_ids = set(rng.choice(pair_ids, size=swap_count, replace=False).tolist())
            for pair_id in pair_ids:
                orientations[str(pair_id)] = -1 if pair_id in swap_ids else 1
        return orientations
    if mode == STRICT_FULL_CELL_MODE:
        for _, group in pairs.groupby(["seed_block", "thickness_mm", "pose_index", "count_target_bin"], sort=True):
            pair_ids = group["clean_match_pair_id"].astype(str).to_numpy()
            if len(pair_ids) % 2 != 0:
                raise RuntimeError(
                    "Strict seed-block/thickness/pose/count-bin balanced orientation requires an even pair count per cell."
                )
            swap_count = len(pair_ids) // 2
            swap_ids = set(rng.choice(pair_ids, size=swap_count, replace=False).tolist())
            for pair_id in pair_ids:
                orientations[str(pair_id)] = -1 if pair_id in swap_ids else 1
        return orientations
    if mode != PRIMARY_MODE:
        raise ValueError(f"Unknown paired-clean null mode: {mode}")

    for seed_block, group in pairs.groupby("seed_block", sort=True):
        thickness_values = sorted(group["thickness_mm"].unique().tolist())
        pose_values = sorted(group["pose_index"].unique().tolist())
        count_values = sorted(group["count_target_bin"].unique().tolist())
        t_perm = dict(zip(thickness_values, rng.permutation(len(thickness_values)).tolist()))
        p_perm = dict(zip(pose_values, rng.permutation(len(pose_values)).tolist()))
        c_perm = dict(zip(count_values, rng.permutation(len(count_values)).tolist()))
        block_flip = int(rng.integers(0, 2))
        for _, row in group.iterrows():
            parity = (
                t_perm[float(row["thickness_mm"])]
                + p_perm[int(row["pose_index"])]
                + c_perm[str(row["count_target_bin"])]
                + block_flip
            ) % 2
            orientations[str(row["clean_match_pair_id"])] = -1 if parity else 1
    return orientations


def apply_pair_orientations(train: pd.DataFrame, orientations: dict[str, int]) -> tuple[np.ndarray, float, dict[str, float]]:
    labels = train["material"].astype(str).to_numpy().copy()
    pseudo = pd.Series(labels.copy(), index=train.index, dtype=object)
    orientation_by_row = pd.Series(0, index=train.index, dtype=int)
    for pair_id, group_index in train.groupby("clean_match_pair_id", sort=True).groups.items():
        sign = int(orientations[str(pair_id)])
        group_index = list(group_index)
        current = pseudo.loc[group_index].astype(str).to_numpy()
        if sign < 0:
            pseudo.loc[group_index] = current[::-1]
        orientation_by_row.loc[group_index] = sign
    shuffled = pseudo.loc[train.index].astype(str).to_numpy()
    diagnostics = orientation_balance_diagnostics(train, orientation_by_row)
    return shuffled, float(np.mean(shuffled != labels)) if len(labels) else 0.0, diagnostics


def orientation_balance_diagnostics(train: pd.DataFrame, orientation_by_row: pd.Series) -> dict[str, float]:
    frame = train[["clean_match_pair_id", "seed_block", "thickness_mm", "pose_index", "count_target_bin"]].copy()
    frame["orientation"] = orientation_by_row.loc[train.index].to_numpy()
    pair_frame = frame.drop_duplicates("clean_match_pair_id").copy()
    diagnostics: dict[str, float] = {}
    for name, cols in {
        "overall": [],
        "seed_block": ["seed_block"],
        "seed_block_pose": ["seed_block", "pose_index"],
        "seed_block_count_bin": ["seed_block", "count_target_bin"],
        "seed_block_thickness": ["seed_block", "thickness_mm"],
        "seed_block_thickness_pose": ["seed_block", "thickness_mm", "pose_index"],
        "seed_block_thickness_count_bin": ["seed_block", "thickness_mm", "count_target_bin"],
        "seed_block_pose_count_bin": ["seed_block", "pose_index", "count_target_bin"],
        "seed_block_thickness_pose_count_bin": ["seed_block", "thickness_mm", "pose_index", "count_target_bin"],
    }.items():
        if not cols:
            sums = [float(pair_frame["orientation"].sum())]
        else:
            sums = pair_frame.groupby(cols, sort=True)["orientation"].sum().astype(float).tolist()
        diagnostics[f"{name}_max_abs_orientation_sum"] = float(max(abs(item) for item in sums)) if sums else 0.0
    return diagnostics


def evaluate_paired_null(
    frame: pd.DataFrame,
    main_cols: list[str],
    seeds: list[int],
    sk: dict[str, Any],
    null_modes: tuple[str, ...] = NULL_MODES,
) -> pd.DataFrame:
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    validation = frame[frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    holdout = frame[frame["split"].astype(str).eq("stress_holdout") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("Train/validation/stress_holdout source-on rows are required for paired-clean null diagnosis.")
    pairs = pair_table(train)
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    eval_frames = {"validation": validation, "stress_holdout": holdout}
    rows: list[dict[str, Any]] = []
    for mode in null_modes:
        for seed in seeds:
            orientations = orientation_map_for_mode(pairs, seed, mode)
            y_train, effective_shuffle_fraction, orientation_diag = apply_pair_orientations(train, orientations)
            for model_name, estimator in model_specs(sk, seed):
                fitted = deepcopy(estimator)
                fitted.fit(x_train, y_train)
                validation_threshold = 0.5
                for eval_split, eval_frame in eval_frames.items():
                    x_eval = eval_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
                    y_true = eval_frame["material"].astype(str).to_numpy()
                    prob = magnetite_probability(fitted, x_eval)
                    fixed = threshold_metrics(y_true, prob, 0.5)
                    if eval_split == "validation":
                        validation_threshold, selected = selected_threshold(y_true, prob)
                    else:
                        selected = threshold_metrics(y_true, prob, validation_threshold)
                    for policy, metrics in [("fixed_0p5", fixed), ("validation_selected", selected)]:
                        rows.append(
                            {
                                "shuffle_mode": mode,
                                "shuffle_seed": seed,
                                "model": model_name,
                                "eval_split": eval_split,
                                "threshold_policy": policy,
                                "effective_shuffle_fraction": effective_shuffle_fraction,
                                **orientation_diag,
                                **{key: value for key, value in metrics.items() if key != "threshold_distance_to_0p5"},
                            }
                        )
    return pd.DataFrame(rows)


def summarize_null(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["shuffle_mode", "model", "eval_split", "threshold_policy"], sort=True)
    result = []
    for keys, group in grouped:
        shuffle_mode, model, eval_split, threshold_policy = keys
        result.append(
            {
                "shuffle_mode": shuffle_mode,
                "model": model,
                "eval_split": eval_split,
                "threshold_policy": threshold_policy,
                "seed_count": int(group["shuffle_seed"].nunique()),
                "hm_min_recall_mean": float(group["hm_min_recall"].mean()),
                "hm_min_recall_p95": float(group["hm_min_recall"].quantile(0.95)),
                "hm_min_recall_max": float(group["hm_min_recall"].max()),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_max": float(group["accuracy"].max()),
                "effective_shuffle_fraction_min": float(group["effective_shuffle_fraction"].min()),
                "effective_shuffle_fraction_max": float(group["effective_shuffle_fraction"].max()),
                "effective_shuffle_fraction_mean": float(group["effective_shuffle_fraction"].mean()),
                "overall_max_abs_orientation_sum": float(group["overall_max_abs_orientation_sum"].max()),
                "seed_block_max_abs_orientation_sum": float(group["seed_block_max_abs_orientation_sum"].max()),
                "seed_block_pose_max_abs_orientation_sum": float(group["seed_block_pose_max_abs_orientation_sum"].max()),
                "seed_block_count_bin_max_abs_orientation_sum": float(group["seed_block_count_bin_max_abs_orientation_sum"].max()),
                "seed_block_thickness_max_abs_orientation_sum": float(group["seed_block_thickness_max_abs_orientation_sum"].max()),
                "seed_block_pose_count_bin_max_abs_orientation_sum": float(
                    group["seed_block_pose_count_bin_max_abs_orientation_sum"].max()
                ),
                "seed_block_thickness_pose_count_bin_max_abs_orientation_sum": float(
                    group["seed_block_thickness_pose_count_bin_max_abs_orientation_sum"].max()
                ),
            }
        )
    return pd.DataFrame(result)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], aggregate: pd.DataFrame) -> None:
    lines = [
        "# v8A paired-clean null behavior diagnosis",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Primary mode: `{gate['primary_shuffle_mode']}`",
        f"- Primary fixed p95/max: `{gate['primary_fixed_threshold_hm_min_recall_p95']:.4f}` / `{gate['primary_fixed_threshold_hm_min_recall_max']:.4f}`",
        f"- Primary selected p95/max: `{gate['primary_selected_threshold_hm_min_recall_p95']:.4f}` / `{gate['primary_selected_threshold_hm_min_recall_max']:.4f}`",
        f"- All-mode fixed p95/max: `{gate['all_modes_fixed_threshold_hm_min_recall_p95']:.4f}` / `{gate['all_modes_fixed_threshold_hm_min_recall_max']:.4f}`",
        f"- All-mode selected p95/max: `{gate['all_modes_selected_threshold_hm_min_recall_p95']:.4f}` / `{gate['all_modes_selected_threshold_hm_min_recall_max']:.4f}`",
        "",
        "## Null Summary",
        "",
        markdown_table(
            aggregate.sort_values(["shuffle_mode", "hm_min_recall_max"], ascending=[True, False]),
            [
                "shuffle_mode",
                "model",
                "eval_split",
                "threshold_policy",
                "seed_count",
                "effective_shuffle_fraction_mean",
                "hm_min_recall_mean",
                "hm_min_recall_p95",
                "hm_min_recall_max",
            ],
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_paired_clean_null_behavior_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose paired-clean shuffled-label/null behavior for v8A H/M features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in range(11001, 11061)))
    parser.add_argument(
        "--null-modes",
        default=",".join(NULL_MODES),
        help="Comma-separated paired-null orientation modes. First mode is the primary gate mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing paired-clean null diagnosis because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing paired-clean null diagnosis because {name} reports existing XRT cube reads.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    null_modes = tuple(item.strip() for item in args.null_modes.split(",") if item.strip())
    unknown_modes = sorted(set(null_modes) - set(SUPPORTED_NULL_MODES))
    if unknown_modes:
        raise RuntimeError(f"Unsupported paired-clean null mode(s): {unknown_modes}")
    if not null_modes:
        raise RuntimeError("At least one paired-clean null mode is required.")
    primary_mode = null_modes[0]
    sk = require_sklearn()
    rows = evaluate_paired_null(frame, main_cols, seeds, sk, null_modes)
    aggregate = summarize_null(rows)

    primary = aggregate[aggregate["shuffle_mode"].eq(primary_mode)].copy()
    primary_fixed = primary[primary["threshold_policy"].eq("fixed_0p5")]
    primary_selected = primary[primary["threshold_policy"].eq("validation_selected")]
    primary_fixed_p95 = float(primary_fixed["hm_min_recall_p95"].max()) if not primary_fixed.empty else 1.0
    primary_selected_p95 = float(primary_selected["hm_min_recall_p95"].max()) if not primary_selected.empty else 1.0
    primary_fixed_max = float(primary_fixed["hm_min_recall_max"].max()) if not primary_fixed.empty else 1.0
    primary_selected_max = float(primary_selected["hm_min_recall_max"].max()) if not primary_selected.empty else 1.0
    primary_effective_min = float(primary["effective_shuffle_fraction_min"].min()) if not primary.empty else 0.0
    primary_effective_max = float(primary["effective_shuffle_fraction_max"].max()) if not primary.empty else 1.0
    primary_seed_count = int(primary["seed_count"].min()) if not primary.empty else 0
    primary_seed_block_orientation_max = float(primary["seed_block_max_abs_orientation_sum"].max()) if not primary.empty else 1.0
    primary_pose_count_orientation_max = (
        float(primary["seed_block_pose_count_bin_max_abs_orientation_sum"].max()) if not primary.empty else 1.0
    )
    primary_full_cell_orientation_max = (
        float(primary["seed_block_thickness_pose_count_bin_max_abs_orientation_sum"].max()) if not primary.empty else 1.0
    )
    available_modes = sorted(str(item) for item in aggregate["shuffle_mode"].dropna().unique())
    all_fixed = aggregate[aggregate["threshold_policy"].eq("fixed_0p5")]
    all_selected = aggregate[aggregate["threshold_policy"].eq("validation_selected")]
    all_fixed_p95 = float(all_fixed["hm_min_recall_p95"].max()) if not all_fixed.empty else 1.0
    all_selected_p95 = float(all_selected["hm_min_recall_p95"].max()) if not all_selected.empty else 1.0
    all_fixed_max = float(all_fixed["hm_min_recall_max"].max()) if not all_fixed.empty else 1.0
    all_selected_max = float(all_selected["hm_min_recall_max"].max()) if not all_selected.empty else 1.0

    pass_items = {
        "primary_mode_available": not primary.empty,
        "all_null_modes_available": set(null_modes).issubset(set(available_modes)),
        "shuffle_seed_count": primary_seed_count >= THRESHOLDS["shuffle_seed_count_min"],
        "effective_shuffle_fraction_min_ok": primary_effective_min >= THRESHOLDS["effective_shuffle_fraction_min"],
        "effective_shuffle_fraction_max_ok": primary_effective_max <= THRESHOLDS["effective_shuffle_fraction_max"],
        "seed_block_orientation_balanced": primary_seed_block_orientation_max == 0.0,
        "strict_pose_count_orientation_balanced": (
            primary_pose_count_orientation_max == 0.0 if primary_mode == STRICT_POSE_COUNT_MODE else True
        ),
        "strict_full_cell_orientation_balanced": (
            primary_full_cell_orientation_max == 0.0 if primary_mode == STRICT_FULL_CELL_MODE else True
        ),
        "fixed_threshold_null_p95_under_ceiling": primary_fixed_p95 <= THRESHOLDS["null_hm_min_recall_p95_ceiling"],
        "selected_threshold_null_p95_under_ceiling": primary_selected_p95 <= THRESHOLDS["null_hm_min_recall_p95_ceiling"],
        "fixed_threshold_null_max_under_ceiling": primary_fixed_max <= THRESHOLDS["null_hm_min_recall_single_seed_max"],
        "selected_threshold_null_max_under_ceiling": primary_selected_max <= THRESHOLDS["null_hm_min_recall_single_seed_max"],
        "all_modes_fixed_threshold_null_p95_under_ceiling": all_fixed_p95 <= THRESHOLDS["null_hm_min_recall_p95_ceiling"],
        "all_modes_selected_threshold_null_p95_under_ceiling": all_selected_p95 <= THRESHOLDS["null_hm_min_recall_p95_ceiling"],
        "all_modes_fixed_threshold_null_max_under_ceiling": all_fixed_max <= THRESHOLDS["null_hm_min_recall_single_seed_max"],
        "all_modes_selected_threshold_null_max_under_ceiling": all_selected_max <= THRESHOLDS["null_hm_min_recall_single_seed_max"],
    }
    failure_labels = {
        "primary_mode_available": "primary_paired_clean_mode_missing",
        "all_null_modes_available": "paired_clean_null_mode_missing",
        "shuffle_seed_count": "shuffle_seed_count_below_minimum",
        "effective_shuffle_fraction_min_ok": "effective_shuffle_fraction_below_minimum",
        "effective_shuffle_fraction_max_ok": "effective_shuffle_fraction_above_maximum",
        "seed_block_orientation_balanced": "seed_block_orientation_not_balanced",
        "strict_pose_count_orientation_balanced": "seed_block_pose_count_orientation_not_balanced",
        "strict_full_cell_orientation_balanced": "seed_block_thickness_pose_count_orientation_not_balanced",
        "fixed_threshold_null_p95_under_ceiling": "fixed_threshold_null_p95_exceeded_ceiling",
        "selected_threshold_null_p95_under_ceiling": "selected_threshold_null_p95_exceeded_ceiling",
        "fixed_threshold_null_max_under_ceiling": "fixed_threshold_null_single_seed_max_exceeded_ceiling",
        "selected_threshold_null_max_under_ceiling": "selected_threshold_null_single_seed_max_exceeded_ceiling",
        "all_modes_fixed_threshold_null_p95_under_ceiling": "all_modes_fixed_threshold_null_p95_exceeded_ceiling",
        "all_modes_selected_threshold_null_p95_under_ceiling": "all_modes_selected_threshold_null_p95_exceeded_ceiling",
        "all_modes_fixed_threshold_null_max_under_ceiling": "all_modes_fixed_threshold_null_single_seed_max_exceeded_ceiling",
        "all_modes_selected_threshold_null_max_under_ceiling": "all_modes_selected_threshold_null_single_seed_max_exceeded_ceiling",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/diagnose_v8a_paired_clean_null_behavior.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_paired_clean_null_behavior_diagnosis",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "gate_passed": gate_passed,
        "decision": "paired_clean_null_behavior_clean" if gate_passed else "paired_clean_null_behavior_root_cause_needed",
        "primary_shuffle_mode": primary_mode,
        "required_shuffle_modes": list(null_modes),
        "supported_shuffle_modes": list(SUPPORTED_NULL_MODES),
        "available_shuffle_modes": available_modes,
        "shuffle_seed_count": int(len(seeds)),
        "main_feature_count": int(len(main_cols)),
        "primary_seed_count_min": primary_seed_count,
        "primary_effective_shuffle_fraction_min": primary_effective_min,
        "primary_effective_shuffle_fraction_max": primary_effective_max,
        "primary_seed_block_orientation_max_abs_sum": primary_seed_block_orientation_max,
        "primary_seed_block_pose_count_bin_orientation_max_abs_sum": primary_pose_count_orientation_max,
        "primary_seed_block_thickness_pose_count_bin_orientation_max_abs_sum": primary_full_cell_orientation_max,
        "primary_fixed_threshold_hm_min_recall_p95": primary_fixed_p95,
        "primary_selected_threshold_hm_min_recall_p95": primary_selected_p95,
        "primary_fixed_threshold_hm_min_recall_max": primary_fixed_max,
        "primary_selected_threshold_hm_min_recall_max": primary_selected_max,
        "all_modes_fixed_threshold_hm_min_recall_p95": all_fixed_p95,
        "all_modes_selected_threshold_hm_min_recall_p95": all_selected_p95,
        "all_modes_fixed_threshold_hm_min_recall_max": all_fixed_max,
        "all_modes_selected_threshold_hm_min_recall_max": all_selected_max,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    rows.to_csv(output_dir / "v8a_paired_clean_null_behavior_rows.csv", index=False, lineterminator="\n")
    aggregate.to_csv(output_dir / "v8a_paired_clean_null_behavior_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_paired_clean_null_behavior_gate.json", json_clean(gate))
    write_report(output_dir, gate, aggregate)
    print(
        "decision={decision} gate_passed={passed} primary_fixed_p95={fixed_p95:.4f} primary_fixed_max={fixed_max:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            fixed_p95=primary_fixed_p95,
            fixed_max=primary_fixed_max,
        )
    )


if __name__ == "__main__":
    main()
