from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


TRAIN_SEEDS = [101, 202, 303]
VALIDATION_SEED = 404
TEST_SEED = 505
ACCEPTANCE_TARGETS = {
    "top1_accuracy": 0.85,
    "macro_f1": 0.80,
    "min_class_recall": 0.70,
}
METHODS = [
    "PhysicsOnly",
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "HierarchicalExtraTrees",
    "XGBoostGPU",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def material_group_map(project_root: Path) -> dict[str, str]:
    catalog = pd.read_csv(project_root / v2.MATERIALS_FILE)
    return {
        str(row.material_name): str(row.group_label)
        for row in catalog.itertuples(index=False)
        if str(row.material_name) in v2.TARGET_MATERIALS
    }


def discover_status(material_records: list[v2.RunRecord], calibration_records: list[v2.RunRecord]) -> dict:
    material_keys = {
        (record.material, round(record.thickness_mm, 3), record.source_id, record.random_seed)
        for record in material_records
    }
    calibration_keys = {(record.source_id, record.random_seed) for record in calibration_records}
    materials = sorted({record.material for record in material_records})
    thicknesses = sorted({record.thickness_mm for record in material_records})
    sources = sorted({record.source_id for record in material_records})
    seeds = sorted({record.random_seed for record in material_records})
    expected_material_keys = {
        (material, thickness, source, seed)
        for material in v2.TARGET_MATERIALS
        for thickness in thicknesses
        for source in sources
        for seed in [*TRAIN_SEEDS, VALIDATION_SEED, TEST_SEED]
    }
    expected_calibration_keys = {
        (source, seed)
        for source in sources
        for seed in [*TRAIN_SEEDS, VALIDATION_SEED, TEST_SEED]
    }
    return {
        "material_metadata_found": len(material_records),
        "calibration_metadata_found": len(calibration_records),
        "materials_found": materials,
        "sources_found": sources,
        "thicknesses_found": thicknesses,
        "seeds_found": seeds,
        "missing_material_runs": len(expected_material_keys - material_keys),
        "missing_calibration_runs": len(expected_calibration_keys - calibration_keys),
        "complete_selected_rebuild_matrix": expected_material_keys.issubset(material_keys)
        and expected_calibration_keys.issubset(calibration_keys),
    }


def build_frame(project_root: Path, raw_dir: Path, photons_per_sample: int) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photons_per_sample
    try:
        material_records, calibration_records = v2.discover_records(project_root, raw_dir)
        status = discover_status(material_records, calibration_records)
        calibration = v2.calibration_table(calibration_records)
        samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(samples, calibration)
        fused, table_mode = v2.fuse_sources(calibrated)
        status["table_mode"] = table_mode
        status["rows"] = int(len(fused))
        return fused, status
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def split_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["random_seed"].astype(int).isin(TRAIN_SEEDS)].copy()
    validation = frame[frame["random_seed"].astype(int).eq(VALIDATION_SEED)].copy()
    test = frame[frame["random_seed"].astype(int).eq(TEST_SEED)].copy()
    return train, validation, test


def append_dictionary(train: pd.DataFrame, eval_frame: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    dictionary = v2.fit_dictionary(train, base_cols)
    train_aug = v2.append_dictionary_features(train, dictionary)
    eval_aug = v2.append_dictionary_features(eval_frame, dictionary)
    return train_aug, eval_aug, v2.numeric_feature_columns(train_aug), dictionary


def score_xgboost_gpu(train: pd.DataFrame, eval_frame: pd.DataFrame, feature_cols: list[str], sk) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing XGBoost dependency: {exc}") from exc
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["material"].astype(str))
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.95,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(train[feature_cols], y_train, verbose=False)
    scores = model.predict_proba(eval_frame[feature_cols])
    classes = np.array(encoder.classes_)
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions, scores, classes, "cuda_requested"


def score_hierarchical_extra_trees(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.array(v2.TARGET_MATERIALS)
    train_groups = train["material"].astype(str).map(group_map)
    group_model = sk["ExtraTreesClassifier"](n_estimators=1000, random_state=91, n_jobs=-1, class_weight="balanced")
    group_model.fit(train[feature_cols], train_groups)
    group_scores = group_model.predict_proba(eval_frame[feature_cols])
    group_classes = np.array(group_model.classes_)
    scores = np.zeros((len(eval_frame), len(classes)), dtype=float)
    for group in group_classes:
        materials = [material for material in classes if group_map.get(material) == group]
        part = train[train["material"].isin(materials)].copy()
        if part["material"].nunique() == 1:
            material_scores = np.ones((len(eval_frame), 1), dtype=float)
            material_classes = np.array(materials)
        else:
            material_model = sk["ExtraTreesClassifier"](n_estimators=1000, random_state=97, n_jobs=-1, class_weight="balanced")
            material_model.fit(part[feature_cols], part["material"])
            material_scores = material_model.predict_proba(eval_frame[feature_cols])
            material_classes = np.array(material_model.classes_)
        group_index = np.where(group_classes == group)[0][0]
        for local_index, material in enumerate(material_classes):
            global_index = np.where(classes == material)[0][0]
            scores[:, global_index] = group_scores[:, group_index] * material_scores[:, local_index]
    scores_sum = scores.sum(axis=1, keepdims=True)
    scores = np.divide(scores, scores_sum, out=np.zeros_like(scores), where=scores_sum > 0)
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions, scores, classes


def score_method(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    extra = {}
    if method == "XGBoostGPU":
        predictions, scores, classes, gpu_status = score_xgboost_gpu(train, eval_frame, feature_cols, sk)
        extra["gpu_status"] = gpu_status
    elif method == "HierarchicalExtraTrees":
        predictions, scores, classes = score_hierarchical_extra_trees(train, eval_frame, feature_cols, group_map, sk)
    else:
        _, predictions, scores, classes = v2.train_and_score(method, train, eval_frame, feature_cols, sk)
    metrics = v2.evaluate_scores(method, eval_frame, predictions, scores, classes, sk)
    metrics.update(extra)
    return metrics, predictions, scores, classes, extra


def validation_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for method in METHODS:
        try:
            metrics, _, _, _, _ = score_method(method, train, validation, feature_cols, group_map, sk)
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "method": method,
                "samples": int(len(validation)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "min_class_recall": math.nan,
                "error": str(exc),
            }
        rows.append(metrics)
    table = pd.DataFrame(rows)
    ranked = table.sort_values(
        ["min_class_recall", "top1_accuracy", "macro_f1"],
        ascending=[False, False, False],
    )
    return table, ranked.iloc[0].to_dict()


def per_class_recall(frame: pd.DataFrame, predictions: np.ndarray, sk) -> pd.DataFrame:
    return v2.per_class_recall_table(frame, predictions, "test", sk)


def talker_reasons(decision: pd.Series, group_map: dict[str, str]) -> str:
    true_material = str(decision["material"])
    predicted = str(decision["predicted_material"])
    candidates = str(decision.get("top3_candidates", "")).split(";")
    reasons = []
    if true_material == predicted:
        reasons.append("top1_match")
    elif true_material in candidates:
        reasons.append("true_material_in_top3")
    else:
        reasons.append("true_material_outside_top3")
    if group_map.get(true_material) == group_map.get(predicted):
        reasons.append("same_absorption_group")
    else:
        reasons.append("cross_group_error")
    if float(decision.get("score_margin", 0.0)) < 0.20:
        reasons.append("small_margin")
    return ";".join(reasons)


def dictionary_talker(
    output_dir: Path,
    dictionary: dict,
    final_train: pd.DataFrame,
    test_aug: pd.DataFrame,
    decisions: pd.DataFrame,
    group_map: dict[str, str],
) -> None:
    entries = []
    for material in dictionary["materials"]:
        part = decisions[decisions["material"].astype(str).eq(material)]
        train_part = final_train[final_train["material"].astype(str).eq(material)]
        common_confusions = (
            part.loc[~part["is_correct"], "predicted_material"].value_counts().head(3).to_dict()
        )
        entry = {
            "material": material,
            "group_label": group_map.get(material, ""),
            "train_samples": int(len(train_part)),
            "test_samples": int(len(part)),
            "test_top1_accuracy": float(part["is_correct"].mean()) if len(part) else math.nan,
            "top3_contains_true_rate": float(
                np.mean([material in str(value).split(";") for value in part["top3_candidates"]])
            ) if len(part) else math.nan,
            "mean_score_margin": float(part["score_margin"].mean()) if len(part) else math.nan,
            "common_confusions": common_confusions,
            "talker_summary": (
                f"{material}: group={group_map.get(material, '')}; "
                f"test_top1={float(part['is_correct'].mean()) if len(part) else math.nan:.3f}; "
                f"main_confusions={common_confusions or 'none'}"
            ),
        }
        entries.append(entry)
    talker = {
        "generated_by": "analysis/material_sorting_selected_rebuild.py",
        "purpose": "endogenous material dictionary talker for undergraduate ten-material rebuild",
        "scope": "uses only repo material catalog, selected rebuild simulation fingerprints, validation/final decisions, and confusion evidence",
        "entries": entries,
    }
    v2.write_manifest(output_dir / "material_dictionary_talker.json", talker)
    write_csv(pd.DataFrame(entries), output_dir / "material_dictionary_talker.csv")
    explanation_rows = decisions.copy()
    explanation_rows["talker_reason"] = explanation_rows.apply(lambda row: talker_reasons(row, group_map), axis=1)
    write_csv(explanation_rows, output_dir / "final_test_decisions_with_talker.csv")


def evaluate_budget(project_root: Path, raw_dir: Path, output_dir: Path, photons_per_sample: int, sk) -> dict:
    group_map = material_group_map(project_root)
    frame, status = build_frame(project_root, raw_dir, photons_per_sample)
    base_cols = v2.numeric_feature_columns(frame)
    train, validation, test = split_frames(frame)
    train_aug, validation_aug, feature_cols, _ = append_dictionary(train, validation, base_cols)
    validation_table, selected = validation_selection(train_aug, validation_aug, feature_cols, group_map, sk)
    final_train = pd.concat([train, validation], ignore_index=True)
    final_train_aug, test_aug, final_feature_cols, final_dictionary = append_dictionary(final_train, test, base_cols)
    selected_method = str(selected["method"])
    final_metrics, predictions, scores, classes, _ = score_method(
        selected_method,
        final_train_aug,
        test_aug,
        final_feature_cols,
        group_map,
        sk,
    )
    decisions = v2.decision_frame(test_aug, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    per_class = per_class_recall(test_aug, predictions, sk)
    prefix = f"p{photons_per_sample}"
    write_csv(validation_table, output_dir / f"{prefix}_validation_model_selection.csv")
    write_csv(pd.DataFrame([final_metrics]), output_dir / f"{prefix}_final_test_summary.csv")
    write_csv(per_class, output_dir / f"{prefix}_per_class_recall_final_test.csv")
    write_csv(decisions, output_dir / f"{prefix}_final_test_decisions.csv")
    if photons_per_sample == 1000:
        dictionary_talker(output_dir, final_dictionary, final_train_aug, test_aug, decisions, group_map)
    result = {
        "photons_per_sample": photons_per_sample,
        "selected_method": selected_method,
        "validation_selected_metrics": selected,
        "final_test_metrics": final_metrics,
        "status": status,
        "passes_all_acceptance_targets": bool(
            final_metrics["top1_accuracy"] >= ACCEPTANCE_TARGETS["top1_accuracy"]
            and final_metrics["macro_f1"] >= ACCEPTANCE_TARGETS["macro_f1"]
            and final_metrics["min_class_recall"] >= ACCEPTANCE_TARGETS["min_class_recall"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected rebuild protocol with multi-seed train/validation/final test.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/selected_rebuild")
    parser.add_argument("--output-dir", default="results/material_sorting_selected_rebuild")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--photon-budgets", default="500,1000,2000")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_int_list(args.photon_budgets)
    sk = v2.require_sklearn()
    results = [evaluate_budget(project_root, raw_dir, output_dir, budget, sk) for budget in budgets]
    summary = pd.DataFrame(
        [
            {
                "photons_per_sample": result["photons_per_sample"],
                "selected_method": result["selected_method"],
                "validation_top1_accuracy": result["validation_selected_metrics"].get("top1_accuracy"),
                "validation_macro_f1": result["validation_selected_metrics"].get("macro_f1"),
                "validation_min_class_recall": result["validation_selected_metrics"].get("min_class_recall"),
                "final_top1_accuracy": result["final_test_metrics"].get("top1_accuracy"),
                "final_top3_accuracy": result["final_test_metrics"].get("top3_accuracy"),
                "final_macro_f1": result["final_test_metrics"].get("macro_f1"),
                "final_min_class_recall": result["final_test_metrics"].get("min_class_recall"),
                "passes_all_acceptance_targets": result["passes_all_acceptance_targets"],
            }
            for result in results
        ]
    )
    write_csv(summary, output_dir / "selected_rebuild_summary.csv")
    best = summary.sort_values(
        ["passes_all_acceptance_targets", "final_min_class_recall", "final_top1_accuracy", "final_macro_f1"],
        ascending=[False, False, False, False],
    ).iloc[0].to_dict()
    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/material_sorting_selected_rebuild.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "name": "selected_rebuild_multi_seed_energy_protocol",
            "train_seeds": TRAIN_SEEDS,
            "validation_seed": VALIDATION_SEED,
            "test_seed": TEST_SEED,
            "photon_budgets": budgets,
            "acceptance_targets": ACCEPTANCE_TARGETS,
            "model_selection_policy": "validation seed selects model by min_class_recall, then top1_accuracy, then macro_f1",
        },
        "results": results,
        "best_result": best,
        "stage_conclusion": "accepted_ten_material_candidate" if bool(best["passes_all_acceptance_targets"]) else "diagnostic_only_not_ready",
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    v2.write_manifest(output_dir / "material_sorting_selected_rebuild_manifest.json", manifest)
    print(f"Wrote selected rebuild evaluation to {output_dir}")


if __name__ == "__main__":
    main()
