from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


METHODS = ["PhysicsOnly", "LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
SOURCE_ORDER = ["mono_60kev", "mono_100kev", "spectrum_120kv"]
BASE_MODEL_TABLE = Path("results/material_sorting_v2/material_fingerprint_model_table.csv")
BASE_PHOTONS_PER_SAMPLE = 100
ACCEPTANCE_TARGETS = {
    "top1_accuracy": 0.85,
    "macro_f1": 0.80,
    "min_class_recall": 0.70,
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def relative_label(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def prepare_frame(
    project_root: Path,
    raw_dir: Path,
    photon_budget: int,
    source_subset: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photon_budget
    try:
        material_records, calibration_records = v2.discover_records(project_root, raw_dir)
        if source_subset is not None:
            source_set = set(source_subset)
            material_records = [record for record in material_records if record.source_id in source_set]
            calibration_records = [record for record in calibration_records if record.source_id in source_set]
        if not material_records or not calibration_records:
            raise ValueError(f"No material/calibration records for source subset: {source_subset}")
        calibration = v2.calibration_table(calibration_records)
        material_samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(material_samples, calibration)
        fused, table_mode = v2.fuse_sources(calibrated)
        status = {
            "material_metadata_found": len(material_records),
            "calibration_metadata_found": len(calibration_records),
            "materials_found": sorted({record.material for record in material_records}),
            "sources_found": sorted({record.source_id for record in material_records}),
            "thicknesses_found": sorted({record.thickness_mm for record in material_records}),
            "seeds_found": sorted({record.random_seed for record in material_records}),
            "table_mode": table_mode,
            "rows": int(len(fused)),
        }
        return fused, status
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def load_model_table(project_root: Path, model_table_path: Path | None = None) -> pd.DataFrame | None:
    path = project_root / (model_table_path or BASE_MODEL_TABLE)
    if not path.exists():
        return None
    return pd.read_csv(path)


def aggregate_model_table(base: pd.DataFrame, photon_budget: int) -> pd.DataFrame:
    if photon_budget == BASE_PHOTONS_PER_SAMPLE:
        return base.copy()
    if photon_budget % BASE_PHOTONS_PER_SAMPLE != 0:
        raise ValueError(f"Photon budget must be a multiple of {BASE_PHOTONS_PER_SAMPLE}: {photon_budget}")
    factor = photon_budget // BASE_PHOTONS_PER_SAMPLE
    frame = base.copy()
    frame["sample_id"] = (frame["sample_id"].astype(int) // factor).astype(int)
    key_cols = ["material", "thickness_mm", "random_seed", "sample_id"]
    aggregations = {}
    for col in frame.columns:
        if col in key_cols:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if (
            col.endswith("_sum")
            or "__I_" in col
            or col.endswith("__hit_count")
            or col.endswith("__direct_primary_count")
            or col.endswith("__scattered_primary_count")
        ):
            aggregations[col] = "sum"
        else:
            aggregations[col] = "mean"
    aggregated = frame.groupby(key_cols, as_index=False).agg(aggregations)
    return aggregated.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def filter_source_columns(frame: pd.DataFrame, source_subset: tuple[str, ...]) -> pd.DataFrame:
    prefixes = tuple(f"{source}__" for source in source_subset)
    keep = []
    for col in frame.columns:
        if "__" not in col:
            if col == "dual_source_log_transmission_ratio_60_100" and not {"mono_60kev", "mono_100kev"}.issubset(source_subset):
                continue
            keep.append(col)
            continue
        if col.startswith(prefixes):
            keep.append(col)
    return frame[keep].copy()


def prepare_cached_frame(
    base_model_table: pd.DataFrame | None,
    project_root: Path,
    raw_dir: Path,
    photon_budget: int,
    source_subset: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    if base_model_table is None:
        return prepare_frame(project_root, raw_dir, photon_budget, source_subset)
    frame = aggregate_model_table(base_model_table, photon_budget)
    if source_subset is not None:
        frame = filter_source_columns(frame, source_subset)
    status = {
        "material_metadata_found": int(frame[["material", "thickness_mm", "random_seed"]].drop_duplicates().shape[0]),
        "calibration_metadata_found": 0,
        "materials_found": sorted(frame["material"].astype(str).unique()),
        "sources_found": list(source_subset or SOURCE_ORDER),
        "thicknesses_found": sorted(float(value) for value in frame["thickness_mm"].unique()),
        "seeds_found": sorted(int(value) for value in frame["random_seed"].unique()),
        "table_mode": "cached_multi_source_fused" if source_subset is None or len(source_subset) > 1 else "cached_single_source",
        "rows": int(len(frame)),
        "source": "material_fingerprint_model_table",
    }
    return frame, status


def split_by_seed(frame: pd.DataFrame, train_seeds: set[int], eval_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["random_seed"].astype(int).isin(train_seeds)].copy()
    eval_frame = frame[frame["random_seed"].astype(int).eq(eval_seed)].copy()
    return train, eval_frame


def append_dictionary(train: pd.DataFrame, eval_frame: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dictionary = v2.fit_dictionary(train, base_cols)
    train_aug = v2.append_dictionary_features(train, dictionary)
    eval_aug = v2.append_dictionary_features(eval_frame, dictionary)
    return train_aug, eval_aug, v2.numeric_feature_columns(train_aug)


def score_method(method: str, train: pd.DataFrame, eval_frame: pd.DataFrame, feature_cols: list[str], sk) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    _, predictions, scores, classes = v2.train_and_score(method, train, eval_frame, feature_cols, sk)
    metrics = v2.evaluate_scores(method, eval_frame, predictions, scores, classes, sk)
    return metrics, predictions, scores, classes


def select_on_validation(train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str], sk) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        try:
            metrics, _, _, _ = score_method(method, train, validation, feature_cols, sk)
        except Exception as exc:  # noqa: BLE001 - diagnostics should record failed model candidates.
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
    return table.sort_values(["top1_accuracy", "macro_f1", "min_class_recall"], ascending=[False, False, False])


def evaluate_budget(project_root: Path, raw_dir: Path, photon_budget: int, sk, base_model_table: pd.DataFrame | None) -> dict:
    frame, status = prepare_cached_frame(base_model_table, project_root, raw_dir, photon_budget)
    base_cols = v2.numeric_feature_columns(frame)
    train, validation, test = v2.split_frames(frame)
    train_aug, validation_aug, feature_cols = append_dictionary(train, validation, base_cols)
    validation_table = select_on_validation(train_aug, validation_aug, feature_cols, sk)
    selected_method = str(validation_table.iloc[0]["method"])

    final_train = pd.concat([train, validation], ignore_index=True)
    final_train_aug, test_aug, final_feature_cols = append_dictionary(final_train, test, base_cols)
    final_metrics, predictions, scores, classes = score_method(selected_method, final_train_aug, test_aug, final_feature_cols, sk)
    validation_metrics = validation_table.iloc[0].to_dict()
    decisions = v2.decision_frame(test_aug, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    return {
        "budget": photon_budget,
        "frame": frame,
        "base_feature_cols": base_cols,
        "final_train_aug": final_train_aug,
        "test_aug": test_aug,
        "final_feature_cols": final_feature_cols,
        "selected_method": selected_method,
        "validation_table": validation_table,
        "validation_metrics": validation_metrics,
        "final_metrics": final_metrics,
        "predictions": predictions,
        "scores": scores,
        "classes": classes,
        "decisions": decisions,
        "status": status,
    }


def photon_budget_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        final_metrics = result["final_metrics"]
        validation_metrics = result["validation_metrics"]
        status = result["status"]
        rows.append(
            {
                "photons_per_sample": result["budget"],
                "selected_method": result["selected_method"],
                "table_mode": status["table_mode"],
                "train_rows": int((result["frame"]["random_seed"] == v2.TRAIN_SEED).sum()),
                "validation_rows": int((result["frame"]["random_seed"] == v2.VALIDATION_SEED).sum()),
                "test_rows": int((result["frame"]["random_seed"] == v2.TEST_SEED).sum()),
                "validation_top1_accuracy": validation_metrics.get("top1_accuracy"),
                "validation_top3_accuracy": validation_metrics.get("top3_accuracy"),
                "validation_macro_f1": validation_metrics.get("macro_f1"),
                "validation_min_class_recall": validation_metrics.get("min_class_recall"),
                "final_top1_accuracy": final_metrics.get("top1_accuracy"),
                "final_top3_accuracy": final_metrics.get("top3_accuracy"),
                "final_macro_f1": final_metrics.get("macro_f1"),
                "final_min_class_recall": final_metrics.get("min_class_recall"),
                "passes_all_acceptance_targets": bool(
                    final_metrics.get("top1_accuracy", 0.0) >= ACCEPTANCE_TARGETS["top1_accuracy"]
                    and final_metrics.get("macro_f1", 0.0) >= ACCEPTANCE_TARGETS["macro_f1"]
                    and final_metrics.get("min_class_recall", 0.0) >= ACCEPTANCE_TARGETS["min_class_recall"]
                ),
            }
        )
    return pd.DataFrame(rows)


def model_comparison_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        table = result["validation_table"].copy()
        table.insert(0, "photons_per_sample", result["budget"])
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def standardized_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame[feature_cols].to_numpy(dtype=float)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (x - center) / scale, center, scale


def separability_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        frame = result["frame"]
        dev = frame[frame["random_seed"].astype(int).isin({v2.TRAIN_SEED, v2.VALIDATION_SEED})].copy()
        physics_cols = v2.physics_feature_columns(result["base_feature_cols"])
        x, _, _ = standardized_matrix(dev, physics_cols)
        labels = dev["material"].astype(str).to_numpy()
        centroids = {material: x[labels == material].mean(axis=0) for material in sorted(set(labels))}
        for material in sorted(centroids):
            own = x[labels == material]
            within_radius = float(np.linalg.norm(own - centroids[material], axis=1).mean()) if len(own) else math.nan
            neighbor_rows = []
            for other in sorted(centroids):
                if other == material:
                    continue
                neighbor_rows.append((other, float(np.linalg.norm(centroids[material] - centroids[other]))))
            nearest_material, nearest_distance = min(neighbor_rows, key=lambda item: item[1])
            rows.append(
                {
                    "photons_per_sample": result["budget"],
                    "material": material,
                    "dev_samples": int(len(own)),
                    "physics_feature_count": len(physics_cols),
                    "within_material_radius": within_radius,
                    "nearest_material": nearest_material,
                    "nearest_centroid_distance": nearest_distance,
                    "separability_ratio": nearest_distance / within_radius if within_radius > 1e-12 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def centroid_distances(frame: pd.DataFrame, feature_cols: list[str]) -> dict[tuple[str, str], float]:
    x, _, _ = standardized_matrix(frame, feature_cols)
    labels = frame["material"].astype(str).to_numpy()
    centroids = {material: x[labels == material].mean(axis=0) for material in sorted(set(labels))}
    distances = {}
    for left, right in itertools.permutations(sorted(centroids), 2):
        distances[(left, right)] = float(np.linalg.norm(centroids[left] - centroids[right]))
    return distances


def confusion_distance_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        decisions = result["decisions"]
        misses = decisions[~decisions["is_correct"]].copy()
        distances = centroid_distances(result["final_train_aug"], result["final_feature_cols"])
        for (true_material, predicted_material), part in misses.groupby(["material", "predicted_material"]):
            rows.append(
                {
                    "photons_per_sample": result["budget"],
                    "true_material": true_material,
                    "predicted_material": predicted_material,
                    "count": int(len(part)),
                    "mean_score_margin": float(part["score_margin"].mean()),
                    "dev_centroid_distance": distances.get((true_material, predicted_material), math.nan),
                    "top3_contains_true_rate": float(
                        np.mean([true_material in str(value).split(";") for value in part["top3_candidates"]])
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["photons_per_sample", "true_material", "predicted_material", "count"])
    return pd.DataFrame(rows).sort_values(["photons_per_sample", "count"], ascending=[True, False])


def seed_variance_rows(project_root: Path, raw_dir: Path, budgets: list[int], sk, base_model_table: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        frame, _ = prepare_cached_frame(base_model_table, project_root, raw_dir, budget)
        base_cols = v2.numeric_feature_columns(frame)
        seeds = sorted(int(seed) for seed in frame["random_seed"].unique())
        for holdout_seed in seeds:
            train, eval_frame = split_by_seed(frame, set(seeds) - {holdout_seed}, holdout_seed)
            train_aug, eval_aug, feature_cols = append_dictionary(train, eval_frame, base_cols)
            for method in ["PhysicsOnly", "ExtraTrees"]:
                try:
                    metrics, _, _, _ = score_method(method, train_aug, eval_aug, feature_cols, sk)
                    rows.append(
                        {
                            "photons_per_sample": budget,
                            "method": method,
                            "holdout_seed": holdout_seed,
                            "train_seeds": ";".join(str(seed) for seed in seeds if seed != holdout_seed),
                            "top1_accuracy": metrics["top1_accuracy"],
                            "top3_accuracy": metrics["top3_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "min_class_recall": metrics["min_class_recall"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "photons_per_sample": budget,
                            "method": method,
                            "holdout_seed": holdout_seed,
                            "train_seeds": ";".join(str(seed) for seed in seeds if seed != holdout_seed),
                            "error": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


def source_screening_rows(project_root: Path, raw_dir: Path, photon_budget: int, sk, base_model_table: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for size in [1, 2, 3]:
        for subset in itertools.combinations(SOURCE_ORDER, size):
            frame, status = prepare_cached_frame(base_model_table, project_root, raw_dir, photon_budget, subset)
            base_cols = v2.numeric_feature_columns(frame)
            train, validation, _ = v2.split_frames(frame)
            train_aug, validation_aug, feature_cols = append_dictionary(train, validation, base_cols)
            for method in ["PhysicsOnly", "ExtraTrees"]:
                try:
                    metrics, _, _, _ = score_method(method, train_aug, validation_aug, feature_cols, sk)
                    rows.append(
                        {
                            "photons_per_sample": photon_budget,
                            "source_subset": ";".join(subset),
                            "source_count": len(subset),
                            "table_mode": status["table_mode"],
                            "method": method,
                            "feature_count": len(feature_cols),
                            "validation_top1_accuracy": metrics["top1_accuracy"],
                            "validation_top3_accuracy": metrics["top3_accuracy"],
                            "validation_macro_f1": metrics["macro_f1"],
                            "validation_min_class_recall": metrics["min_class_recall"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "photons_per_sample": photon_budget,
                            "source_subset": ";".join(subset),
                            "source_count": len(subset),
                            "method": method,
                            "error": str(exc),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["method", "validation_top1_accuracy", "validation_macro_f1"],
        ascending=[True, False, False],
    )


def write_report(output_dir: Path, manifest: dict, photon_table: pd.DataFrame, separability: pd.DataFrame, source_screening: pd.DataFrame) -> None:
    best = photon_table.sort_values(["final_top1_accuracy", "final_macro_f1"], ascending=[False, False]).iloc[0]
    hardest = separability.sort_values("separability_ratio", ascending=True).head(6)
    best_sources = source_screening[source_screening["method"].eq("ExtraTrees")].head(5)
    lines = [
        "# 本科十材料分选重做诊断",
        "",
        "本报告由 `analysis/material_sorting_rebuild_diagnostics.py` 生成，用于解释 v2 full matrix 失败后应先改输入协议还是先升级模型。",
        "",
        "## 结论",
        "",
        f"- 当前最好 photon budget 为 `{int(best['photons_per_sample'])}` photons/sample，final Top-1 为 `{best['final_top1_accuracy']:.4f}`，macro-F1 为 `{best['final_macro_f1']:.4f}`，min recall 为 `{best['final_min_class_recall']:.4f}`。",
        "- 这些结果仍必须按十材料负结果处理，除非同时达到 Top-1 >= 0.85、macro-F1 >= 0.80、min recall >= 0.70。",
        "- 如果 photon budget 提升不能显著改善结果，下一步应优先做 mono energy scan 和多能输入筛选，而不是直接把当前标量表交给 CNN/Transformer。",
        "",
        "## 最难分材料",
        "",
    ]
    for row in hardest.itertuples(index=False):
        lines.append(
            f"- `{row.material}` 最近邻为 `{row.nearest_material}`，separability ratio `{row.separability_ratio:.4f}`。"
        )
    lines.extend(["", "## 当前源组合筛选", ""])
    for row in best_sources.itertuples(index=False):
        lines.append(
            f"- `{row.source_subset}`: validation Top-1 `{row.validation_top1_accuracy:.4f}`, macro-F1 `{row.validation_macro_f1:.4f}`。"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `photon_budget_curve.csv`：不同 photon 聚合预算下的 validation 选择和 final test 结果。",
            "- `photon_budget_model_comparison.csv`：各模型在 validation seed 上的对比。",
            "- `per_material_separability.csv`：每种材料的最近邻、类内半径和 separability ratio。",
            "- `seed_variance.csv`：leave-one-seed-out 稳定性诊断。",
            "- `source_pair_screening.csv`：现有 60 keV、100 keV、120 kV spectrum 的单源/双源/三源筛选。",
            "- `confusion_pair_distance.csv`：final test 误分对、score margin 和开发集 centroid 距离。",
            "",
            "本报告不包含 V3 或人工复核高级线；那些内容应留到导师/高级仓库。",
            "",
            f"Manifest: `{manifest['generated_by']}` at `{manifest['generated_at_utc']}`.",
        ]
    )
    (output_dir / "UNDERGRAD_TEN_MATERIAL_REBUILD_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-v2 diagnostics for rebuilding undergraduate ten-material sorting.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/full")
    parser.add_argument("--output-dir", default="results/material_sorting_rebuild")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--photon-budgets", default="100,200,500,1000")
    parser.add_argument("--screen-budget", type=int, default=500)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_int_list(args.photon_budgets)

    sk = v2.require_sklearn()
    base_model_table = load_model_table(project_root)
    budget_results = [evaluate_budget(project_root, raw_dir, budget, sk, base_model_table) for budget in budgets]
    photon_table = photon_budget_rows(budget_results)
    model_comparison = model_comparison_rows(budget_results)
    separability = separability_rows(budget_results)
    confusion_distances = confusion_distance_rows(budget_results)
    seed_variance = seed_variance_rows(project_root, raw_dir, budgets, sk, base_model_table)
    source_screening = source_screening_rows(project_root, raw_dir, args.screen_budget, sk, base_model_table)

    write_csv(photon_table, output_dir / "photon_budget_curve.csv")
    write_csv(model_comparison, output_dir / "photon_budget_model_comparison.csv")
    write_csv(separability, output_dir / "per_material_separability.csv")
    write_csv(confusion_distances, output_dir / "confusion_pair_distance.csv")
    write_csv(seed_variance, output_dir / "seed_variance.csv")
    write_csv(source_screening, output_dir / "source_pair_screening.csv")

    best = photon_table.sort_values(["final_top1_accuracy", "final_macro_f1"], ascending=[False, False]).iloc[0].to_dict()
    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/material_sorting_rebuild_diagnostics.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": relative_label(raw_dir, project_root),
        "output_dir": relative_label(output_dir, project_root),
        "photon_budgets": budgets,
        "screen_budget": args.screen_budget,
        "used_cached_model_table": base_model_table is not None,
        "acceptance_targets": ACCEPTANCE_TARGETS,
        "best_photon_budget_result": best,
        "stage_conclusion": "rebuild_diagnostic_only_not_success_claim",
        "claim_boundary": [
            "This script reuses completed v2 raw runs and does not turn v2 into a ten-material success claim.",
            "The final test seed remains evaluation-only for each photon-budget curve point.",
            "CNN/Transformer routes require sequence or image-like inputs; they are not applied to the current scalar table here.",
            "V3 human-review product planning is intentionally out of scope for the public undergraduate teammate repo.",
        ],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    v2.write_manifest(output_dir / "material_sorting_rebuild_manifest.json", manifest)
    write_report(output_dir, manifest, photon_table, separability, source_screening)
    print(f"Wrote rebuild diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
