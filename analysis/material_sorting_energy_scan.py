from __future__ import annotations

import argparse
import itertools
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


ENERGY_SOURCES = [f"mono_{energy}kev" for energy in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]]
TRAIN_SEED = 101
VALIDATION_SEED = 202
ACCEPTANCE_TARGETS = {
    "top1_accuracy": 0.85,
    "macro_f1": 0.80,
    "min_class_recall": 0.70,
}
CRITICAL_PAIRS = [
    ("Hematite", "Magnetite"),
    ("Quartz", "Albite"),
    ("Chalcopyrite", "Pyrite"),
    ("Dolomite", "Orthoclase"),
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_source_subsets(value: str) -> list[tuple[str, ...]]:
    subsets = []
    for subset in value.split("|"):
        sources = tuple(item.strip() for item in subset.split("+") if item.strip())
        if sources:
            subsets.append(sources)
    return subsets


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def discover_filtered_records(project_root: Path, raw_dir: Path, sources: tuple[str, ...]) -> tuple[list[v2.RunRecord], list[v2.RunRecord]]:
    material_records, calibration_records = v2.discover_records(project_root, raw_dir)
    source_set = set(sources)
    material_records = [record for record in material_records if record.source_id in source_set]
    calibration_records = [record for record in calibration_records if record.source_id in source_set]
    return material_records, calibration_records


def build_frame(project_root: Path, raw_dir: Path, sources: tuple[str, ...], photons_per_sample: int) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photons_per_sample
    try:
        material_records, calibration_records = discover_filtered_records(project_root, raw_dir, sources)
        calibration = v2.calibration_table(calibration_records)
        samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(samples, calibration)
        fused, table_mode = v2.fuse_sources(calibrated)
        status = {
            "material_metadata_found": len(material_records),
            "calibration_metadata_found": len(calibration_records),
            "materials_found": sorted({record.material for record in material_records}),
            "sources_found": sorted(sources),
            "seeds_found": sorted({record.random_seed for record in material_records}),
            "photons_per_sample": photons_per_sample,
            "table_mode": table_mode,
            "rows": int(len(fused)),
        }
        return fused, status
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def filter_source_columns(frame: pd.DataFrame, sources: tuple[str, ...]) -> pd.DataFrame:
    prefixes = tuple(f"{source}__" for source in sources)
    keep = []
    for col in frame.columns:
        if "__" not in col:
            keep.append(col)
        elif col.startswith(prefixes):
            keep.append(col)
    return frame[keep].copy()


def append_dictionary(train: pd.DataFrame, validation: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dictionary = v2.fit_dictionary(train, base_cols)
    train_aug = v2.append_dictionary_features(train, dictionary)
    validation_aug = v2.append_dictionary_features(validation, dictionary)
    return train_aug, validation_aug, v2.numeric_feature_columns(train_aug)


def xgboost_gpu_score(train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str], sk) -> tuple[dict, str]:
    try:
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        return {
            "method": "XGBoostGPU",
            "samples": int(len(validation)),
            "top1_accuracy": math.nan,
            "top3_accuracy": math.nan,
            "macro_f1": math.nan,
            "min_class_recall": math.nan,
            "error": f"missing dependency: {exc}",
        }, "missing"
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["material"].astype(str))
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=700,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.95,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
    )
    try:
        model.fit(train[feature_cols], y_train, verbose=False)
        scores = model.predict_proba(validation[feature_cols])
        classes = encoder.classes_
        predictions = classes[np.argmax(scores, axis=1)]
        metrics = v2.evaluate_scores("XGBoostGPU", validation, predictions, scores, classes, sk)
        return metrics, "cuda_requested"
    except Exception as exc:  # noqa: BLE001
        return {
            "method": "XGBoostGPU",
            "samples": int(len(validation)),
            "top1_accuracy": math.nan,
            "top3_accuracy": math.nan,
            "macro_f1": math.nan,
            "min_class_recall": math.nan,
            "error": str(exc),
        }, "failed"


def score_combo(frame: pd.DataFrame, sk, include_gpu: bool, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["random_seed"].astype(int).eq(TRAIN_SEED)].copy()
    validation = frame[frame["random_seed"].astype(int).eq(VALIDATION_SEED)].copy()
    base_cols = v2.numeric_feature_columns(frame)
    train_aug, validation_aug, feature_cols = append_dictionary(train, validation, base_cols)
    rows = []
    predictions_by_method = {}
    for method in methods:
        try:
            _, predictions, scores, classes = v2.train_and_score(method, train_aug, validation_aug, feature_cols, sk)
            metrics = v2.evaluate_scores(method, validation_aug, predictions, scores, classes, sk)
            predictions_by_method[method] = predictions
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "method": method,
                "samples": int(len(validation_aug)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "min_class_recall": math.nan,
                "error": str(exc),
            }
        rows.append(metrics)
    if include_gpu:
        metrics, gpu_status = xgboost_gpu_score(train_aug, validation_aug, feature_cols, sk)
        metrics["gpu_status"] = gpu_status
        rows.append(metrics)
    metrics_table = pd.DataFrame(rows)
    pair_rows = []
    best_method = metrics_table.sort_values(["top1_accuracy", "macro_f1", "min_class_recall"], ascending=[False, False, False]).iloc[0]["method"]
    if best_method in predictions_by_method:
        pred = predictions_by_method[str(best_method)]
        for left, right in CRITICAL_PAIRS:
            part = validation_aug[validation_aug["material"].isin([left, right])].copy()
            if part.empty:
                continue
            mask = validation_aug["material"].isin([left, right]).to_numpy()
            pair_pred = pred[mask]
            pair_rows.append(
                {
                    "method": best_method,
                    "pair": f"{left}|{right}",
                    "pair_samples": int(len(part)),
                    "pair_accuracy": float(np.mean(part["material"].astype(str).to_numpy() == pair_pred)),
                    "left_recall": float(np.mean(pair_pred[part["material"].astype(str).to_numpy() == left] == left)),
                    "right_recall": float(np.mean(pair_pred[part["material"].astype(str).to_numpy() == right] == right)),
                }
            )
    return metrics_table, pd.DataFrame(pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mono-energy scan inputs for ten-material sorting rebuild.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/energy_scan")
    parser.add_argument("--output-dir", default="results/material_sorting_energy_scan")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--photon-budgets", default="500,1000")
    parser.add_argument("--max-source-count", type=int, default=4)
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--gpu-top-n", type=int, default=12)
    parser.add_argument("--screen-methods", default="ExtraTrees,PhysicsOnly")
    parser.add_argument(
        "--source-subsets",
        default="",
        help="Optional pipe-separated candidate subsets, with sources joined by '+'. Example: mono_40kev+mono_110kev+mono_200kev|mono_40kev+mono_50kev+mono_110kev+mono_200kev",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_int_list(args.photon_budgets)
    screen_methods = [item.strip() for item in str(args.screen_methods).split(",") if item.strip()]
    forced_subsets = parse_source_subsets(args.source_subsets)
    sk = v2.require_sklearn()

    metric_rows = []
    pair_rows = []
    gpu_rows = []
    statuses = []
    for photons in budgets:
        full_frame, full_status = build_frame(project_root, raw_dir, tuple(ENERGY_SOURCES), photons)
        statuses.append(full_status)
        budget_metric_rows = []
        subsets = forced_subsets or [
            sources
            for size in range(1, args.max_source_count + 1)
            for sources in itertools.combinations(ENERGY_SOURCES, size)
        ]
        for sources in subsets:
            frame = filter_source_columns(full_frame, sources)
            metrics_table, pairs = score_combo(frame, sk, include_gpu=False, methods=screen_methods)
            for row in metrics_table.to_dict(orient="records"):
                row.update(
                    {
                        "photons_per_sample": photons,
                        "source_subset": ";".join(sources),
                        "source_count": len(sources),
                        "feature_table_rows": int(len(frame)),
                        "table_mode": "filtered_energy_scan_fused" if len(sources) > 1 else "filtered_energy_scan_single",
                    }
                )
                metric_rows.append(row)
                budget_metric_rows.append(row)
            for row in pairs.to_dict(orient="records"):
                row.update({"photons_per_sample": photons, "source_subset": ";".join(sources), "source_count": len(sources)})
                pair_rows.append(row)
        if args.include_gpu and budget_metric_rows:
            budget_metrics = pd.DataFrame(budget_metric_rows)
            candidates = (
                budget_metrics[budget_metrics["method"].eq("ExtraTrees")]
                .sort_values(["top1_accuracy", "macro_f1", "min_class_recall"], ascending=[False, False, False])
                .head(args.gpu_top_n)
            )
            for candidate in candidates.itertuples(index=False):
                sources = tuple(str(candidate.source_subset).split(";"))
                frame = filter_source_columns(full_frame, sources)
                train = frame[frame["random_seed"].astype(int).eq(TRAIN_SEED)].copy()
                validation = frame[frame["random_seed"].astype(int).eq(VALIDATION_SEED)].copy()
                base_cols = v2.numeric_feature_columns(frame)
                train_aug, validation_aug, feature_cols = append_dictionary(train, validation, base_cols)
                metrics, gpu_status = xgboost_gpu_score(train_aug, validation_aug, feature_cols, sk)
                metrics.update(
                    {
                        "photons_per_sample": photons,
                        "source_subset": ";".join(sources),
                        "source_count": len(sources),
                        "feature_table_rows": int(len(frame)),
                        "table_mode": "filtered_energy_scan_fused" if len(sources) > 1 else "filtered_energy_scan_single",
                        "gpu_status": gpu_status,
                    }
                )
                gpu_rows.append(metrics)

    metrics = pd.DataFrame([*metric_rows, *gpu_rows]).sort_values(
        ["top1_accuracy", "macro_f1", "min_class_recall", "source_count"],
        ascending=[False, False, False, True],
    )
    selection_rank = metrics.sort_values(
        ["min_class_recall", "top1_accuracy", "macro_f1", "source_count"],
        ascending=[False, False, False, True],
    )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["pair_accuracy", "source_count"], ascending=[False, True])
    write_csv(metrics, output_dir / "energy_scan_source_screening.csv")
    write_csv(pairs, output_dir / "energy_scan_pair_screening.csv")
    status_frame = pd.DataFrame(statuses)
    if not status_frame.empty and "sources_found" in status_frame.columns:
        status_frame["sources_found"] = status_frame["sources_found"].apply(lambda value: ";".join(value) if isinstance(value, list) else value)
        status_frame["materials_found"] = status_frame["materials_found"].apply(lambda value: ";".join(value) if isinstance(value, list) else value)
        status_frame["seeds_found"] = status_frame["seeds_found"].apply(lambda value: ";".join(str(item) for item in value) if isinstance(value, list) else value)
    status_frame = status_frame.drop_duplicates(subset=["photons_per_sample", "sources_found"])
    write_csv(status_frame, output_dir / "energy_scan_inventory.csv")

    best = metrics.iloc[0].to_dict()
    selected = selection_rank.iloc[0].to_dict()
    selected_sources = str(selected["source_subset"]).split(";")
    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/material_sorting_energy_scan.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": args.raw_dir,
        "output_dir": args.output_dir,
        "photon_budgets": budgets,
        "max_source_count": args.max_source_count,
        "forced_source_subsets": ["+".join(sources) for sources in forced_subsets],
        "include_gpu": bool(args.include_gpu),
        "best_validation_result": best,
        "selected_validation_result": selected,
        "selection_policy": "rank by min_class_recall, then top1_accuracy, then macro_f1; this avoids repeating the v2 failure mode where mean accuracy improved while one class collapsed",
        "recommended_selected_source_ids": selected_sources,
        "acceptance_targets": ACCEPTANCE_TARGETS,
        "stage_conclusion": "energy_screening_only_requires_selected_rebuild_test_seed",
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    v2.write_manifest(output_dir / "material_sorting_energy_scan_manifest.json", manifest)
    print(f"Wrote energy scan analysis to {output_dir}")
    print(f"Recommended selected sources: {','.join(selected_sources)}")


if __name__ == "__main__":
    main()
