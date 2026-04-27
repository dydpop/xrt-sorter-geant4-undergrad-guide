from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PHOTONS_PER_SAMPLE = 100
ENERGY_WINDOWS = {
    "e_lt40": (None, 40.0),
    "e_40_80": (40.0, 80.0),
    "e_ge80": (80.0, None),
}
TARGET_MATERIALS = [
    "Quartz",
    "Calcite",
    "Orthoclase",
    "Albite",
    "Dolomite",
    "Pyrite",
    "Hematite",
    "Magnetite",
    "Chalcopyrite",
    "Galena",
]
REVIEW_PROBABILITY_THRESHOLD = 0.65
REVIEW_MARGIN_THRESHOLD = 0.15
DISTANCE_QUANTILE = 0.95
DISTANCE_MULTIPLIER = 1.25


def require_sklearn():
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "material_sorting.py requires scikit-learn. Install with: "
            "pip install pandas scikit-learn"
        ) from exc

    return {
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "make_pipeline": make_pipeline,
        "recall_score": recall_score,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def source_id_from_metadata(meta: dict) -> str:
    mode = str(meta.get("source_mode", "unknown"))
    if mode == "mono":
        return f"mono_{int(float(meta.get('mono_energy_keV', 0)))}kev"
    if mode == "spectrum":
        return "spectrum_120kv"
    return mode


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog(project_root: Path) -> pd.DataFrame:
    path = project_root / "source_models" / "materials" / "material_catalog.csv"
    df = pd.read_csv(path)
    enabled = df[df["enabled_for_undergrad"].astype(str).str.lower().isin(["true", "1", "yes"])]
    missing = sorted(set(TARGET_MATERIALS) - set(enabled["material_name"]))
    if missing:
        raise ValueError(f"Material catalog is missing enabled targets: {missing}")
    return enabled


def legacy_run_records(project_root: Path) -> list[dict]:
    records = []
    build_dir = project_root / "build"
    catalog = load_catalog(project_root)
    for row in catalog.itertuples(index=False):
        event_file = build_dir / str(row.event_file)
        hit_file = build_dir / str(row.event_file).replace("_events.csv", "_hits.csv")
        if not event_file.exists():
            continue
        records.append(
            {
                "profile": "legacy_undergrad_10mm_spectrum",
                "material": str(row.material_name),
                "formula": str(row.formula),
                "category": str(row.category),
                "event_file": event_file,
                "hit_file": hit_file,
                "source_id": "spectrum_120kv",
                "source_mode": "spectrum",
                "mono_energy_keV": 80.0,
                "thickness_mm": 10.0,
                "random_seed": 0,
                "run_id": f"legacy_{str(row.material_name).lower()}",
            }
        )
    return records


def matrix_run_records(project_root: Path, raw_dir: Path) -> list[dict]:
    records = []
    for meta_path in sorted(raw_dir.rglob("*_metadata.json")):
        meta = read_metadata(meta_path)
        if str(meta.get("run_role", "")).lower() == "calibration":
            continue
        if str(meta.get("ore_material_mode", "")).lower() == "air_path":
            continue
        material = str(meta.get("ore_primary_material", ""))
        if material not in TARGET_MATERIALS:
            continue
        records.append(
            {
                "profile": "material_sorting_matrix",
                "material": material,
                "formula": "",
                "category": "",
                "event_file": Path(meta["event_file"]),
                "hit_file": Path(meta["hit_file"]),
                "source_id": source_id_from_metadata(meta),
                "source_mode": str(meta.get("source_mode", "")),
                "mono_energy_keV": float(meta.get("mono_energy_keV", 0.0)),
                "thickness_mm": float(meta.get("ore_thickness_mm", 0.0)),
                "random_seed": int(meta.get("random_seed", -1)),
                "run_id": str(meta.get("run_id", meta_path.stem)),
            }
        )
    return records


def build_run_samples(record: dict) -> pd.DataFrame:
    events = pd.read_csv(record["event_file"]).sort_values("event_id").reset_index(drop=True)
    complete = len(events) // PHOTONS_PER_SAMPLE
    events = events.iloc[: complete * PHOTONS_PER_SAMPLE].copy()
    if events.empty:
        raise ValueError(f"No complete samples in {record['event_file']}")
    events["sample_id"] = events["event_id"] // PHOTONS_PER_SAMPLE
    event_group = (
        events.groupby("sample_id")
        .agg(
            n_events=("event_id", "count"),
            event_id_min=("event_id", "min"),
            event_id_max=("event_id", "max"),
            edep_sum=("detector_edep_keV", "sum"),
            edep_mean=("detector_edep_keV", "mean"),
            edep_std=("detector_edep_keV", "std"),
            edep_max=("detector_edep_keV", "max"),
            detector_gamma_sum=("detector_gamma_entries", "sum"),
            primary_gamma_sum=("primary_gamma_entries", "sum"),
            nonzero_edep_rate=("detector_edep_keV", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    event_group["edep_std"] = event_group["edep_std"].fillna(0.0)
    event_group["detector_gamma_rate"] = event_group["detector_gamma_sum"] / event_group["n_events"]
    event_group["primary_transmission_rate"] = (
        event_group["primary_gamma_sum"] / event_group["n_events"]
    )

    hit_file = Path(record["hit_file"])
    if hit_file.exists() and hit_file.stat().st_size > 100:
        hits = pd.read_csv(hit_file)
        hits["sample_id"] = hits["event_id"] // PHOTONS_PER_SAMPLE
        hits["r_mm"] = np.sqrt(hits["y_mm"] ** 2 + hits["z_mm"] ** 2)
        for name, (low, high) in ENERGY_WINDOWS.items():
            mask = pd.Series(True, index=hits.index)
            if low is not None:
                mask &= hits["photon_energy_keV"] >= low
            if high is not None:
                mask &= hits["photon_energy_keV"] < high
            hits[name] = mask.astype(int)
        hit_group = (
            hits.groupby("sample_id")
            .agg(
                hit_count=("event_id", "count"),
                hit_energy_mean=("photon_energy_keV", "mean"),
                hit_energy_std=("photon_energy_keV", "std"),
                hit_energy_min=("photon_energy_keV", "min"),
                hit_energy_max=("photon_energy_keV", "max"),
                primary_hit_rate=("is_primary", "mean"),
                direct_primary_count=("is_direct_primary", "sum"),
                scattered_primary_count=("is_scattered_primary", "sum"),
                theta_mean=("theta_deg", "mean"),
                theta_std=("theta_deg", "std"),
                y_mean=("y_mm", "mean"),
                y_std=("y_mm", "std"),
                z_mean=("z_mm", "mean"),
                z_std=("z_mm", "std"),
                r_mean=("r_mm", "mean"),
                r_std=("r_mm", "std"),
                e_lt40_count=("e_lt40", "sum"),
                e_40_80_count=("e_40_80", "sum"),
                e_ge80_count=("e_ge80", "sum"),
            )
            .reset_index()
        )
    else:
        hit_group = pd.DataFrame({"sample_id": event_group["sample_id"]})

    frame = event_group.merge(hit_group, on="sample_id", how="left").fillna(0.0)
    frame.insert(0, "material", record["material"])
    frame.insert(1, "source_id", record["source_id"])
    frame.insert(2, "thickness_mm", float(record["thickness_mm"]))
    frame.insert(3, "random_seed", int(record["random_seed"]))
    frame.insert(4, "run_id", record["run_id"])
    for col in ["hit_count", "direct_primary_count", "scattered_primary_count", "e_lt40_count", "e_40_80_count", "e_ge80_count"]:
        if col in frame:
            frame[f"{col}_rate"] = frame[col] / frame["n_events"]
    frame["scatter_direct_ratio"] = (
        frame.get("scattered_primary_count", 0.0) / (frame.get("direct_primary_count", 0.0) + 1e-6)
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "material",
        "formula",
        "category",
        "source_id",
        "run_id",
        "sample_id",
        "event_id_min",
        "event_id_max",
        "random_seed",
    }
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) and float(frame[col].std()) > 1e-12:
            cols.append(col)
    return cols


def sanitize_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    clean = frame.copy()
    before_nan = int(clean[feature_cols].isna().sum().sum()) if feature_cols else 0
    before_inf = int(np.isinf(clean[feature_cols].to_numpy(dtype=float)).sum()) if feature_cols else 0
    if feature_cols:
        clean[feature_cols] = clean[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return clean, {"nan_replaced": before_nan, "inf_replaced": before_inf}


def fused_table(samples: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    source_count = samples["source_id"].nunique()
    if source_count < 2:
        return samples.copy(), "single_source"
    keys = ["material", "thickness_mm", "random_seed", "sample_id"]
    cols = feature_columns(samples)
    pieces = []
    for source_id, part in samples.groupby("source_id"):
        renamed = part[keys + cols].copy()
        renamed = renamed.rename(columns={col: f"{source_id}__{col}" for col in cols})
        pieces.append(renamed)
    fused = pieces[0]
    for piece in pieces[1:]:
        fused = fused.merge(piece, on=keys, how="inner")
    if fused.empty:
        return samples.copy(), "single_source_unpaired"
    if {
        "mono_60kev__primary_transmission_rate",
        "mono_100kev__primary_transmission_rate",
    }.issubset(fused.columns):
        fused["dual_energy_transmission_ratio_60_100"] = (
            fused["mono_60kev__primary_transmission_rate"] + 1e-6
        ) / (fused["mono_100kev__primary_transmission_rate"] + 1e-6)
        fused["dual_energy_log_ratio_60_100"] = np.log(
            fused["dual_energy_transmission_ratio_60_100"]
        )
    return fused, "multi_source_fused"


def split_samples(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    seeds = sorted(seed for seed in frame["random_seed"].unique() if seed >= 0)
    if len(seeds) >= 2:
        holdout = seeds[-1]
        return (
            frame[frame["random_seed"] != holdout].copy(),
            frame[frame["random_seed"] == holdout].copy(),
            f"seed_holdout_seed_{holdout}",
        )

    train_parts = []
    test_parts = []
    for _, part in frame.groupby("material"):
        part = part.sort_values(["thickness_mm", "sample_id"]).reset_index(drop=True)
        cut = max(1, len(part) // 2)
        train_parts.append(part.iloc[:cut])
        test_parts.append(part.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts), "within_run_half_split"


def build_models(sk):
    ExtraTreesClassifier = sk["ExtraTreesClassifier"]
    RandomForestClassifier = sk["RandomForestClassifier"]
    LogisticRegression = sk["LogisticRegression"]
    StandardScaler = sk["StandardScaler"]
    SVC = sk["SVC"]
    make_pipeline = sk["make_pipeline"]
    CalibratedClassifierCV = sk["CalibratedClassifierCV"]

    extra = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
    )
    try:
        calibrated_extra = CalibratedClassifierCV(estimator=extra, cv=3)
    except TypeError:
        calibrated_extra = CalibratedClassifierCV(base_estimator=extra, cv=3)
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=1,
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced")
        ),
        "calibrated_extra_trees": calibrated_extra,
    }


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, k: int) -> float:
    order = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    hits = []
    for true, indices in zip(y_true, order):
        hits.append(true in classes[indices])
    return float(np.mean(hits))


def fit_centroid_gate(train: pd.DataFrame, feature_cols: list[str]):
    X = train[feature_cols].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    z = (X - mean) / std
    y = train["material"].to_numpy()
    centroids = {material: z[y == material].mean(axis=0) for material in sorted(set(y))}
    distances = []
    for material, row in zip(y, z):
        distances.append(float(np.linalg.norm(row - centroids[material])))
    threshold = float(np.quantile(distances, DISTANCE_QUANTILE) * DISTANCE_MULTIPLIER)
    return mean, std, centroids, threshold


def centroid_distances(test: pd.DataFrame, feature_cols: list[str], gate) -> np.ndarray:
    mean, std, centroids, _ = gate
    z = (test[feature_cols].to_numpy(dtype=float) - mean) / std
    distances = []
    for row in z:
        distances.append(min(float(np.linalg.norm(row - centroid)) for centroid in centroids.values()))
    return np.array(distances)


def write_csv(frame: pd.DataFrame | pd.Series, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, lineterminator="\n")


def decision_frame(test: pd.DataFrame, probabilities: np.ndarray, classes: np.ndarray, gate, feature_cols: list[str]) -> pd.DataFrame:
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    top3_idx = order[:, :3]
    top1_prob = probabilities[np.arange(len(probabilities)), top1_idx]
    top2_prob = probabilities[np.arange(len(probabilities)), top2_idx]
    margin = top1_prob - top2_prob
    distances = centroid_distances(test, feature_cols, gate)
    _, _, _, distance_threshold = gate

    rows = []
    for i, (_, sample) in enumerate(test.reset_index(drop=True).iterrows()):
        reasons = []
        if top1_prob[i] < REVIEW_PROBABILITY_THRESHOLD:
            reasons.append("low_probability")
        if margin[i] < REVIEW_MARGIN_THRESHOLD:
            reasons.append("small_top1_top2_margin")
        if distances[i] > distance_threshold:
            reasons.append("far_from_training_centroid")
        decision = "auto_sort" if not reasons else "review_unknown_or_ambiguous"
        rows.append(
            {
                "material": sample["material"],
                "predicted_material": classes[top1_idx[i]],
                "top1_probability": top1_prob[i],
                "top2_probability": top2_prob[i],
                "margin": margin[i],
                "centroid_distance": distances[i],
                "decision": decision,
                "review_reason": ";".join(reasons) if reasons else "",
                "top3_candidates": ";".join(classes[top3_idx[i]]),
                "is_correct": sample["material"] == classes[top1_idx[i]],
            }
        )
    return pd.DataFrame(rows)


def evaluate_models(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, str]:
    sk = require_sklearn()
    models = build_models(sk)
    accuracy_score = sk["accuracy_score"]
    confusion_matrix = sk["confusion_matrix"]
    f1_score = sk["f1_score"]
    recall_score = sk["recall_score"]

    X_train = train[feature_cols]
    y_train = train["material"].to_numpy()
    X_test = test[feature_cols]
    y_test = test["material"].to_numpy()
    labels = np.array(TARGET_MATERIALS)
    summaries = []
    confusions = {}
    main_decisions = None
    main_method = "calibrated_extra_trees"
    gate = fit_centroid_gate(train, feature_cols)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            classes = np.array(model.classes_)
        else:
            probabilities = np.zeros((len(X_test), len(labels)))
            classes = labels
        top3 = topk_accuracy(y_test, probabilities, classes, 3) if probabilities.size else math.nan
        recalls = recall_score(y_test, predictions, labels=labels, average=None, zero_division=0)
        summary = {
            "method": name,
            "train_samples": len(train),
            "test_samples": len(test),
            "feature_count": len(feature_cols),
            "top1_accuracy": accuracy_score(y_test, predictions),
            "top3_accuracy": top3,
            "macro_f1": f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0),
            "min_class_recall": float(np.min(recalls)),
        }
        summaries.append(summary)
        cm = pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=labels),
            index=labels,
            columns=labels,
        )
        confusions[name] = cm
        write_csv(cm, output_dir / f"material_confusion_{name}.csv", index=True)
        if name == main_method:
            main_decisions = decision_frame(test, probabilities, classes, gate, feature_cols)

    if main_decisions is None:
        raise RuntimeError("Main material sorting method did not run.")
    return pd.DataFrame(summaries), confusions, main_decisions, main_method


def leave_one_material_out(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    sk = require_sklearn()
    rows = []
    for material in TARGET_MATERIALS:
        train = frame[frame["material"] != material].copy()
        test = frame[frame["material"] == material].copy()
        if train.empty or test.empty:
            continue
        model = build_models(sk)["calibrated_extra_trees"]
        model.fit(train[feature_cols], train["material"])
        probabilities = model.predict_proba(test[feature_cols])
        classes = np.array(model.classes_)
        gate = fit_centroid_gate(train, feature_cols)
        decisions = decision_frame(test, probabilities, classes, gate, feature_cols)
        rows.append(
            {
                "held_out_material": material,
                "test_samples": len(test),
                "review_or_unknown_samples": int((decisions["decision"] != "auto_sort").sum()),
                "open_set_review_recall": float((decisions["decision"] != "auto_sort").mean()),
                "forced_auto_sort_samples": int((decisions["decision"] == "auto_sort").sum()),
            }
        )
    return pd.DataFrame(rows)


def acceptance_status(summary: pd.DataFrame, decisions: pd.DataFrame, open_set: pd.DataFrame) -> dict:
    main = summary[summary["method"] == "calibrated_extra_trees"].iloc[0].to_dict()
    auto = decisions[decisions["decision"] == "auto_sort"]
    auto_precision = float(auto["is_correct"].mean()) if len(auto) else 0.0
    review_rate = float((decisions["decision"] != "auto_sort").mean())
    open_recall = float(open_set["open_set_review_recall"].mean()) if len(open_set) else 0.0
    criteria = {
        "closed_set_top1_accuracy_ge_0_85": main["top1_accuracy"] >= 0.85,
        "closed_set_macro_f1_ge_0_80": main["macro_f1"] >= 0.80,
        "closed_set_min_recall_ge_0_70": main["min_class_recall"] >= 0.70,
        "closed_set_top3_accuracy_ge_0_95": main["top3_accuracy"] >= 0.95,
        "auto_sort_precision_ge_0_90": auto_precision >= 0.90,
        "review_rate_le_0_30": review_rate <= 0.30,
        "open_set_review_recall_ge_0_90": open_recall >= 0.90,
    }
    return {
        "main_method": "calibrated_extra_trees",
        "main_metrics": main,
        "auto_sort_precision": auto_precision,
        "review_rate": review_rate,
        "mean_open_set_review_recall": open_recall,
        "criteria": criteria,
        "all_criteria_met": all(criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Material-level XRT sorting with review gate.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs", help="Matrix raw output directory.")
    parser.add_argument("--output-dir", default="results/material_sorting")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = project_root / args.raw_dir
    try:
        raw_dir_label = raw_dir.relative_to(project_root).as_posix()
    except ValueError:
        raw_dir_label = str(raw_dir)

    records = matrix_run_records(project_root, raw_dir) if raw_dir.exists() else []
    matrix_record_count = len(records)
    matrix_materials = sorted({record["material"] for record in records})
    data_source = "material_sorting_matrix"
    if set(matrix_materials) != set(TARGET_MATERIALS):
        records = legacy_run_records(project_root)
        data_source = "legacy_undergrad_10mm_spectrum"
    if not records:
        raise FileNotFoundError("No material sorting raw runs found and no legacy build files are available.")

    samples = pd.concat([build_run_samples(record) for record in records], ignore_index=True)
    model_table, table_mode = fused_table(samples)
    feature_cols = feature_columns(model_table)
    model_table, qc_summary = sanitize_feature_frame(model_table, feature_cols)
    train, test, split_strategy = split_samples(model_table)

    write_csv(samples, output_dir / "material_virtual_samples_long.csv")
    write_csv(model_table, output_dir / "material_model_table.csv")
    write_csv(pd.Series(feature_cols, name="feature"), output_dir / "material_feature_columns.csv")

    summary, _, decisions, main_method = evaluate_models(train, test, feature_cols, output_dir)
    open_set = leave_one_material_out(model_table, feature_cols)
    status = acceptance_status(summary, decisions, open_set)

    write_csv(summary, output_dir / "material_sorting_summary.csv")
    write_csv(decisions, output_dir / "material_sorting_decisions.csv")
    write_csv(open_set, output_dir / "open_set_leave_one_material_out.csv")

    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/material_sorting.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
        "matrix_raw_status": {
            "raw_dir": raw_dir_label,
            "metadata_files_found": matrix_record_count,
            "materials_found": matrix_materials,
            "complete_target_material_set": set(matrix_materials) == set(TARGET_MATERIALS),
        },
        "table_mode": table_mode,
        "split_strategy": split_strategy,
        "target": "ten known material classes plus review gate",
        "materials": TARGET_MATERIALS,
        "sample_policy": {
            "photons_per_virtual_sample": PHOTONS_PER_SAMPLE,
            "raw_runs": len(records),
            "long_samples": len(samples),
            "model_rows": len(model_table),
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "feature_policy": {
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "numeric_qc": qc_summary,
            "excluded_from_model": [
                "material",
                "formula",
                "density_g_cm3",
                "category",
                "random_seed",
                "run_id",
            ],
        },
        "review_gate": {
            "top1_probability_threshold": REVIEW_PROBABILITY_THRESHOLD,
            "top1_top2_margin_threshold": REVIEW_MARGIN_THRESHOLD,
            "distance_quantile": DISTANCE_QUANTILE,
            "distance_multiplier": DISTANCE_MULTIPLIER,
        },
        "main_method": main_method,
        "acceptance_status": status,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "claim_boundary": [
            "This is a diagnostic material-level sorting attempt for the configured ten materials.",
            "The current result does not pass the predefined material-level acceptance criteria.",
            "The review gate is an experimental safeguard against forced classification of low-confidence or unknown-like samples.",
            "It does not prove all-mineral identification, real-device performance, mixed ore sorting, or industrial deployment.",
        ],
    }
    with (output_dir / "material_sorting_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print("\n================ Material Sorting Evidence ================\n")
    print(f"Data source: {data_source}")
    print(f"Table mode: {table_mode}")
    print(f"Split: {split_strategy}")
    print(summary[["method", "top1_accuracy", "top3_accuracy", "macro_f1", "min_class_recall"]])
    print("\nAcceptance status:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"\nSaved material sorting package to: {output_dir.relative_to(project_root)}")
    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
