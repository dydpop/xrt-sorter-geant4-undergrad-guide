from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PHOTONS_PER_SAMPLE = 100
TRAIN_SAMPLES_PER_MATERIAL = 25
TEST_SAMPLES_PER_MATERIAL = 25
LABEL_ORDER = ["low_absorption", "high_absorption"]

FEATURE_SETS = {
    "A_threshold_transmission_only": ["primary_transmission_rate"],
    "B_logistic_transmission_only": ["primary_transmission_rate"],
    "C_logistic_transmission_edep_gamma": [
        "primary_transmission_rate",
        "mean_detector_edep_keV",
        "detector_gamma_rate",
    ],
}


@dataclass(frozen=True)
class MaterialRecord:
    material_name: str
    formula: str
    density_g_cm3: float
    category: str
    group_label: str
    event_file: str
    config_file: str
    evidence_status: str
    notes: str


def parse_enabled(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_material_catalog(project_root: Path) -> list[MaterialRecord]:
    catalog_path = project_root / "source_models" / "materials" / "material_catalog.csv"
    catalog_df = pd.read_csv(catalog_path)

    required_columns = {
        "material_name",
        "formula",
        "density_g_cm3",
        "category",
        "group_label",
        "enabled_for_undergrad",
        "event_file",
        "config_file",
        "evidence_status",
        "notes",
    }
    missing = required_columns.difference(catalog_df.columns)
    if missing:
        raise ValueError(f"material_catalog.csv is missing columns: {sorted(missing)}")

    enabled_df = catalog_df[catalog_df["enabled_for_undergrad"].map(parse_enabled)].copy()
    if enabled_df.empty:
        raise ValueError("material_catalog.csv has no enabled undergraduate materials.")

    unknown_labels = sorted(set(enabled_df["group_label"]) - set(LABEL_ORDER))
    if unknown_labels:
        raise ValueError(
            "Unsupported group_label values in material_catalog.csv: "
            f"{unknown_labels}. Expected labels: {LABEL_ORDER}"
        )

    records = []
    for row in enabled_df.itertuples(index=False):
        record = MaterialRecord(
            material_name=str(row.material_name),
            formula=str(row.formula),
            density_g_cm3=float(row.density_g_cm3),
            category=str(row.category),
            group_label=str(row.group_label),
            event_file=str(row.event_file),
            config_file=str(row.config_file),
            evidence_status=str(row.evidence_status),
            notes=str(row.notes),
        )
        config_path = project_root / record.config_file
        if not config_path.exists():
            raise FileNotFoundError(
                f"Catalog material {record.material_name} points to missing config: "
                f"{record.config_file}"
            )
        records.append(record)

    return records


def build_virtual_samples(
    df: pd.DataFrame,
    material: MaterialRecord,
    photons_per_sample: int,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df = df.sort_values("event_id").reset_index(drop=True)

    n_complete_samples = len(df) // photons_per_sample
    df = df.iloc[: n_complete_samples * photons_per_sample].copy()

    df["sample_id"] = df["event_id"] // photons_per_sample

    grouped = (
        df.groupby("sample_id")
        .agg(
            n_events=("event_id", "count"),
            event_id_min=("event_id", "min"),
            event_id_max=("event_id", "max"),
            detector_edep_keV_sum=("detector_edep_keV", "sum"),
            detector_gamma_entries_sum=("detector_gamma_entries", "sum"),
            primary_gamma_entries_sum=("primary_gamma_entries", "sum"),
        )
        .reset_index()
    )

    grouped.insert(0, "material", material.material_name)
    grouped.insert(1, "formula", material.formula)
    grouped.insert(2, "density_g_cm3", material.density_g_cm3)
    grouped.insert(3, "category", material.category)
    grouped["group_label"] = material.group_label
    grouped["mean_detector_edep_keV"] = (
        grouped["detector_edep_keV_sum"] / grouped["n_events"]
    )
    grouped["detector_gamma_rate"] = (
        grouped["detector_gamma_entries_sum"] / grouped["n_events"]
    )
    grouped["primary_transmission_rate"] = (
        grouped["primary_gamma_entries_sum"] / grouped["n_events"]
    )

    return grouped


def build_event_summary(
    material: MaterialRecord,
    df: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    unique_event_count = int(df["event_id"].nunique())
    event_count = int(len(df))
    duplicate_event_count = event_count - unique_event_count
    min_event_id = int(df["event_id"].min())
    max_event_id = int(df["event_id"].max())
    expected_contiguous_count = max_event_id - min_event_id + 1
    return {
        "material": material.material_name,
        "formula": material.formula,
        "density_g_cm3": material.density_g_cm3,
        "category": material.category,
        "group_label": material.group_label,
        "event_file": material.event_file,
        "config_file": material.config_file,
        "evidence_status": material.evidence_status,
        "event_count": event_count,
        "unique_event_count": unique_event_count,
        "duplicate_event_count": duplicate_event_count,
        "min_event_id": min_event_id,
        "max_event_id": max_event_id,
        "event_ids_contiguous": unique_event_count == expected_contiguous_count,
        "photons_per_virtual_sample": PHOTONS_PER_SAMPLE,
        "complete_virtual_samples": int(len(samples)),
        "ignored_tail_events": int(event_count % PHOTONS_PER_SAMPLE),
    }


def split_samples(samples_df: pd.DataFrame, materials: list[MaterialRecord]) -> pd.DataFrame:
    split_parts = []
    for material in materials:
        df_m = (
            samples_df[samples_df["material"] == material.material_name]
            .sort_values("sample_id")
            .copy()
        )
        expected = TRAIN_SAMPLES_PER_MATERIAL + TEST_SAMPLES_PER_MATERIAL
        if len(df_m) != expected:
            raise ValueError(
                f"{material.material_name} has {len(df_m)} virtual samples, "
                f"expected {expected}. Check /run/beamOn and PHOTONS_PER_SAMPLE."
            )
        df_m["split"] = "train"
        df_m.loc[df_m.index[TRAIN_SAMPLES_PER_MATERIAL:], "split"] = "test"
        split_parts.append(df_m)
    return pd.concat(split_parts, ignore_index=True)


def threshold_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    low_mean = train_df[train_df["group_label"] == "low_absorption"][
        "primary_transmission_rate"
    ].mean()
    high_mean = train_df[train_df["group_label"] == "high_absorption"][
        "primary_transmission_rate"
    ].mean()
    threshold = 0.5 * (low_mean + high_mean)

    preds = test_df["primary_transmission_rate"].apply(
        lambda x: "low_absorption" if x > threshold else "high_absorption"
    )

    acc = accuracy_score(test_df["group_label"], preds)
    cm_df = confusion_dataframe(test_df["group_label"], preds)
    return acc, threshold, cm_df, preds


def logistic_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["group_label"]
    y_test = test_df["group_label"]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=test_df.index)

    acc = accuracy_score(y_test, y_pred)
    cm_df = confusion_dataframe(y_test, y_pred)
    return acc, cm_df, y_pred


def confusion_dataframe(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)


def group_members(materials: list[MaterialRecord], group_label: str) -> list[str]:
    return [
        material.material_name
        for material in materials
        if material.group_label == group_label
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    build_dir = project_root / "build"
    results_dir = project_root / "results" / "undergrad_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    materials = load_material_catalog(project_root)

    event_summaries = []
    all_samples = []
    for material in materials:
        file_path = build_dir / material.event_file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {file_path}. Run all enabled material configs listed in "
                "source_models/materials/material_catalog.csv first."
            )
        df = pd.read_csv(file_path)
        samples = build_virtual_samples(df, material, PHOTONS_PER_SAMPLE)
        event_summaries.append(build_event_summary(material, df, samples))
        all_samples.append(samples)

    event_summary_df = pd.DataFrame(event_summaries)
    samples_df = pd.concat(all_samples, ignore_index=True)
    split_df = split_samples(samples_df, materials)
    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    acc_a, threshold, cm_a, preds_a = threshold_classifier(train_df, test_df)
    acc_b, cm_b, preds_b = logistic_classifier(
        train_df, test_df, FEATURE_SETS["B_logistic_transmission_only"]
    )
    acc_c, cm_c, preds_c = logistic_classifier(
        train_df, test_df, FEATURE_SETS["C_logistic_transmission_edep_gamma"]
    )

    summary_df = pd.DataFrame(
        [
            {
                "method": "A_threshold_transmission_only",
                "model_family": "threshold",
                "n_features": 1,
                "feature_columns": "primary_transmission_rate",
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "correct_test_samples": int(round(acc_a * len(test_df))),
                "accuracy": acc_a,
                "threshold": threshold,
            },
            {
                "method": "B_logistic_transmission_only",
                "model_family": "standard_scaler_plus_logistic_regression",
                "n_features": 1,
                "feature_columns": "primary_transmission_rate",
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "correct_test_samples": int(round(acc_b * len(test_df))),
                "accuracy": acc_b,
                "threshold": "",
            },
            {
                "method": "C_logistic_transmission_edep_gamma",
                "model_family": "standard_scaler_plus_logistic_regression",
                "n_features": 3,
                "feature_columns": (
                    "primary_transmission_rate;mean_detector_edep_keV;"
                    "detector_gamma_rate"
                ),
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "correct_test_samples": int(round(acc_c * len(test_df))),
                "accuracy": acc_c,
                "threshold": "",
            },
        ]
    )

    feature_summary_df = (
        split_df.groupby(["group_label", "split"])[
            ["primary_transmission_rate", "mean_detector_edep_keV", "detector_gamma_rate"]
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    feature_summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in feature_summary_df.columns
    ]

    material_feature_summary_df = (
        split_df.groupby(["material", "formula", "group_label"])[
            ["primary_transmission_rate", "mean_detector_edep_keV", "detector_gamma_rate"]
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    material_feature_summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in material_feature_summary_df.columns
    ]

    prediction_frames = []
    for method, preds in [
        ("A_threshold_transmission_only", preds_a),
        ("B_logistic_transmission_only", preds_b),
        ("C_logistic_transmission_edep_gamma", preds_c),
    ]:
        frame = test_df[
            [
                "material",
                "formula",
                "sample_id",
                "group_label",
                "primary_transmission_rate",
                "mean_detector_edep_keV",
                "detector_gamma_rate",
            ]
        ].copy()
        frame.insert(0, "method", method)
        frame["predicted_group_label"] = preds.values
        frame["is_correct"] = frame["group_label"] == frame["predicted_group_label"]
        prediction_frames.append(frame)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    event_summary_df.to_csv(results_dir / "event_row_summary.csv", index=False)
    samples_df.to_csv(results_dir / "absorption_group_virtual_samples.csv", index=False)
    split_df.to_csv(results_dir / "train_test_split_samples.csv", index=False)
    feature_summary_df.to_csv(results_dir / "feature_group_summary.csv", index=False)
    material_feature_summary_df.to_csv(
        results_dir / "material_feature_summary.csv", index=False
    )
    summary_df.to_csv(results_dir / "absorption_group_classification_summary.csv", index=False)
    predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)
    cm_a.to_csv(results_dir / "absorption_group_confusion_threshold.csv")
    cm_b.to_csv(results_dir / "absorption_group_confusion_logistic_1f.csv")
    cm_c.to_csv(results_dir / "absorption_group_confusion_logistic_3f.csv")

    # Keep the original result entry points fresh for readers who open results/ first.
    summary_df[["method", "n_features", "accuracy"]].to_csv(
        project_root / "results" / "absorption_group_classification_summary.csv",
        index=False,
    )
    cm_a.to_csv(project_root / "results" / "absorption_group_confusion_threshold.csv")
    cm_b.to_csv(project_root / "results" / "absorption_group_confusion_logistic_1f.csv")
    cm_c.to_csv(project_root / "results" / "absorption_group_confusion_logistic_3f.csv")

    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/classify_absorption_groups.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_event_dir": "build",
        "output_dir": "results/undergrad_validation",
        "catalog_file": "source_models/materials/material_catalog.csv",
        "simulation_scope": {
            "materials": [material.material_name for material in materials],
            "enabled_material_count": len(materials),
            "classification_target": "coarse absorption group",
            "low_absorption": group_members(materials, "low_absorption"),
            "high_absorption": group_members(materials, "high_absorption"),
            "geometry_summary": (
                "catalog-enabled single-material slab configurations, "
                "10 mm ore thickness, W-target 120 kV spectrum"
            ),
            "catalog_note": (
                "The material catalog drives the undergraduate validation set. "
                "Adding a material still requires a Geant4 material definition, "
                "a config file, and regenerated evidence before claims are updated."
            ),
        },
        "sample_policy": {
            "photons_per_virtual_sample": PHOTONS_PER_SAMPLE,
            "train_samples_per_material": TRAIN_SAMPLES_PER_MATERIAL,
            "test_samples_per_material": TEST_SAMPLES_PER_MATERIAL,
            "total_virtual_samples": int(len(split_df)),
            "total_train_samples": int(len(train_df)),
            "total_test_samples": int(len(test_df)),
        },
        "material_catalog": [material.__dict__ for material in materials],
        "event_summary": event_summaries,
        "methods": summary_df.to_dict(orient="records"),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "claim_boundary": [
            "The evidence is simulated data from the undergraduate public configurations enabled in the material catalog.",
            "Accuracy is evaluated on a same-distribution simulated test split, not on physical equipment data.",
            "The result does not prove general classification of all minerals, mixed ore streams, or industrial deployment.",
            "Catalog-driven extension means the workflow is easier to expand; each new material still needs simulation evidence before claims are updated.",
            "Geant4 simulation is stochastic; regenerating event CSV files without fixed random seeds can produce small numerical differences.",
        ],
    }
    (results_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n================ Absorption Group Validation Evidence ================\n")
    print(event_summary_df[["material", "event_count", "complete_virtual_samples"]])
    print("\nTrain/test samples:")
    print(split_df.groupby(["group_label", "split"]).size())
    print("\nClassification summary:")
    print(summary_df[["method", "test_samples", "correct_test_samples", "accuracy"]])
    print(f"\nSaved validation package to: {results_dir.relative_to(project_root)}")
    print("\n=====================================================================\n")


if __name__ == "__main__":
    main()
