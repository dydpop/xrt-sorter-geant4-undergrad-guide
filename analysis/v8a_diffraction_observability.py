from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HM_PAIR = ("Hematite", "Magnetite")
PASS_FEATURE_AUC = 0.95
PASS_FEATURE_D_PRIME = 3.0
PASS_HM_MIN_RECALL = 0.80
PASS_THICKNESS_BLIND_HM_MIN_RECALL = 0.78
MAX_CONTROL_HM_MIN_RECALL = 0.75
MAX_SHUFFLED_LABEL_HM_MIN_RECALL = 0.65
MAX_OVERLAP_ONLY_HM_MIN_RECALL = 0.75
NO_GO_HM_MIN_RECALL = 0.75
EPS = 1e-9

# Approximate Cu K-alpha powder-peak anchors from the project literature scan.
# Intensities are relative prototype weights, not a quantitative reference pattern.
POWDER_PEAKS: dict[str, list[tuple[float, float]]] = {
    "Hematite": [
        (24.1, 0.45),
        (33.2, 1.00),
        (35.6, 0.72),
        (40.9, 0.24),
        (49.5, 0.46),
        (54.1, 0.55),
        (57.5, 0.36),
        (62.5, 0.42),
        (64.0, 0.28),
    ],
    "Magnetite": [
        (18.3, 0.18),
        (30.1, 0.55),
        (35.5, 1.00),
        (37.0, 0.22),
        (43.1, 0.68),
        (53.4, 0.48),
        (57.0, 0.50),
        (62.6, 0.38),
        (74.0, 0.16),
    ],
}

UNIQUE_WINDOWS = {
    "hematite_unique": [24.1, 33.2, 40.9, 49.5, 54.1, 64.0],
    "magnetite_unique": [18.3, 30.1, 37.0, 43.1, 53.4, 74.0],
    "overlap": [35.55, 57.25, 62.55],
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def is_overlap_feature(column: str) -> bool:
    overlap_tokens = ("35p5", "35p55", "35p6", "57p0", "57p25", "57p5", "62p5", "62p55", "62p6", "signature_overlap")
    overlap_bins = ("bin_35_36", "bin_57_58", "bin_62_63")
    return any(token in column for token in overlap_tokens) or any(token in column for token in overlap_bins)


def is_coalesced_overlap_feature(column: str) -> bool:
    overlap_bins = ("bin_35_36", "bin_57_58", "bin_62_63")
    return column.startswith("signature_overlap") or any(token in column for token in overlap_bins)


def require_sklearn() -> dict:
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-learn is required for v8A diffraction observability.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "roc_auc_score": roc_auc_score,
    }


def make_axis(two_theta_min: float, two_theta_max: float, bin_width: float) -> np.ndarray:
    bins = int(round((two_theta_max - two_theta_min) / bin_width)) + 1
    return np.linspace(two_theta_min, two_theta_max, bins, dtype=np.float64)


def gaussian(axis: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / max(sigma, EPS)) ** 2)


def stable_rng(random_seed: int, material: str, split_seed: int, thickness: float, pose_index: int) -> np.random.Generator:
    material_offset = 100000 if material == "Magnetite" else 0
    thickness_offset = int(round(thickness * 100.0))
    seed = random_seed + material_offset + split_seed * 101 + thickness_offset * 17 + pose_index * 1009
    return np.random.default_rng(seed)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite to replace prototype artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)


def simulate_spectrum(
    *,
    axis: np.ndarray,
    material: str,
    thickness_mm: float,
    rng: np.random.Generator,
    peak_width_deg: float,
    angle_jitter_deg: float,
    orientation_sigma: float,
    background_level: float,
    background_slope_sigma: float,
    attenuation_strength: float,
    counts_scale: float,
    read_noise_sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    background_scale = rng.lognormal(mean=0.0, sigma=0.12)
    slope = rng.normal(0.0, background_slope_sigma)
    centered_axis = (axis - float(axis.mean())) / max(float(np.ptp(axis)), EPS)
    spectrum = background_level * background_scale * (1.0 + slope * centered_axis)
    spectrum = np.clip(spectrum, background_level * 0.15, None)

    global_shift = rng.normal(0.0, angle_jitter_deg)
    signal_scale = rng.lognormal(mean=0.0, sigma=0.22) * math.exp(-attenuation_strength * thickness_mm)
    peak_width = max(peak_width_deg * rng.lognormal(mean=0.0, sigma=0.12), 0.035)
    for center, relative_intensity in POWDER_PEAKS[material]:
        peak_shift = global_shift + rng.normal(0.0, angle_jitter_deg * 0.35)
        oriented_intensity = relative_intensity * rng.lognormal(mean=0.0, sigma=orientation_sigma)
        spectrum += signal_scale * oriented_intensity * gaussian(axis, center + peak_shift, peak_width)

    expected_counts = np.clip(spectrum * counts_scale, 0.0, None)
    noisy = rng.poisson(expected_counts) / max(counts_scale, EPS)
    noisy = noisy + rng.normal(0.0, read_noise_sigma, size=noisy.shape)
    noisy = np.clip(noisy, 0.0, None)
    controls = {
        "raw_total_intensity": float(noisy.sum()),
        "raw_mean_intensity": float(noisy.mean()),
        "raw_max_intensity": float(noisy.max()),
        "estimated_background": float(np.quantile(noisy, 0.10)),
        "signal_scale_proxy": float(signal_scale),
        "global_angle_shift_deg": float(global_shift),
    }
    return noisy.astype(np.float64), controls


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    baseline = float(np.quantile(spectrum, 0.08))
    clipped = np.clip(spectrum - baseline, 0.0, None)
    area = float(np.trapezoid(clipped))
    if area <= EPS:
        return clipped
    return clipped / area


def window_feature(axis: np.ndarray, spectrum: np.ndarray, center: float, half_width: float, background_width: float) -> dict[str, float]:
    peak_mask = np.abs(axis - center) <= half_width
    shoulder_mask = (np.abs(axis - center) > half_width) & (np.abs(axis - center) <= background_width)
    if not peak_mask.any():
        return {"area": 0.0, "max": 0.0, "contrast": 0.0}
    peak_values = spectrum[peak_mask]
    shoulder = float(np.median(spectrum[shoulder_mask])) if shoulder_mask.any() else 0.0
    area = float(np.sum(np.clip(peak_values - shoulder, 0.0, None)))
    max_value = float(np.max(peak_values))
    contrast = float(max_value - shoulder)
    return {"area": area, "max": max_value, "contrast": contrast}


def extract_features(axis: np.ndarray, spectrum: np.ndarray) -> dict[str, float]:
    norm = normalize_spectrum(spectrum)
    features: dict[str, float] = {}
    all_peak_centers: list[tuple[str, float]] = []
    for material, peaks in POWDER_PEAKS.items():
        prefix = "h" if material == "Hematite" else "m"
        for center, _ in peaks:
            all_peak_centers.append((prefix, center))
    for prefix, center in all_peak_centers:
        values = window_feature(axis, norm, center, half_width=0.30, background_width=0.95)
        center_name = f"{center:.2f}".replace(".", "p")
        for metric, value in values.items():
            features[f"peak_{prefix}_{center_name}_{metric}"] = value

    for family, centers in UNIQUE_WINDOWS.items():
        areas = [window_feature(axis, norm, center, half_width=0.32, background_width=0.95)["area"] for center in centers]
        contrasts = [window_feature(axis, norm, center, half_width=0.32, background_width=0.95)["contrast"] for center in centers]
        features[f"signature_{family}_area_sum"] = float(np.sum(areas))
        features[f"signature_{family}_contrast_sum"] = float(np.sum(contrasts))

    h_area = features["signature_hematite_unique_area_sum"]
    m_area = features["signature_magnetite_unique_area_sum"]
    features["signature_h_over_m_log_ratio"] = float(math.log((h_area + EPS) / (m_area + EPS)))
    features["signature_h_minus_m_area"] = float(h_area - m_area)
    features["signature_overlap_area_sum"] = features["signature_overlap_area_sum"]

    # Coarse spectrum shape bins preserve peak locations without using total count scale.
    for start in np.arange(math.floor(float(axis.min())), math.ceil(float(axis.max())), 1.0):
        mask = (axis >= start) & (axis < start + 1.0)
        if mask.any():
            features[f"bin_{int(start):02d}_{int(start + 1):02d}_area"] = float(norm[mask].sum())
    return features


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    axis = make_axis(args.two_theta_min, args.two_theta_max, args.bin_width_deg)
    rows: list[dict] = []
    config = {
        "two_theta_min": args.two_theta_min,
        "two_theta_max": args.two_theta_max,
        "bin_width_deg": args.bin_width_deg,
        "peak_width_deg": args.peak_width_deg,
        "angle_jitter_deg": args.angle_jitter_deg,
        "orientation_sigma": args.orientation_sigma,
        "background_level": args.background_level,
        "background_slope_sigma": args.background_slope_sigma,
        "attenuation_strength": args.attenuation_strength,
        "counts_scale": args.counts_scale,
        "read_noise_sigma": args.read_noise_sigma,
    }
    for split, seeds in {"train": parse_csv_ints(args.train_seeds), "validation": parse_csv_ints(args.validation_seeds)}.items():
        for split_seed in seeds:
            for material in HM_PAIR:
                for thickness in parse_csv_floats(args.thickness_list):
                    for pose_index in range(args.poses_per_condition):
                        rng = stable_rng(args.random_seed, material, split_seed, thickness, pose_index)
                        spectrum, controls = simulate_spectrum(
                            axis=axis,
                            material=material,
                            thickness_mm=thickness,
                            rng=rng,
                            peak_width_deg=args.peak_width_deg,
                            angle_jitter_deg=args.angle_jitter_deg,
                            orientation_sigma=args.orientation_sigma,
                            background_level=args.background_level,
                            background_slope_sigma=args.background_slope_sigma,
                            attenuation_strength=args.attenuation_strength,
                            counts_scale=args.counts_scale,
                            read_noise_sigma=args.read_noise_sigma,
                        )
                        features = extract_features(axis, spectrum)
                        row = {
                            "protocol": "v8A_diffraction_sidecar_pilot",
                            "development_only": True,
                            "shadow_or_final_used": False,
                            "split": split,
                            "material": material,
                            "random_seed": split_seed,
                            "thickness_mm": thickness,
                            "pose_index": pose_index,
                            "sample_id": f"v8a_{split}_{material}_{split_seed}_{thickness:g}_{pose_index}",
                            **controls,
                            **features,
                        }
                        rows.append(row)
    frame = pd.DataFrame(rows)
    manifest = {
        "protocol_name": "v8A_diffraction_sidecar_pilot",
        "development_only": True,
        "shadow_or_final_used": False,
        "input_data": "hardcoded_reference_powder_peak_table_only",
        "reads_existing_xrt_cubes": False,
        "reads_shadow_or_final": False,
        "sample_count": int(len(frame)),
        "train_sample_count": int(frame["split"].eq("train").sum()),
        "validation_sample_count": int(frame["split"].eq("validation").sum()),
        "materials": list(HM_PAIR),
        "powder_peaks_two_theta_deg": {material: [{"two_theta_deg": c, "relative_weight": w} for c, w in peaks] for material, peaks in POWDER_PEAKS.items()},
        "synthetic_config": config,
    }
    return frame, manifest


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = {}
    for material in HM_PAIR:
        mask = y_true == material
        recalls[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return recalls


def evaluate_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    method_name: str,
    estimator,
    *,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 8811,
) -> tuple[dict, pd.DataFrame]:
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("validation").to_numpy()
    x_train = frame.loc[train, feature_cols].to_numpy(dtype=np.float64)
    x_validation = frame.loc[validation, feature_cols].to_numpy(dtype=np.float64)
    y_train = frame.loc[train, "material"].astype(str).to_numpy()
    y_validation = frame.loc[validation, "material"].astype(str).to_numpy()
    if shuffle_train_labels:
        rng = np.random.default_rng(shuffle_seed)
        y_train = rng.permutation(y_train)
    estimator.fit(x_train, y_train)
    predictions = np.asarray(estimator.predict(x_validation)).astype(str)
    recalls = pair_recalls(y_validation, predictions)
    hm_min = float(min(recalls.values()))
    decisions = frame.loc[validation, ["sample_id", "material", "split", "random_seed", "thickness_mm", "pose_index"]].copy()
    decisions["method"] = method_name
    decisions["prediction"] = predictions
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == predictions
    by_thickness = []
    for thickness, group in decisions.groupby("thickness_mm", sort=True):
        thickness_recalls = pair_recalls(group["material"].astype(str).to_numpy(), group["prediction"].astype(str).to_numpy())
        by_thickness.append(min(thickness_recalls.values()))
    return (
        {
            "method": method_name,
            "feature_count": int(len(feature_cols)),
            "validation_samples": int(len(y_validation)),
            "hematite_recall": recalls["Hematite"],
            "magnetite_recall": recalls["Magnetite"],
            "hm_min_recall": hm_min,
            "thickness_blind_hm_min_recall": hm_min,
            "worst_thickness_hm_min_recall": float(min(by_thickness)) if by_thickness else 0.0,
        },
        decisions,
    )


def observability_metrics(frame: pd.DataFrame, feature_cols: list[str], roc_auc_score) -> pd.DataFrame:
    validation = frame["split"].eq("validation")
    y = frame.loc[validation, "material"].astype(str).to_numpy()
    y_binary = (y == "Magnetite").astype(int)
    rows = []
    for col in feature_cols:
        values = frame.loc[validation, col].to_numpy(dtype=np.float64)
        h = values[y == "Hematite"]
        m = values[y == "Magnetite"]
        pooled = math.sqrt(0.5 * (float(np.var(h)) + float(np.var(m))) + EPS)
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


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    visible = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(format_markdown_value(row[col]) for col in columns) + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(output_dir: Path, gate: dict, model_selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top_feature_rows = metrics.head(8)
    best_rows = model_selection.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).head(4)
    lines = [
        "# v8A synthetic diffraction observability report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Development-only: `true`",
        "",
        "Shadow/final used: `false`",
        "",
        "Scope: synthetic powder-peak observability only; this is not a Geant4 transport result and not a hardware validation.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Best feature AUC: `{gate['best_feature_oriented_auc']:.4f}`",
        f"- Best feature d-prime: `{gate['best_feature_d_prime']:.4f}`",
        f"- Best peak-shape H/M min recall: `{gate['best_peak_shape_hm_min_recall']:.4f}`",
        f"- Worst-thickness H/M min recall: `{gate['best_peak_shape_worst_thickness_hm_min_recall']:.4f}`",
        f"- Control-only H/M min recall: `{gate['control_hm_min_recall']:.4f}`",
        f"- Shuffled-label H/M min recall: `{gate['shuffled_label_hm_min_recall']:.4f}`",
        f"- Overlap-only H/M min recall: `{gate['overlap_only_hm_min_recall']:.4f}`",
        "",
        "## Best models",
        "",
        markdown_table(
            best_rows,
            [
                "method",
                "hm_min_recall",
                "hematite_recall",
                "magnetite_recall",
                "worst_thickness_hm_min_recall",
            ],
        ),
        "",
        "## Top observability features",
        "",
        markdown_table(top_feature_rows, ["feature", "oriented_auc", "d_prime_abs"]),
        "",
        "## Interpretation",
        "",
        "A passing result only means that the tabulated powder-peak sidecar has enough synthetic H/M signal after the preregistered perturbations and sanity guards. It does not justify claims about ordinary XRT, standard Rayleigh scattering, or real detector throughput. The next step after a pass is a preregistered transport/detector integration plan, not a full material-sorting run.",
        "",
    ]
    (output_dir / "v8a_observability_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tiny v8A synthetic diffraction observability prototype.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_diffraction_sidecar_pilot")
    parser.add_argument("--random-seed", type=int, default=8801)
    parser.add_argument("--train-seeds", default="6101,6102,6103,6104,6105,6106,6107,6108,6109,6110,6111,6112")
    parser.add_argument("--validation-seeds", default="6201,6202,6203,6204,6205,6206")
    parser.add_argument("--thickness-list", default="3,5,8,10,15,20,30,40")
    parser.add_argument("--poses-per-condition", type=int, default=4)
    parser.add_argument("--two-theta-min", type=float, default=15.0)
    parser.add_argument("--two-theta-max", type=float, default=80.0)
    parser.add_argument("--bin-width-deg", type=float, default=0.10)
    parser.add_argument("--peak-width-deg", type=float, default=0.18)
    parser.add_argument("--angle-jitter-deg", type=float, default=0.055)
    parser.add_argument("--orientation-sigma", type=float, default=0.42)
    parser.add_argument("--background-level", type=float, default=0.070)
    parser.add_argument("--background-slope-sigma", type=float, default=0.18)
    parser.add_argument("--attenuation-strength", type=float, default=0.012)
    parser.add_argument("--counts-scale", type=float, default=2600.0)
    parser.add_argument("--read-noise-sigma", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty prototype output directory.")
    args = parser.parse_args()

    sk = require_sklearn()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    frame, manifest = build_dataset(args)
    metadata_cols = {
        "protocol",
        "development_only",
        "shadow_or_final_used",
        "split",
        "material",
        "random_seed",
        "thickness_mm",
        "pose_index",
        "sample_id",
    }
    control_cols = [
        "thickness_mm",
        "raw_total_intensity",
        "raw_mean_intensity",
        "raw_max_intensity",
        "estimated_background",
        "signal_scale_proxy",
    ]
    feature_cols = [col for col in frame.columns if col not in metadata_cols and col not in control_cols and col != "global_angle_shift_deg"]
    control_feature_cols = [col for col in control_cols if col in frame.columns]
    peak_shape_feature_cols = [col for col in feature_cols if not is_overlap_feature(col)]
    overlap_feature_cols = [col for col in feature_cols if is_coalesced_overlap_feature(col)]

    models = [
        (
            "ExtraTreesPeakShape",
            sk["ExtraTreesClassifier"](
                n_estimators=600,
                random_state=8802,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
            ),
            peak_shape_feature_cols,
        ),
        (
            "LogisticPeakShape",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=8803),
            ),
            peak_shape_feature_cols,
        ),
        (
            "ExtraTreesControlOnly",
            sk["ExtraTreesClassifier"](
                n_estimators=400,
                random_state=8804,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
            ),
            control_feature_cols,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](
                n_estimators=400,
                random_state=8805,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
            ),
            feature_cols,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](
                n_estimators=400,
                random_state=8806,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
            ),
            overlap_feature_cols,
        ),
    ]
    selection_rows = []
    decisions = []
    for method_name, estimator, cols in models:
        row, method_decisions = evaluate_model(
            frame,
            cols,
            method_name,
            estimator,
            shuffle_train_labels=method_name == "ExtraTreesShuffledTrainLabels",
        )
        selection_rows.append(row)
        decisions.append(method_decisions)
    model_selection = pd.DataFrame(selection_rows)
    validation_decisions = pd.concat(decisions, ignore_index=True)
    metrics = observability_metrics(frame, feature_cols, sk["roc_auc_score"])

    peak_models = model_selection[model_selection["method"].isin(["ExtraTreesPeakShape", "LogisticPeakShape"])]
    best_peak = peak_models.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0].to_dict()
    control_hm_min = float(model_selection.loc[model_selection["method"].eq("ExtraTreesControlOnly"), "hm_min_recall"].iloc[0])
    shuffled_hm_min = float(model_selection.loc[model_selection["method"].eq("ExtraTreesShuffledTrainLabels"), "hm_min_recall"].iloc[0])
    overlap_only_hm_min = float(model_selection.loc[model_selection["method"].eq("ExtraTreesOverlapOnly"), "hm_min_recall"].iloc[0])
    best_feature = metrics.iloc[0].to_dict()
    physical_observability_pass = bool(
        best_feature["oriented_auc"] >= PASS_FEATURE_AUC or best_feature["d_prime_abs"] >= PASS_FEATURE_D_PRIME
    )
    ml_pass = bool(
        best_peak["hm_min_recall"] >= PASS_HM_MIN_RECALL
        and best_peak["thickness_blind_hm_min_recall"] >= PASS_THICKNESS_BLIND_HM_MIN_RECALL
        and best_peak["worst_thickness_hm_min_recall"] >= PASS_THICKNESS_BLIND_HM_MIN_RECALL
    )
    control_guard_pass = bool(control_hm_min < MAX_CONTROL_HM_MIN_RECALL)
    shuffled_label_guard_pass = bool(shuffled_hm_min < MAX_SHUFFLED_LABEL_HM_MIN_RECALL)
    overlap_only_guard_pass = bool(overlap_only_hm_min < MAX_OVERLAP_ONLY_HM_MIN_RECALL)
    gate_passed = bool(physical_observability_pass and ml_pass and control_guard_pass and shuffled_label_guard_pass and overlap_only_guard_pass)
    if gate_passed:
        decision = "proceed_to_v8a_transport_preregistration"
    elif best_peak["hm_min_recall"] < NO_GO_HM_MIN_RECALL:
        decision = "no_go_refine_or_stop_diffraction_sidecar"
    else:
        decision = "gray_zone_strengthen_perturbations_before_transport"

    gate = {
        "generated_by": "analysis/v8a_diffraction_observability.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_diffraction_sidecar_pilot",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "input_data": "hardcoded_reference_powder_peak_table_only",
        "claim_scope": "synthetic_powder_peak_observability_only_not_geant4_not_hardware_validation",
        "gate_passed": gate_passed,
        "decision": decision,
        "physical_observability_pass": physical_observability_pass,
        "ml_pass": ml_pass,
        "control_guard_pass": control_guard_pass,
        "shuffled_label_guard_pass": shuffled_label_guard_pass,
        "overlap_only_guard_pass": overlap_only_guard_pass,
        "main_feature_set_excludes_overlap_windows": True,
        "best_feature": str(best_feature["feature"]),
        "best_feature_oriented_auc": float(best_feature["oriented_auc"]),
        "best_feature_d_prime": float(best_feature["d_prime_abs"]),
        "best_peak_shape_method": str(best_peak["method"]),
        "best_peak_shape_hm_min_recall": float(best_peak["hm_min_recall"]),
        "best_peak_shape_hematite_recall": float(best_peak["hematite_recall"]),
        "best_peak_shape_magnetite_recall": float(best_peak["magnetite_recall"]),
        "best_peak_shape_worst_thickness_hm_min_recall": float(best_peak["worst_thickness_hm_min_recall"]),
        "control_hm_min_recall": control_hm_min,
        "shuffled_label_hm_min_recall": shuffled_hm_min,
        "overlap_only_hm_min_recall": overlap_only_hm_min,
        "thresholds": {
            "pass_feature_auc": PASS_FEATURE_AUC,
            "pass_feature_d_prime": PASS_FEATURE_D_PRIME,
            "pass_hm_min_recall": PASS_HM_MIN_RECALL,
            "pass_thickness_blind_hm_min_recall": PASS_THICKNESS_BLIND_HM_MIN_RECALL,
            "max_control_hm_min_recall": MAX_CONTROL_HM_MIN_RECALL,
            "max_shuffled_label_hm_min_recall": MAX_SHUFFLED_LABEL_HM_MIN_RECALL,
            "max_overlap_only_hm_min_recall": MAX_OVERLAP_ONLY_HM_MIN_RECALL,
            "no_go_hm_min_recall": NO_GO_HM_MIN_RECALL,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest.update(
        {
            "generated_at_utc": gate["generated_at_utc"],
            "feature_count": int(len(feature_cols)),
            "peak_shape_feature_count": int(len(peak_shape_feature_cols)),
            "overlap_only_feature_count": int(len(overlap_feature_cols)),
            "main_feature_set_excludes_overlap_windows": True,
        }
    )

    frame.to_csv(output_dir / "v8a_synthetic_powder_features.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "v8a_observability_metrics.csv", index=False, lineterminator="\n")
    model_selection.to_csv(output_dir / "v8a_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_validation_decisions.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_synthetic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, model_selection, metrics)
    print(
        "decision={decision} gate_passed={passed} best_hm_min_recall={hm:.4f} best_auc={auc:.4f} control_hm={control:.4f}".format(
            decision=decision,
            passed=str(gate_passed).lower(),
            hm=gate["best_peak_shape_hm_min_recall"],
            auc=gate["best_feature_oriented_auc"],
            control=control_hm_min,
        )
    )


if __name__ == "__main__":
    main()
