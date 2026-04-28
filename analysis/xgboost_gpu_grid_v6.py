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

import material_sorting_selected_rebuild as selected
import material_sorting_v2 as v2
import strict_generalization_audit as strict


GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.015, 0.03, 0.06],
    "subsample": [0.85, 1.0],
    "colsample_bytree": [0.75, 0.90],
    "reg_lambda": [1.0, 3.0],
}
N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE = 20260429


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def split_frame(frame: pd.DataFrame, train_seeds: list[int], inner_seeds: list[int], validation_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    inner = frame[seed_series.isin(inner_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    if train.empty or inner.empty or validation.empty:
        raise ValueError("Train, inner validation, and external validation splits must all be non-empty.")
    return train, inner, validation


def augment_splits(train: pd.DataFrame, inner: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    base_cols = v2.numeric_feature_columns(pd.concat([train, inner, validation], ignore_index=True))
    train_aug, inner_aug, feature_cols, dictionary = selected.append_dictionary(train, inner, base_cols)
    validation_aug = v2.append_dictionary_features(validation, dictionary)
    return train_aug, inner_aug, validation_aug, feature_cols


def grid_rows() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*(GRID[key] for key in keys))]


def fit_candidate(params: dict, train: pd.DataFrame, inner: pd.DataFrame, feature_cols: list[str]):
    try:
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing XGBoost dependency: {exc}") from exc

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["material"].astype(str))
    y_inner = encoder.transform(inner["material"].astype(str))
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    model.fit(
        train[feature_cols],
        y_train,
        eval_set=[(inner[feature_cols], y_inner)],
        verbose=False,
    )
    return model, encoder


def model_scores(model, encoder, frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = model.predict_proba(frame[feature_cols])
    classes = np.array(encoder.classes_)
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions, scores, classes


def best_iteration(model) -> int:
    value = getattr(model, "best_iteration", None)
    if value is None:
        return N_ESTIMATORS
    return int(value) + 1


def best_inner_logloss(model) -> float:
    try:
        result = model.evals_result()
        values = result["validation_0"]["mlogloss"]
        return float(values[min(len(values), best_iteration(model)) - 1])
    except Exception:
        return math.nan


def candidate_complexity(params: dict, trees: int) -> int:
    return int(params["max_depth"]) * int(trees)


def evaluate_candidates(train: pd.DataFrame, inner: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str], sk) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    fitted = []
    for index, params in enumerate(grid_rows()):
        try:
            model, encoder = fit_candidate(params, train, inner, feature_cols)
            predictions, scores, classes = model_scores(model, encoder, validation, feature_cols)
            metrics = strict.add_hm_metrics(
                strict.evaluate_context_scores("XGBoostGPUGridV6", validation, predictions, scores, classes, sk),
                validation,
                predictions,
                sk,
            )
            trees = best_iteration(model)
            row = {
                "candidate_id": index,
                **params,
                "n_estimators": N_ESTIMATORS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "best_iteration_trees": trees,
                "best_inner_mlogloss": best_inner_logloss(model),
                "complexity_score": candidate_complexity(params, trees),
                **metrics,
                "error": "",
            }
            fitted.append((index, model, encoder, predictions, scores, classes))
        except Exception as exc:  # noqa: BLE001
            row = {
                "candidate_id": index,
                **params,
                "n_estimators": N_ESTIMATORS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "best_iteration_trees": math.nan,
                "best_inner_mlogloss": math.nan,
                "complexity_score": math.nan,
                "method": "XGBoostGPUGridV6",
                "samples": int(len(validation)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "min_class_recall": math.nan,
                "hematite_recall": math.nan,
                "magnetite_recall": math.nan,
                "hm_min_recall": math.nan,
                "error": str(exc),
            }
        rows.append(row)
    table = pd.DataFrame(rows)
    ranked = table.dropna(subset=["hm_min_recall", "min_class_recall", "macro_f1", "top1_accuracy", "complexity_score"]).sort_values(
        ["hm_min_recall", "min_class_recall", "macro_f1", "top1_accuracy", "complexity_score"],
        ascending=[False, False, False, False, True],
    )
    if ranked.empty:
        raise RuntimeError("No XGBoost GPU grid candidate produced finite validation metrics.")
    selected_row = ranked.iloc[0].to_dict()
    selected_id = int(selected_row["candidate_id"])
    selected_fit = next(item for item in fitted if item[0] == selected_id)
    table["selected"] = table["candidate_id"].astype(int).eq(selected_id)
    _, _, _, predictions, scores, classes = selected_fit
    return table, selected_row, predictions, scores, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered XGBoost GPU grid for Accuracy Sprint v6.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v5_hm_lowwide")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v6_gpu_search")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--train-seeds", default="1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516")
    parser.add_argument("--inner-validation-seeds", default="1517,1518,1519,1520")
    parser.add_argument("--validation-seeds", default="1601,1602,1603,1604,1605,1606,1607,1608,1609,1610")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = parse_int_list(args.train_seeds)
    inner_seeds = parse_int_list(args.inner_validation_seeds)
    validation_seeds = parse_int_list(args.validation_seeds)
    sk = v2.require_sklearn()
    group_map = selected.material_group_map(project_root)
    raw_dirs = strict.parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)
    frame, status = strict.build_frame_from_raw_dirs(project_root, raw_dirs, args.photon_budget)
    train, inner, validation = split_frame(frame, train_seeds, inner_seeds, validation_seeds)
    train_aug, inner_aug, validation_aug, feature_cols = augment_splits(train, inner, validation)

    table, selected_row, predictions, scores, classes = evaluate_candidates(train_aug, inner_aug, validation_aug, feature_cols, sk)
    per_class = strict.per_class_recall_context(validation_aug, predictions, "validation", sk)
    decisions = v2.decision_frame(validation_aug, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    failure = strict.failure_analysis_frame(per_class, decisions, group_map)
    pairwise = strict.hm_pairwise_audit_frame(train_aug, validation_aug, feature_cols, sk, "validation")

    write_csv(table, output_dir / "gpu_grid_search_candidates.csv")
    write_csv(pd.DataFrame([selected_row]), output_dir / "gpu_grid_selected_summary.csv")
    write_csv(per_class, output_dir / "per_class_recall_validation.csv")
    write_csv(decisions, output_dir / "validation_decisions.csv")
    write_csv(failure, output_dir / "failure_analysis.csv")
    write_csv(pairwise, output_dir / "hm_pairwise_audit.csv")

    manifest = {
        "generated_by": "analysis/xgboost_gpu_grid_v6.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "accuracy_v6_pre_registered_xgboost_gpu_grid",
        "raw_dir": args.raw_dir,
        "raw_dirs": [path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.as_posix() for path in raw_dirs],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "train_seeds": train_seeds,
        "inner_validation_seeds": inner_seeds,
        "external_validation_seeds": validation_seeds,
        "grid": GRID,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_STATE,
        "device": "cuda",
        "selection_policy": "rank by hm_min_recall, then min_class_recall, macro_f1, top1_accuracy, then smaller max_depth*best_iteration",
        "selected_candidate": selected_row,
        "status": status,
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    v2.write_manifest(output_dir / "gpu_grid_manifest.json", manifest)
    print(f"Wrote v6 XGBoost GPU grid search to {output_dir}")
    print(f"selected_candidate_id={int(selected_row['candidate_id'])} hm_min_recall={selected_row['hm_min_recall']}")


if __name__ == "__main__":
    main()
