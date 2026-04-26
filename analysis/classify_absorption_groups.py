from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

PHOTONS_PER_SAMPLE = 100

EVENT_FILES = {
    "Quartz": "xrt_real_source_quartz_events.csv",
    "Orthoclase": "xrt_real_source_orthoclase_events.csv",
    "Calcite": "xrt_real_source_calcite_events.csv",
    "Pyrite": "xrt_real_source_pyrite_events.csv",
    "Hematite": "xrt_real_source_hematite_events.csv",
    "Magnetite": "xrt_real_source_magnetite_events.csv",
}

GROUP_MAP = {
    "Quartz": "low_absorption",
    "Orthoclase": "low_absorption",
    "Calcite": "low_absorption",
    "Pyrite": "high_absorption",
    "Hematite": "high_absorption",
    "Magnetite": "high_absorption",
}


def build_virtual_samples(df: pd.DataFrame, material_name: str, photons_per_sample: int) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    n_complete_samples = len(df) // photons_per_sample
    df = df.iloc[: n_complete_samples * photons_per_sample].copy()

    df["sample_id"] = df["event_id"] // photons_per_sample

    grouped = (
        df.groupby("sample_id")
        .agg(
            n_events=("event_id", "count"),
            detector_edep_keV_sum=("detector_edep_keV", "sum"),
            detector_gamma_entries_sum=("detector_gamma_entries", "sum"),
            primary_gamma_entries_sum=("primary_gamma_entries", "sum"),
        )
        .reset_index()
    )

    grouped["material"] = material_name
    grouped["group_label"] = GROUP_MAP[material_name]
    grouped["mean_detector_edep_keV"] = grouped["detector_edep_keV_sum"] / grouped["n_events"]
    grouped["detector_gamma_rate"] = grouped["detector_gamma_entries_sum"] / grouped["n_events"]
    grouped["primary_transmission_rate"] = grouped["primary_gamma_entries_sum"] / grouped["n_events"]

    return grouped


def threshold_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    low_mean = train_df[train_df["group_label"] == "low_absorption"]["primary_transmission_rate"].mean()
    high_mean = train_df[train_df["group_label"] == "high_absorption"]["primary_transmission_rate"].mean()
    threshold = 0.5 * (low_mean + high_mean)

    preds = test_df["primary_transmission_rate"].apply(
        lambda x: "low_absorption" if x > threshold else "high_absorption"
    )

    acc = accuracy_score(test_df["group_label"], preds)
    cm = confusion_matrix(
        test_df["group_label"], preds,
        labels=["low_absorption", "high_absorption"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["low_absorption", "high_absorption"],
        columns=["low_absorption", "high_absorption"]
    )

    return acc, threshold, cm_df


def logistic_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["group_label"]
    y_test = test_df["group_label"]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(
        y_test, y_pred,
        labels=["low_absorption", "high_absorption"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["low_absorption", "high_absorption"],
        columns=["low_absorption", "high_absorption"]
    )

    return acc, cm_df


def main():
    project_root = Path(__file__).resolve().parents[1]
    build_dir = project_root / "build"
    results_dir = project_root / "analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for material, filename in EVENT_FILES.items():
        file_path = build_dir / filename
        df = pd.read_csv(file_path)
        samples = build_virtual_samples(df, material, PHOTONS_PER_SAMPLE)
        all_samples.append(samples)

    samples_df = pd.concat(all_samples, ignore_index=True)
    samples_df.to_csv(results_dir / "absorption_group_virtual_samples.csv", index=False)

    print("\n================ Absorption Group Sample Overview ================\n")
    print(samples_df.groupby("group_label")["primary_transmission_rate"].agg(["count", "mean", "std", "min", "max"]))
    print("\n=================================================================\n")

    # 训练 / 测试切分：每种材料前25个样本训练，后25个测试
    train_parts = []
    test_parts = []
    for material in EVENT_FILES.keys():
        df_m = samples_df[samples_df["material"] == material].sort_values("sample_id")
        train_parts.append(df_m.iloc[:25])
        test_parts.append(df_m.iloc[25:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # 方法 A：阈值法（只看 transmission）
    acc_a, threshold, cm_a = threshold_classifier(train_df, test_df)

    # 方法 B：Logistic，单特征
    acc_b, cm_b = logistic_classifier(train_df, test_df, ["primary_transmission_rate"])

    # 方法 C：Logistic，多特征
    acc_c, cm_c = logistic_classifier(
        train_df, test_df,
        ["primary_transmission_rate", "mean_detector_edep_keV", "detector_gamma_rate"]
    )

    print("\n================ Coarse Group Classification Results ================\n")

    print("--- A_threshold_transmission_only ---")
    print(f"threshold = {threshold:.6f}")
    print(f"accuracy  = {acc_a:.4f}")
    print(cm_a)
    print()

    print("--- B_logistic_transmission_only ---")
    print(f"accuracy  = {acc_b:.4f}")
    print(cm_b)
    print()

    print("--- C_logistic_transmission_edep_gamma ---")
    print(f"accuracy  = {acc_c:.4f}")
    print(cm_c)
    print()

    print("=====================================================================\n")

    summary_df = pd.DataFrame([
        {"method": "A_threshold_transmission_only", "n_features": 1, "accuracy": acc_a},
        {"method": "B_logistic_transmission_only", "n_features": 1, "accuracy": acc_b},
        {"method": "C_logistic_transmission_edep_gamma", "n_features": 3, "accuracy": acc_c},
    ])
    summary_df.to_csv(results_dir / "absorption_group_classification_summary.csv", index=False)

    cm_a.to_csv(results_dir / "absorption_group_confusion_threshold.csv")
    cm_b.to_csv(results_dir / "absorption_group_confusion_logistic_1f.csv")
    cm_c.to_csv(results_dir / "absorption_group_confusion_logistic_3f.csv")

    print(f"已保存样本表: {results_dir / 'absorption_group_virtual_samples.csv'}")
    print(f"已保存分类汇总: {results_dir / 'absorption_group_classification_summary.csv'}")


if __name__ == "__main__":
    main()