from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MATERIALS_FILE = Path("source_models/materials/material_catalog.csv")
OUTPUT_ROOT = Path("source_models/config/material_sorting_matrix")
THICKNESS_MM = [5.0, 10.0, 20.0]
SEEDS = [101, 202, 303]
SOURCES = [
    {"source_id": "mono_60kev", "source_mode": "mono", "mono_energy_keV": 60.0},
    {"source_id": "mono_100kev", "source_mode": "mono", "mono_energy_keV": 100.0},
    {
        "source_id": "spectrum_120kv",
        "source_mode": "spectrum",
        "mono_energy_keV": 80.0,
        "spectrum_file": "../../../spectra/w_target_120kV_1mmAl.csv",
    },
]
PROFILE_EVENTS = {"pilot": 2000, "full": 10000}


@dataclass(frozen=True)
class MatrixRun:
    profile: str
    material_name: str
    material_slug: str
    source_id: str
    source_mode: str
    mono_energy_keV: float
    thickness_mm: float
    random_seed: int
    config_path: Path
    output_prefix: str
    output_dir: str
    expected_events: int


def slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def load_materials(project_root: Path) -> pd.DataFrame:
    df = pd.read_csv(project_root / MATERIALS_FILE)
    enabled = df[df["enabled_for_undergrad"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if enabled.empty:
        raise ValueError("No enabled materials in material catalog.")
    return enabled


def render_config(run: MatrixRun, source: dict) -> str:
    spectrum_file = source.get("spectrum_file", "../../../spectra/w_target_120kV_1mmAl.csv")
    label = (
        f"material_sorting_{run.profile}_{run.material_slug}_{run.source_id}_"
        f"{int(run.thickness_mm)}mm_seed{run.random_seed}"
    )
    return f"""# Material sorting matrix config: {label}
run_id = {label}
experiment_label = {label}
output_prefix = {run.output_prefix}
output_dir = {run.output_dir}
benchmark_suite = material_sorting_v1
research_route = undergraduate_sorting_upgrade
prediction_stage = material_level
prep_profile = clean_dry_single_piece
feed_size_band = controlled_slab
feed_condition = simulated_single_material
sample_photons = 100
random_seed = {run.random_seed}

source_mode = {run.source_mode}
spectrum_file = {spectrum_file}
phase_space_file = ../../../phase_space/source_phase_space.csv
mono_energy_keV = {run.mono_energy_keV:.1f}
source_x_cm = -30.0
source_y_mm = 0.0
source_z_mm = 0.0
beam_half_y_mm = 5.0
beam_half_z_mm = 5.0
dir_x = 1.0
dir_y = 0.0
dir_z = 0.0

ore_material_mode = single
ore_primary_material = {run.material_name}
ore_secondary_material = Magnetite
ore_secondary_mass_fraction = 0.0

ore_shape = slab
ore_thickness_mm = {run.thickness_mm:.1f}
ore_half_y_mm = 100.0
ore_half_z_mm = 100.0

heterogeneity_mode = none
inclusion_material = Magnetite
inclusion_shape = ellipsoid
inclusion_thickness_mm = 4.0
inclusion_radius_y_mm = 20.0
inclusion_radius_z_mm = 20.0
inclusion_offset_y_mm = 0.0
inclusion_offset_z_mm = 0.0

detector_thickness_mm = 5.0
detector_half_y_mm = 100.0
detector_half_z_mm = 100.0
detector_x_cm = 25.0

world_x_cm = 100.0
world_y_cm = 50.0
world_z_cm = 50.0
envelope_x_cm = 80.0
envelope_y_cm = 30.0
envelope_z_cm = 30.0
"""


def build_matrix(project_root: Path, profile: str) -> list[MatrixRun]:
    materials = load_materials(project_root)
    expected_events = PROFILE_EVENTS[profile]
    out_dir = OUTPUT_ROOT / profile
    raw_output_dir = f"material_sorting_runs/{profile}"
    runs: list[MatrixRun] = []
    for row in materials.itertuples(index=False):
        material_slug = slugify(str(row.material_name))
        for thickness in THICKNESS_MM:
            for source in SOURCES:
                for seed in SEEDS:
                    output_prefix = (
                        f"ms_{profile}_{material_slug}_{source['source_id']}_"
                        f"{int(thickness)}mm_seed{seed}"
                    )
                    config_path = out_dir / f"{output_prefix}.txt"
                    runs.append(
                        MatrixRun(
                            profile=profile,
                            material_name=str(row.material_name),
                            material_slug=material_slug,
                            source_id=source["source_id"],
                            source_mode=source["source_mode"],
                            mono_energy_keV=float(source["mono_energy_keV"]),
                            thickness_mm=thickness,
                            random_seed=seed,
                            config_path=config_path,
                            output_prefix=output_prefix,
                            output_dir=raw_output_dir,
                            expected_events=expected_events,
                        )
                    )
    return runs


def write_matrix(project_root: Path, profile: str) -> Path:
    runs = build_matrix(project_root, profile)
    out_dir = project_root / OUTPUT_ROOT / profile
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    by_key = {(source["source_id"], source["source_mode"]): source for source in SOURCES}
    for run in runs:
        source = by_key[(run.source_id, run.source_mode)]
        with (project_root / run.config_path).open("w", encoding="utf-8", newline="\n") as f:
            f.write(render_config(run, source))
        rows.append(
            {
                "profile": run.profile,
                "material": run.material_name,
                "source_id": run.source_id,
                "source_mode": run.source_mode,
                "mono_energy_keV": run.mono_energy_keV,
                "thickness_mm": run.thickness_mm,
                "random_seed": run.random_seed,
                "expected_events": run.expected_events,
                "config_path": run.config_path.as_posix(),
                "output_prefix": run.output_prefix,
                "output_dir": run.output_dir,
            }
        )

    matrix_path = out_dir / "material_sorting_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate material-level sorting configs.")
    parser.add_argument("--profile", choices=sorted(PROFILE_EVENTS), default="pilot")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    matrix_path = write_matrix(project_root, args.profile)
    runs = list(csv.DictReader(matrix_path.open(encoding="utf-8")))
    print(f"Wrote {len(runs)} {args.profile} configs to {matrix_path}")
    print("Expected matrix: 10 materials x 3 thicknesses x 3 sources x 3 seeds = 270 runs")


if __name__ == "__main__":
    main()
