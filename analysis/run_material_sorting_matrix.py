from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated Geant4 material sorting configs.")
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of matrix rows to run.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset for resumable batches.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / args.profile / "material_sorting_matrix.csv"
    rows = load_rows(matrix_path)
    if args.start:
        rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    build_dir = project_root / "build"
    exe = build_dir / "xrt_sorter"
    macro = project_root / "analysis" / "configs" / f"run_material_sorting_{args.profile}.mac"
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}. Run cmake --build build first.")
    if not macro.exists():
        raise FileNotFoundError(f"Missing macro: {macro}")

    status_dir = project_root / "results" / "material_sorting"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"run_status_{args.profile}.csv"
    status_rows = []

    for index, row in enumerate(rows, start=args.start):
        config_path = project_root / row["config_path"]
        env = os.environ.copy()
        env["XRT_EXPERIMENT_CONFIG"] = str(config_path)
        started = time.time()
        proc = subprocess.run(
            [str(exe), str(macro)],
            cwd=build_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.time() - started
        status_rows.append(
            {
                "row_index": index,
                "profile": args.profile,
                "material": row["material"],
                "source_id": row["source_id"],
                "thickness_mm": row["thickness_mm"],
                "random_seed": row["random_seed"],
                "returncode": proc.returncode,
                "elapsed_seconds": f"{elapsed:.3f}",
                "config_path": row["config_path"],
                "output_prefix": row["output_prefix"],
            }
        )
        print(
            f"[{index}] {row['material']} {row['source_id']} "
            f"{row['thickness_mm']}mm seed={row['random_seed']} rc={proc.returncode}"
        )
        if proc.returncode != 0:
            log_path = status_dir / f"failed_{args.profile}_{index}.log"
            log_path.write_text(proc.stdout, encoding="utf-8")
            raise RuntimeError(f"Run failed for {row['config_path']}; see {log_path}")

    if status_rows:
        with status_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(status_rows[0].keys()), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(status_rows)
        print(f"Wrote status to {status_path}")
    else:
        print("No rows selected.")


if __name__ == "__main__":
    main()
