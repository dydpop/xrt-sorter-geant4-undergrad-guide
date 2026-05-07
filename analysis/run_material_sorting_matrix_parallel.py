from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_material_sorting_matrix import (
    infer_macro_profile,
    load_completed_status,
    load_rows,
    load_status_rows,
    status_key,
    write_status_rows,
)


def run_one(project_root: Path, build_dir: Path, exe: Path, macro: Path, row: dict[str, str]) -> tuple[dict[str, str], str]:
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
    status_row = {
        "row_index": row.get("row_index", ""),
        "profile": row.get("profile", ""),
        "run_role": row.get("run_role", "material"),
        "material": row["material"],
        "source_id": row["source_id"],
        "thickness_mm": row["thickness_mm"],
        "random_seed": row["random_seed"],
        "returncode": str(proc.returncode),
        "elapsed_seconds": f"{elapsed:.3f}",
        "config_path": row["config_path"],
        "output_prefix": row["output_prefix"],
    }
    return status_row, proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated Geant4 material sorting configs with parent-owned status writes.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--macro-profile", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--role", choices=["all", "calibration", "material"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / args.profile / "material_sorting_matrix.csv"
    rows = []
    for matrix_index, row in enumerate(load_rows(matrix_path)):
        indexed = dict(row)
        indexed["row_index"] = str(matrix_index)
        indexed["profile"] = args.profile
        rows.append(indexed)
    if args.role != "all":
        rows = [row for row in rows if row.get("run_role", "material") == args.role]
    if args.start:
        rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    build_dir = project_root / "build"
    exe = build_dir / "xrt_sorter"
    macro_profile = args.macro_profile.strip() or infer_macro_profile(args.profile)
    macro = project_root / "analysis" / "configs" / f"run_material_sorting_{macro_profile}.mac"
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}. Run cmake --build build first.")
    if not macro.exists():
        raise FileNotFoundError(f"Missing macro: {macro}")

    status_dir = project_root / "results" / "material_sorting"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"run_status_{args.profile}.csv"
    existing_status_rows = load_status_rows(status_path)
    completed = set() if args.rerun_existing else load_completed_status(existing_status_rows)
    selected_keys = {status_key(row) for row in rows}
    failed = {status_key(row) for row in existing_status_rows if row.get("returncode", "") not in {"", "0"}}
    done = selected_keys & completed
    failed_selected = (selected_keys & failed) - done
    pending_rows = [row for row in rows if status_key(row) not in done and status_key(row) not in failed_selected]
    if args.status_only:
        pending = selected_keys - done - failed_selected
        print(f"profile={args.profile} role={args.role}")
        print(f"selected_rows={len(selected_keys)} completed={len(done)} failed={len(failed_selected)} pending={len(pending)}")
        return

    workers = max(1, int(args.workers))
    status_rows = list(existing_status_rows)
    print(
        f"profile={args.profile} selected={len(selected_keys)} already_completed={len(done)} "
        f"already_failed={len(failed_selected)} pending={len(pending_rows)} workers={workers}"
    )
    if not pending_rows:
        print("No rows selected.")
        return

    failures = []
    completed_now = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, project_root, build_dir, exe, macro, row): row for row in pending_rows}
        for future in as_completed(futures):
            row = futures[future]
            status_row, stdout = future.result()
            status_rows.append(status_row)
            write_status_rows(status_path, status_rows)
            completed_now += 1
            ok = status_row["returncode"] == "0"
            print(
                f"[{completed_now}/{len(pending_rows)}] row={status_row['row_index']} "
                f"{status_row['material']} seed={status_row['random_seed']} rc={status_row['returncode']}"
            )
            if not ok:
                failures.append(status_row)
                log_path = status_dir / f"failed_{args.profile}_{status_row['row_index']}.log"
                log_path.write_text(stdout, encoding="utf-8")
                if args.fail_fast:
                    raise RuntimeError(f"Run failed for {row['config_path']}; see {log_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} rows failed; see results/material_sorting/failed_{args.profile}_*.log")
    print(f"Wrote status to {status_path}")


if __name__ == "__main__":
    main()
