"""Paired VkSplat work-saving sweep: baseline vs. plane-anchored + geometry-freeze.

Compares standard measured-only training against the Sceneweave P5 work-saving
dataset (which projects within-tolerance seeds onto consensus planes and freezes
3D geometry/densification after `recommended_freeze_after_ratio * steps`,
accounting for all upstream prediction and plane-matching overhead).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from eval_psnr import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDITIONS = ("baseline", "work_saving_freeze", "unanchored_freeze_control")


def make_command(
    dataset: Path,
    out: Path,
    *,
    steps: int,
    seed: int,
    freeze_after_iter: int = 0,
    surface_mode: str = "isotropic",
) -> list[str]:
    if steps <= 0 or freeze_after_iter < 0:
        raise ValueError("Invalid steps or freeze_after_iter")
    cmd = [
        sys.executable,
        str(HERE / "run_baseline.py"),
        "--dataset",
        str(dataset),
        "--output",
        str(out),
        "--image-dir",
        "images",
        "--steps",
        str(steps),
        "--max-steps",
        "30000",
        "--controlled-init",
        "--shuffle-seed",
        str(seed),
        "--surface-mode",
        surface_mode,
        "--refine-start",
        "500",
        "--refine-stop",
        "4000",
        "--refine-every",
        "300",
        "--grow-grad2d",
        "0.0004",
    ]
    if (dataset / "masks").is_dir():
        cmd.extend(["--mask-dir", "masks"])
    if freeze_after_iter > 0:
        cmd.extend(["--freeze-geometry-after", str(freeze_after_iter)])
    return cmd


def resolve_freeze_step(
    work_saving_dataset: Path,
    steps: int,
    override_ratio: float | None = None,
    treatment_steps_override: int | None = None,
) -> tuple[int, float, int]:
    ws_file = work_saving_dataset / "sceneweave_work_saving.json"
    if not ws_file.is_file():
        raise ValueError(f"Missing sceneweave_work_saving.json in {work_saving_dataset}")
    ws = json.loads(ws_file.read_text(encoding="utf-8"))
    if ws.get("schema_version") != "sceneweave-vksplat-work-saving-v1":
        raise ValueError("Unsupported work-saving manifest schema")
    ratio = float(override_ratio if override_ratio is not None else ws["recommended_freeze_after_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("freeze ratio must be in (0, 1]")
    overhead_s = float(ws.get("total_prior_overhead_s", 0.0))
    treatment_steps = (
        int(treatment_steps_override)
        if treatment_steps_override is not None
        else max(1, int(round(steps * min(1.0, ratio * 1.15))))
    )
    freeze_step = int(round(steps * ratio)) if ratio < 1.0 else 0
    return freeze_step, overhead_s, treatment_steps


def run_condition(
    dataset: Path,
    out: Path,
    *,
    condition: str,
    steps: int,
    seed: int,
    freeze_after_iter: int,
    prior_overhead_s: float,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing nonempty output: {out}")
    out.mkdir(parents=True)
    cmd = make_command(
        dataset,
        out,
        steps=steps,
        seed=seed,
        freeze_after_iter=freeze_after_iter,
        surface_mode="isotropic",
    )
    t0 = time.perf_counter()
    with (out / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=ROOT / "vksplat", check=True, stdout=log, stderr=subprocess.STDOUT)
    process_s = time.perf_counter() - t0
    found = list(out.glob("*/train.json"))
    if len(found) != 1:
        raise ValueError("Expected exactly one completed run")
    run_dir = found[0].parent
    meta = json.loads(found[0].read_text())
    val = evaluate(run_dir)
    (run_dir / "eval_psnr.json").write_text(json.dumps(val, indent=2) + "\n")
    total_end_to_end_s = float(meta["time_elapsed"]) + float(prior_overhead_s)
    return {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "freeze_geometry_after_iter": freeze_after_iter,
        "prior_overhead_s": prior_overhead_s,
        "train_s": float(meta["time_elapsed"]),
        "end_to_end_s": total_end_to_end_s,
        "process_s": process_s + float(prior_overhead_s),
        "psnr_db": float(val["mean_psnr_db"]),
        "splats": int(meta["num_splats"]),
        "peak_vram_bytes": int(meta["peak_vram"]),
        "run_dir": str(run_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--work-saving-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--treatment-steps", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--freeze-ratio", type=float, default=None)
    args = p.parse_args()

    base = args.baseline.resolve()
    ws_dir = args.work_saving_dataset.resolve()
    out = args.output.resolve()
    freeze_step, overhead_s, treat_steps = resolve_freeze_step(
        ws_dir, args.steps, args.freeze_ratio, args.treatment_steps
    )
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for cond in CONDITIONS:
            dset = ws_dir if cond == "work_saving_freeze" else base
            f_iter = freeze_step if cond != "baseline" else 0
            c_steps = args.steps if cond == "baseline" else treat_steps
            ovh = overhead_s if cond == "work_saving_freeze" else 0.0
            run_out = out / f"seed_{seed}" / cond
            row = run_condition(
                dset,
                run_out,
                condition=cond,
                steps=c_steps,
                seed=seed,
                freeze_after_iter=f_iter,
                prior_overhead_s=ovh,
            )
            rows.append(row)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": "sceneweave-work-saving-sweep-v0",
                "baseline": str(base),
                "work_saving_dataset": str(ws_dir),
                "baseline_steps": args.steps,
                "treatment_steps": treat_steps,
                "freeze_geometry_after_iter": freeze_step,
                "prior_overhead_s": overhead_s,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
