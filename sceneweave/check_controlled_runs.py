"""Compare B0/B2/B3 controlled runs and audit identical normalization and held-out views.

Expected layout: ROOT/steps_N/{baseline,oracle_completion,capacity_control}/RUN/train.json.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from eval_psnr import evaluate


CONDITIONS=("baseline","oracle_completion","capacity_control")


def load_condition(root: Path, steps: int, name: str):
    directory=root/f"steps_{steps}"/name
    runs=list(directory.rglob("train.json"))
    if len(runs)!=1:
        raise ValueError(f"Expected exactly one run in {directory}; found {len(runs)}")
    run=runs[0].parent
    meta=json.loads((run/"train.json").read_text())
    settings=json.loads((directory/"sceneweave_run_manifest.json").read_text())
    if not settings["controlled_init"]:
        raise ValueError(f"Not a controlled-init run: {directory}")
    eval_file=run/"eval_psnr.json"
    quality=json.loads(eval_file.read_text()) if eval_file.exists() else evaluate(run)
    if not eval_file.exists():eval_file.write_text(json.dumps(quality,indent=2)+"\n")
    matrix=np.asarray(meta["dataparser_transform"],dtype=float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid normalization transform: {run}")
    val_names=[Path(v["image_path"]).name for v in meta["val_images"]]
    return {
        "matrix": matrix, "val_names": val_names,
        "data": dict(steps=steps,condition=name,psnr_db=quality["mean_psnr_db"],
                     median_psnr_db=quality["median_psnr_db"],
                     gpu_training_s=meta["time_elapsed"],num_splats=meta["num_splats"],
                     reference_count=settings["reference_seed_count"],
                     added_scale_m=settings["added_scale_m"],shuffle_seed=settings["shuffle_seed"],
                     run_dir=str(run))
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--steps",type=int,nargs="+",default=[250,1000])
    args=parser.parse_args()
    rows=[]
    for steps in args.steps:
        records=[load_condition(args.root,steps,k) for k in CONDITIONS]
        first=records[0]
        for rec in records[1:]:
            if not np.allclose(first["matrix"],rec["matrix"],rtol=0,atol=2e-6):
                raise AssertionError(f"World normalization differs between {first['data']['condition']} and {rec['data']['condition']}")
            if first["val_names"]!=rec["val_names"]:
                raise AssertionError("Held-out image lists differ")
            if first["data"]["shuffle_seed"]!=rec["data"]["shuffle_seed"]:
                raise AssertionError("Shuffle seeds differ")
            if first["data"]["reference_count"]!=rec["data"]["reference_count"]:
                raise AssertionError("Original seed prefix count differs")
            if first["data"]["added_scale_m"]!=rec["data"]["added_scale_m"]:
                raise AssertionError("Added-point initialization scale differs")
        rows += [r["data"] for r in records]
        vals={x["condition"]:x["psnr_db"] for x in [r["data"] for r in records]}
        print(f"{steps} steps: B0={vals['baseline']:.3f} dB, "
              f"B2={vals['oracle_completion']:.3f} dB, "
              f"B3={vals['capacity_control']:.3f} dB; "
              f"oracle-control={vals['oracle_completion']-vals['capacity_control']:+.3f} dB")
        print("Shared normalization, held-out images and shuffle seed: YES")
    destination=args.root/"controlled_summary.csv"
    with destination.open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:",destination)


if __name__=="__main__":main()
