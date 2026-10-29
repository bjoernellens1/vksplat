"""P1d reproducible multi-seed fixed-budget screening; not checkpoint-based training.

Run through sceneweave/run_amd.sh; command resumes completed runs.
Each run trains from scratch and records *endpoint* time/quality.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean, stdev

from eval_psnr import evaluate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONDITIONS=("baseline","oracle_completion","capacity_control")

def datasets(base,completion):
    return {"baseline":base, "oracle_completion":completion/"oracle_completion",
            "capacity_control":completion/"capacity_control"}

def run_one(dataset,folder,steps,seed,scale):
    folder.mkdir(parents=True,exist_ok=True)
    result_file=folder/"result.json"
    if result_file.exists():
        result=json.loads(result_file.read_text())
        if (result["steps"],result["seed"],result["added_scale_m"])!=(steps,seed,scale):
            raise ValueError(f"Cached run parameters differ: {result_file}")
        return result
    existing=list(folder.rglob("train.json"))
    if existing:
        raise RuntimeError(f"Incomplete previous run at {folder}. Inspect before retrying.")
    cmd=[sys.executable,str(HERE/"run_baseline.py"),
         "--dataset",str(dataset),"--image-dir","images","--output",str(folder),
         "--steps",str(steps),"--controlled-init","--shuffle-seed",str(seed),
         "--added-scale-m",str(scale)]
    with (folder/"training.log").open("w") as log:
        subprocess.run(cmd,cwd=ROOT/"vksplat",check=True,stdout=log,stderr=subprocess.STDOUT)
    runs=[p.parent for p in folder.rglob("train.json")]
    if len(runs)!=1:
        raise RuntimeError(f"Expected one VkSplat run, found {len(runs)}: {folder}")
    r=runs[0]
    meta=json.loads((r/"train.json").read_text())
    quality=evaluate(r)
    (r/"eval_psnr.json").write_text(json.dumps(quality,indent=2)+"\n")
    manifest=json.loads((folder/"sceneweave_run_manifest.json").read_text())
    val=[Path(v["image_path"]).name for v in meta["val_images"]]
    result={"steps":steps,"seed":seed,"added_scale_m":scale,
            "psnr_db":quality["mean_psnr_db"],"median_psnr_db":quality["median_psnr_db"],
            "gpu_training_seconds":meta["time_elapsed"],"num_splats":meta["num_splats"],
            "reference_count":manifest["reference_seed_count"],
            "normalization":meta["dataparser_transform"],"val_images":val,
            "run_dir":str(r)}
    result_file.write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--completion",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--steps",type=int,nargs="+",default=[100,250,500,1000])
    ap.add_argument("--seeds",type=int,nargs="+",default=[41,42,43])
    ap.add_argument("--added-scale-m",type=float,default=0.025)
    a=ap.parse_args()
    if any(s<=0 for s in a.steps) or not a.seeds or not 0<a.added_scale_m<10:
        ap.error("Positive budgets and scale required")
    data=datasets(a.baseline.resolve(),a.completion.resolve())
    out=a.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for steps in a.steps:
        for seed in a.seeds:
            group={}
            for condition,dataset in data.items():
                folder=out/f"steps_{steps}"/f"seed_{seed}"/condition
                print(f"{steps} steps; seed {seed}; {condition}",flush=True)
                group[condition]=run_one(dataset,folder,steps,seed,a.added_scale_m)
            reference=group["baseline"]
            for condition,r in group.items():
                if r["reference_count"]!=reference["reference_count"]:
                    raise AssertionError("Reference count differs")
                import numpy as np
                if not np.allclose(r["normalization"],reference["normalization"],atol=2e-6,rtol=0):
                    raise AssertionError("World normalization differs")
                if r["val_images"]!=reference["val_images"]:
                    raise AssertionError("Validation images differ")
                records.append({k:v for k,v in r.items() if k not in ("normalization","val_images")}
                               | {"condition":condition})
            with (out/"results.csv").open("w",newline="") as fh:
                writer=csv.DictWriter(fh,fieldnames=list(records[0]))
                writer.writeheader();writer.writerows(records)
            print(f"delta oracle-control: {group['oracle_completion']['psnr_db'] - group['capacity_control']['psnr_db']:+.3f} dB",flush=True)
    print("Results:",out/"results.csv")
    for budget in a.steps:
        for condition in CONDITIONS:
            vals=[r["psnr_db"] for r in records if r["steps"]==budget and r["condition"]==condition]
            if vals:
                print(f"{budget:4d} {condition:20s}: {mean(vals):.3f} dB "
                      f"(std {stdev(vals):.3f} dB)" if len(vals)>1
                      else f"{budget:4d} {condition:20s}: {mean(vals):.3f} dB")

if __name__=="__main__":
    main()
