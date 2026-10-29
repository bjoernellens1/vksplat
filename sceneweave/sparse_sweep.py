"""Reproducible first-pass Sceneweave P1 sweep, using the unchanged VkSplat trainer.

Invoke under sceneweave/run_amd.sh so child training subprocesses inherit RADV/DRM setup.
This experiment varies INITIALIZATION DEPTH VIEWS, not RGB supervision views.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from eval_psnr import evaluate
from check_replica_pair import check

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def launch(command,log=None):
    if log is None:
        subprocess.run(command,check=True,cwd=ROOT/"vksplat")
    else:
        with log.open("w") as fh:
            subprocess.run(command,check=True,cwd=ROOT/"vksplat",stdout=fh,stderr=subprocess.STDOUT)

def main():
    p=argparse.ArgumentParser(description="Measured vs oracle sparse-depth seed sweep")
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--mesh",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fx",type=float,required=True)
    p.add_argument("--fy",type=float,required=True)
    p.add_argument("--cx",type=float,required=True)
    p.add_argument("--cy",type=float,required=True)
    p.add_argument("--pose-camera",choices=("opencv","opengl"),required=True)
    p.add_argument("--depth-scale",type=float,required=True)
    p.add_argument("--frame-step",type=int,default=20)
    p.add_argument("--pixel-step",type=int,default=12)
    p.add_argument("--max-points",type=int,default=150000)
    p.add_argument("--seed-frame-steps",type=int,nargs="+",default=[1,4,16])
    p.add_argument("--train-steps",type=int,nargs="+",default=[250,500,1000,3000])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--prepare-only",action="store_true")
    a=p.parse_args()
    if any(x<1 for x in (*a.seed_frame_steps,*a.train_steps,a.repeats)):
        p.error("Steps and repeats must be positive")
    output=a.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    csvpath=output/"sweep.csv"
    for sparse in a.seed_frame_steps:
        folders={}
        for mode in ("measured","oracle"):
            dest=output/"data"/f"seedstep_{sparse:02d}"/mode
            folders[mode]=dest
            if not (dest/"sceneweave_conversion.json").exists():
                if dest.exists() and any(dest.iterdir()):
                    raise RuntimeError(f"Partial dataset exists; inspect or delete it: {dest}")
                cmd=[sys.executable,str(HERE/"replica_to_colmap.py"),
                    "--scene",str(a.scene),"--output",str(dest),"--mode",mode,
                    "--pose-camera",a.pose_camera,"--fx",str(a.fx),"--fy",str(a.fy),
                    "--cx",str(a.cx),"--cy",str(a.cy),"--depth-scale",str(a.depth_scale),
                    "--frame-step",str(a.frame_step),"--pixel-step",str(a.pixel_step),
                    "--max-points",str(a.max_points),"--seed-frame-step",str(sparse)]
                if mode=="oracle":cmd+=["--mesh",str(a.mesh)]
                print(f"Preparing {mode} seed-frame-step={sparse}",flush=True)
                launch(cmd)
        parity=check(folders["measured"],folders["oracle"])
        print("PARITY",sparse,json.dumps(parity),flush=True)
        if a.prepare_only:continue
        for steps in a.train_steps:
            for repeat in range(1,a.repeats+1):
                for mode in ("measured","oracle"):
                    dest=output/"runs"/f"seedstep_{sparse:02d}"/f"steps_{steps}"/f"repeat_{repeat:02d}"/mode
                    dest.mkdir(parents=True,exist_ok=True)
                    log=dest/"stdout.log"
                    if (dest/"result.json").exists():
                        result=json.loads((dest/"result.json").read_text())
                        rows.append(result)
                        continue
                    print(f"Train {mode} seedstep={sparse} steps={steps} repeat={repeat}",flush=True)
                    cmd=[sys.executable,str(HERE/"run_baseline.py"),
                         "--dataset",str(folders[mode]),"--image-dir","images",
                         "--steps",str(steps),"--output",str(dest)]
                    start=time.perf_counter()
                    launch(cmd,log)
                    wall=time.perf_counter()-start
                    run_dirs=[r.parent for r in dest.rglob("train.json") if r.parent!=dest]
                    if len(run_dirs)!=1:raise RuntimeError(f"Expected one run directory under {dest}; found {len(run_dirs)}")
                    run=run_dirs[0]
                    meta=json.loads((run/"train.json").read_text())
                    quality=evaluate(run)
                    (run/"eval_psnr.json").write_text(json.dumps(quality,indent=2)+"\n")
                    manifest=json.loads((folders[mode]/"sceneweave_conversion.json").read_text())
                    result=dict(mode=mode,seed_frame_step=sparse,seed_frame_count=len(manifest["seed_frame_indices"]),
                        seed_points=manifest["num_seed_points"],steps=steps,repeat=repeat,
                        mean_psnr_db=quality["mean_psnr_db"],median_psnr_db=quality["median_psnr_db"],
                        val_count=quality["count"],gpu_training_seconds=meta["time_elapsed"],
                        wall_seconds=wall,num_splats=meta["num_splats"],run_dir=str(run))
                    (dest/"result.json").write_text(json.dumps(result,indent=2)+"\n")
                    rows.append(result)
                    with csvpath.open("w",newline="") as fh:
                        writer=csv.DictWriter(fh,fieldnames=list(result))
                        writer.writeheader();writer.writerows(rows)
    if rows:
        with csvpath.open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        print("Results:",csvpath,flush=True)
    else:print("Preparation complete; no GPU training requested",flush=True)

if __name__=="__main__":main()
