"""Measured-only surface orientation ablation under identical VkSplat schedules.

All conditions use EXACT same dataset, COLMAP positions/colors, KNN-derived
tangent radius, RNG seed, growth policy and renderer. Modes change covariance
eigenvalues (isotropic vs anisotropic) and orientation (random vs aligned).
"""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from eval_psnr import evaluate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODES=("isotropic","random_anisotropic","aligned")

def make_command(dataset,out,*,mode,steps,seed,ratio):
    if mode not in MODES:raise ValueError("Invalid mode")
    if steps<=0 or not 0<ratio<=1:raise ValueError("Invalid budget or aspect ratio")
    return [
        sys.executable,str(HERE/"run_baseline.py"),
        "--dataset",str(dataset),"--output",str(out),
        "--image-dir","images","--mask-dir","masks",
        "--steps",str(steps),"--max-steps","30000",
        "--controlled-init","--shuffle-seed",str(seed),
        "--surface-mode",mode,"--normal-scale-ratio",str(ratio),
        "--refine-start","500","--refine-stop","4000",
        "--refine-every","300","--grow-grad2d","0.0004",
    ]

def run_condition(dataset,out,*,mode,steps,seed,ratio):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing nonempty output: {out}")
    out.mkdir(parents=True)
    cmd=make_command(dataset,out,mode=mode,steps=steps,seed=seed,ratio=ratio)
    t0=time.perf_counter()
    with (out/"run.log").open("w") as log:
        subprocess.run(cmd,cwd=ROOT/"vksplat",check=True,stdout=log,stderr=subprocess.STDOUT)
    process_s=time.perf_counter()-t0
    found=list(out.glob("*/train.json"))
    if len(found)!=1:raise ValueError("Expected exactly one completed run")
    path=found[0];run_dir=path.parent
    meta=json.loads(path.read_text())
    val=evaluate(run_dir)
    (run_dir/"eval_psnr.json").write_text(json.dumps(val,indent=2)+"\n")
    manifest=json.loads((out/"sceneweave_run_manifest.json").read_text())
    return {
        "mode":mode,"seed":seed,"steps":steps,"normal_scale_ratio":ratio,
        "train_s":meta["time_elapsed"],"process_s":process_s,
        "psnr_db":val["mean_psnr_db"],"splats":meta["num_splats"],
        "peak_vram_bytes":meta["peak_vram"],
        "run_dir":str(run_dir),
        "transform":meta["dataparser_transform"],
        "val_images":[Path(x["image_path"]).name for x in meta["val_images"]],
        "reference_seed_count":manifest["reference_seed_count"],
        "shuffle_seed":manifest["shuffle_seed"],
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--seeds",nargs="+",type=int,default=[41,42,43])
    p.add_argument("--normal-scale-ratio",type=float,default=.25)
    p.add_argument("--modes",choices=MODES,nargs="+",default=list(MODES))
    args=p.parse_args()
    dataset=args.dataset.resolve();out=args.output.resolve()
    manifest_file=dataset/"sceneweave_conversion.json"
    if not manifest_file.is_file():p.error("Missing frozen Sceneweave conversion manifest")
    manifest=json.loads(manifest_file.read_text())
    if manifest.get("mode")!="measured":p.error("Only measured-only baseline permitted")
    if not (dataset/"masks").is_dir():p.error("Missing ScanNet++ mask directory")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        # Rotate order across seeds to reduce systematic thermal/cache bias.
        modes=list(args.modes)
        if modes:
            k=seed%len(modes)
            modes=modes[k:]+modes[:k]
        for mode in modes:
            directory=out/f"seed_{seed}"/mode
            print(f"Running seed={seed} mode={mode}",flush=True)
            row=run_condition(dataset,directory,mode=mode,steps=args.steps,
                              seed=seed,ratio=args.normal_scale_ratio)
            rows.append(row)
            print(f"{mode}: PSNR={row['psnr_db']:.3f}dB, "
                  f"train={row['train_s']:.2f}s, splats={row['splats']}",flush=True)
    by_seed={}
    for row in rows:by_seed.setdefault(row["seed"],[]).append(row)
    for seed,rr in by_seed.items():
        a=rr[0]
        for b in rr[1:]:
            if not np.allclose(a["transform"],b["transform"],rtol=0,atol=2e-6):
                raise ValueError(f"Normalization mismatch at seed {seed}")
            if a["val_images"]!=b["val_images"]:
                raise ValueError(f"Validation split mismatch at seed {seed}")
            if a["reference_seed_count"]!=b["reference_seed_count"]:
                raise ValueError(f"Reference count mismatch at seed {seed}")
    compact=[{k:v for k,v in row.items() if k not in ("transform","val_images")}
             for row in rows]
    with (out/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]))
        writer.writeheader();writer.writerows(compact)
    (out/"comparison.json").write_text(json.dumps({
        "schema_version":"sceneweave-measured-surface-init-v0",
        "dataset":str(dataset),"conditions":list(args.modes),
        "matched_dataset_transform_and_holdout":True,
        "fixed_growth_policy":"strict",
        "comparison":"initial_covariance_only",
        "full_mapping_quality":"photometric_only",
        "note":"Measures end-to-end process wall and GPU training but no additional GT metric depth",
        "rows":compact},indent=2)+"\n")
    print("Saved",out/"summary.csv",flush=True)

if __name__=="__main__":main()
