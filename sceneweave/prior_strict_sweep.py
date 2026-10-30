"""Sceneweave controlled VkSplat strict-growth baseline/oracle/capacity comparison.

This is a training-only ablation. Preprocessing, oracle extraction, and model
retrieval costs are NOT included, therefore no total prior speedup is claimed.
"""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
from eval_psnr import evaluate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONDITIONS=("baseline","oracle_completion","capacity_control","retrieved_prior","matched_capacity")

def command(dataset,output,*,steps=10000,seed=42,added_scale_m=.025):
    if steps<=0 or added_scale_m<=0:raise ValueError("Invalid training settings")
    return [sys.executable,str(HERE/"run_baseline.py"),
            "--dataset",str(dataset),"--output",str(output),
            "--image-dir","images","--mask-dir","masks",
            "--steps",str(steps),"--max-steps","30000",
            "--controlled-init","--added-scale-m",str(added_scale_m),
            "--shuffle-seed",str(seed),
            "--refine-start","500","--refine-stop","4000",
            "--refine-every","300","--grow-grad2d","0.0004"]

def run_one(condition,dataset,output,steps,seed,added_scale_m):
    if not (dataset/"masks").is_dir():
        raise ValueError(f"Masked ScanNet++ data required: {dataset/'masks'}")
    result_path=output/"result.json"
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result["condition"]!=condition or result["steps"]!=steps or result["seed"]!=seed:
            raise ValueError(f"Stale experiment cache: {result_path}")
        return result
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing partial nonempty experiment dir: {output}")
    output.mkdir(parents=True)
    cmd=command(dataset,output,steps=steps,seed=seed,added_scale_m=added_scale_m)
    begin=time.perf_counter()
    with (output/"run.log").open("w") as log:
        subprocess.run(cmd,cwd=ROOT/"vksplat",check=True,stdout=log,stderr=subprocess.STDOUT)
    process_s=time.perf_counter()-begin
    runs=list(output.glob("*/train.json"))
    if len(runs)!=1:raise RuntimeError("Expected one completed VkSplat training run")
    run_dir=runs[0].parent
    train=json.loads(runs[0].read_text())
    quality=evaluate(run_dir)
    (run_dir/"eval_psnr.json").write_text(json.dumps(quality,indent=2)+"\n")
    result=dict(condition=condition,steps=steps,seed=seed,
                dataset=str(dataset),run_dir=str(run_dir),
                train_s=train["time_elapsed"],process_s=process_s,
                psnr_db=quality["mean_psnr_db"],splats=train["num_splats"],
                peak_vram_bytes=train["peak_vram"],
                quality_scope="heldout_photometric_only",
                prior_acquisition_cost_included=False,
                controlled_reference_count=train.get("init_reference_count"),
                train_images=train.get("train_images"),
                val_images=train.get("val_images"))
    result_path.write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--paired-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--conditions",nargs="+",choices=CONDITIONS,
                   default=["baseline","oracle_completion","capacity_control"])
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--added-scale-m",type=float,default=.025)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in a.conditions:
        data=a.baseline if name=="baseline" else a.paired_root/name
        print("Running",name,data,flush=True)
        row=run_one(name,data.resolve(),a.output/name,a.steps,a.seed,a.added_scale_m)
        results.append(row)
        print(f"{name}: {row['psnr_db']:.3f} dB, {row['train_s']:.2f}s, "
              f"{row['splats']} splats",flush=True)
    images=[r["val_images"] for r in results]
    if len(results)>1 and any(images[0]!=i for i in images[1:]):
        raise ValueError("Validation frame metadata differ across paired runs")
    compact=[{k:v for k,v in r.items() if k not in ("train_images","val_images")}
             for r in results]
    with (a.output/"summary.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(compact[0]))
        writer.writeheader();writer.writerows(compact)
    (a.output/"comparison.json").write_text(json.dumps({
        "schema_version":"sceneweave-vksplat-prior-strict-v0",
        "results":compact,"matched_val_image_manifests":True,
        "controlled_initialization":True,"fixed_growth_policy":"strict",
        "speedup_scope":"training_only_not_end_to_end",
        "prior_data_acquisition_and_preparation_missing":True,
        "geometry_quality_measured":False
    },indent=2)+"\n")
    print("Saved",a.output/"summary.csv",flush=True)

if __name__=="__main__":main()
