"""Run a compact bounded-growth VkSplat sweep on one fixed COLMAP dataset.

Each configuration is independent, with the same cameras, masks, RNG and LR
schedule. No native changes, no oracle geometry; this is a control experiment.
"""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
from eval_psnr import evaluate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SETTINGS={
    "default":dict(refine_start=500,refine_stop=15000,refine_every=100,grow_grad2d=0.0002),
    "limited":dict(refine_start=500,refine_stop=6000,refine_every=200,grow_grad2d=0.0003),
    "strict":dict(refine_start=500,refine_stop=4000,refine_every=300,grow_grad2d=0.0004),
}

def prepare_command(dataset,dest,steps,seed,setting):
    return [sys.executable,str(HERE/"run_baseline.py"),
        "--dataset",str(dataset),"--output",str(dest),
        "--image-dir","images","--mask-dir","masks",
        "--steps",str(steps),"--max-steps","30000","--shuffle-seed",str(seed),
        "--refine-start",str(setting["refine_start"]),
        "--refine-stop",str(setting["refine_stop"]),
        "--refine-every",str(setting["refine_every"]),
        "--grow-grad2d",str(setting["grow_grad2d"])]

def run_config(dataset,dest,steps,seed,name,setting):
    dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/"result.json"
    if result_path.is_file():
        result=json.loads(result_path.read_text())
        if result["steps"]!=steps or result["seed"]!=seed or result["condition"]!=name:
            raise ValueError("Stale cached result: "+str(dest))
        return result
    if list(dest.rglob("train.json")):
        raise RuntimeError("Partial run exists, inspect manually: "+str(dest))
    cmd=prepare_command(dataset,dest,steps,seed,setting)
    begin=time.perf_counter()
    with (dest/"train.log").open("w") as log:
        subprocess.run(cmd,cwd=ROOT/"vksplat",check=True,stdout=log,stderr=subprocess.STDOUT)
    wall=time.perf_counter()-begin
    runs=list(dest.glob("*/train.json"))
    if len(runs)!=1:raise RuntimeError("Expected one training result: "+str(dest))
    run=runs[0].parent
    meta=json.loads(runs[0].read_text())
    quality=evaluate(run)
    (run/"eval_psnr.json").write_text(json.dumps(quality,indent=2)+"\n")
    result=dict(condition=name,steps=steps,seed=seed,
        psnr_db=quality["mean_psnr_db"],median_psnr_db=quality["median_psnr_db"],
        train_s=meta["time_elapsed"],wall_s=wall,
        splats=meta["num_splats"],peak_vram_bytes=meta["peak_vram"],
        dataset=str(dataset),run_dir=str(run),**setting)
    result_path.write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--conditions",choices=tuple(SETTINGS),nargs="+",default=["limited","strict"])
    a=p.parse_args()
    if a.steps<=0:p.error("steps must be positive")
    data=a.dataset.expanduser().resolve();out=a.output.expanduser().resolve()
    if not (data/"masks").is_dir():p.error("Expected ScanNet++ masks folder")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in a.conditions:
        print("Running",name,flush=True)
        result=run_config(data,out/name,a.steps,a.seed,name,SETTINGS[name])
        rows.append(result)
        print(name,round(result["psnr_db"],3),"dB",
            round(result["train_s"],2),"s",result["splats"],"splats",flush=True)
        with (out/"summary.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
    print("Summary:",out/"summary.csv")

if __name__=="__main__":main()
