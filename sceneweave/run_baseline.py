"""Standalone VkSplat baseline runner. No Sceneweave or GPU policy changes."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import sys
# Script lives in sceneweave/ while the upstream trainer lives in vksplat/.
# Insert the actual trainer directory ahead of Python's namespace-package path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vksplat'))
from simple_trainer import TrainerConfig, MCMCTrainerConfig, train_main

def main():
    p=argparse.ArgumentParser(description="Reproducible no-prior VkSplat training baseline")
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image-dir",default="images_2")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--strategy",choices=("default","mcmc"),default="default")
    p.add_argument("--cap-max",type=int,default=1_000_000)
    args=p.parse_args()
    if args.steps<=0 or args.cap_max<=0:
        p.error("steps and cap-max must be positive")
    dataset=args.dataset.expanduser().resolve()
    for stem in ("cameras","images","points3D"):
        if not any((dataset/"sparse"/"0"/f"{stem}.{ext}").is_file() for ext in ("bin","txt")):
            p.error(f"Missing COLMAP {stem}.bin or {stem}.txt in {dataset/'sparse/0'}")
    manifest_file = dataset/"sceneweave_conversion.json"
    if manifest_file.exists():
        manifest=json.loads(manifest_file.read_text())
        if manifest.get("schema_version")!="1.0.0" or manifest.get("mode") not in ("oracle","measured"):
            p.error("Unsupported Sceneweave conversion manifest")
        print("Prior mode:", manifest["mode"],"oracle:",manifest["oracle_privileged"],flush=True)
    if not (dataset/args.image_dir).is_dir():
        p.error(f"Missing image directory: {dataset/args.image_dir}")
    output=args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg_cls=MCMCTrainerConfig if args.strategy=="mcmc" else TrainerConfig
    cfg=cfg_cls(dataset_dir=str(dataset),image_dir=args.image_dir,sparse_dir="sparse/0",
        output_dir=str(output),train_steps=args.steps,max_steps=30000,
        enable_viewer=False,save_train_renders=False)
    if args.strategy=="mcmc":
        cfg.cap_max=args.cap_max
    metadata={"dataset":str(dataset),"steps":args.steps,"strategy":args.strategy,
              "image_dir":args.image_dir,"created_utc":datetime.now(timezone.utc).isoformat(),
              "prior":"none","quality_scope":"photometric_only"}
    (output/"sceneweave_run_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Baseline manifest:",output/"sceneweave_run_manifest.json",flush=True)
    train_main(cfg)

if __name__=="__main__":
    main()
