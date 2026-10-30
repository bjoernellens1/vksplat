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

def controlled_reference_count(manifest):
    """Validate a shared original-point prefix for controlled initialization."""
    total=int(manifest["num_seed_points"])
    count=int(manifest.get("p1c",{}).get("original_seed_count",total))
    if not 4<=count<=total:
        raise ValueError(f"Invalid reference prefix {count} of {total}")
    return count

def main():
    p=argparse.ArgumentParser(description="Reproducible no-prior VkSplat training baseline")
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image-dir",default="images_2")
    p.add_argument("--mask-dir",default=None,help="Optional grayscale mask directory relative to dataset")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--strategy",choices=("default","mcmc"),default="default")
    p.add_argument("--cap-max",type=int,default=1_000_000)
    p.add_argument("--controlled-init",action="store_true",
                   help="Use shared original seeds for normalization/KNN and fixed-scale additions")
    p.add_argument("--added-scale-m",type=float,default=0.025,
                   help="Metric-world scale for appended Gaussians in controlled-init mode")
    p.add_argument("--shuffle-seed",type=int,default=None,
                   help="Deterministic RGB order; defaults to 42 in controlled mode")
    args=p.parse_args()
    if args.steps<=0 or args.cap_max<=0:
        p.error("steps and cap-max must be positive")
    if args.controlled_init and not (0 < args.added_scale_m < 10):
        p.error("--added-scale-m must be positive and finite")
    dataset=args.dataset.expanduser().resolve()
    for stem in ("cameras","images","points3D"):
        if not any((dataset/"sparse"/"0"/f"{stem}.{ext}").is_file() for ext in ("bin","txt")):
            p.error(f"Missing COLMAP {stem}.bin or {stem}.txt in {dataset/'sparse/0'}")
    manifest=None
    manifest_file = dataset/"sceneweave_conversion.json"
    if manifest_file.exists():
        manifest=json.loads(manifest_file.read_text())
        if manifest.get("schema_version")!="1.0.0" or manifest.get("mode") not in ("oracle","measured"):
            p.error("Unsupported Sceneweave conversion manifest")
        print("Prior mode:", manifest["mode"],"oracle:",manifest["oracle_privileged"],flush=True)
    if args.controlled_init and manifest is None:
        p.error("--controlled-init requires a versioned Sceneweave dataset manifest")
    if not (dataset/args.image_dir).is_dir():
        p.error(f"Missing image directory: {dataset/args.image_dir}")
    if args.mask_dir and not (dataset/args.mask_dir).is_dir():
        p.error(f"Missing mask directory: {dataset/args.mask_dir}")
    output=args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg_cls=MCMCTrainerConfig if args.strategy=="mcmc" else TrainerConfig
    cfg=cfg_cls(dataset_dir=str(dataset),image_dir=args.image_dir,mask_dir=args.mask_dir,sparse_dir="sparse/0",
        output_dir=str(output),train_steps=args.steps,max_steps=30000,
        enable_viewer=False,save_train_renders=False)
    if args.strategy=="mcmc":
        cfg.cap_max=args.cap_max
    if args.controlled_init:
        try:
            cfg.init_reference_count=controlled_reference_count(manifest)
        except (ValueError, TypeError, KeyError) as exc:
            p.error(str(exc))
        total=int(manifest["num_seed_points"])
        cfg.init_added_scale_m=args.added_scale_m
        cfg.shuffle_seed=42 if args.shuffle_seed is None else args.shuffle_seed
        print(f"Controlled init: reference={cfg.init_reference_count}/{total}, "
              f"added scale={cfg.init_added_scale_m:.4f} m, "
              f"shuffle seed={cfg.shuffle_seed}",flush=True)
    else:
        cfg.shuffle_seed=args.shuffle_seed
    metadata={"dataset":str(dataset),"steps":args.steps,"strategy":args.strategy,
              "image_dir":args.image_dir,"mask_dir":args.mask_dir,"created_utc":datetime.now(timezone.utc).isoformat(),
              "prior":manifest["mode"] if manifest else "none",
              "controlled_init":args.controlled_init,
              "reference_seed_count":cfg.init_reference_count,
              "added_scale_m":cfg.init_added_scale_m,
              "shuffle_seed":cfg.shuffle_seed,
              "quality_scope":"photometric_only"}
    (output/"sceneweave_run_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Baseline manifest:",output/"sceneweave_run_manifest.json",flush=True)
    train_main(cfg)

if __name__=="__main__":
    main()
