# P1d: Reproducible convergence screen and ScanNet++ pilot

## Objective

Validate whether the +2.350 dB controlled oracle-completion advantage observed
at 250 steps in Replica room0 is robust across image-order seeds and compute
budgets, then test generalization on one real ScanNet++ scene.

**This first implementation runs independent jobs at each budget.** It does
NOT yet render intermediate checkpoints during a single optimizer run, and
thus does not measure continuous time-to-target quality. Training time from
VkSplat excludes oracle preparation and process setup.

## Replica short screen

On the existing Strix Halo workstation, from the VkSplat checkout:

```bash
cd ~/git/vksplat
git switch main
git pull --ff-only origin main
pixi run python -m py_compile sceneweave/repeated_controlled_sweep.py sceneweave/audit_scannetpp.py
pixi run python sceneweave/test_controlled_init.py
pixi run build
pixi run smoke

BASE=~/git/sceneweave/outputs/replica_p1/sparse_sweep_v1/data/seedstep_16/measured
COMP=~/git/sceneweave/outputs/replica_p1/completion_v1
OUT=~/git/sceneweave/outputs/replica_p1/p1d_multiseed

bash sceneweave/run_amd.sh ../sceneweave/repeated_controlled_sweep.py \
  --baseline "$BASE" --completion "$COMP" --output "$OUT" \
  --steps 100 250 500 1000 --seeds 41 42 43 --added-scale-m 0.025
```

36 total independent training runs; an existing completed `result.json`
is reused. If a run fails mid-training the script explicitly refuses to
silently re-use its partial artifacts; inspect the run directory first.
Runs with the same seed share Gaussian native RNG, normalized transforms,
and Python shuffle; they do not represent full cross-reconstruction random
stream equivalence after divergent densification.

The script verifies the common normalization, held-out image names and
reference-prefix count within each budget/seed triplet. Results:
`$OUT/results.csv` plus per-run logs and JSON.

Interpretation: compare B2-B3 improvements per seed at the same step budget;
report mean and spread. Distinguish 100/250/500 pre-densification runs from
1000-step runs (VkSplat starts densification after step 500). Strong early
gains with narrow variability justify richer priors. A single favorable seed
does not.

## ScanNet++ pilot

Use **one known scene** first, preferably a ScanNet++ v2 train/validation
scene with iPhone RGB/depth, aligned iPhone poses, DSLR undistorted images,
and reference laser mesh. Check the licensed datasets already stored locally;
do not copy or redistribute raw ScanNet++ data.

On the workstation:

```bash
DATA=/mnt/cps_persistent1_shared/datasets/public
find "$DATA" -maxdepth 5 -name pose_intrinsic_imu.json 2>/dev/null | head -20
find "$DATA" -maxdepth 5 -name transforms_undistorted.json 2>/dev/null | head -20

# Set SCENE to an actual directory found above (parent of iphone/ or dslr/).
pixi run python sceneweave/audit_scannetpp.py --scene "$SCENE"
```

Use the official ScanNet++ toolbox rather than inventing camera conventions:
- `https://github.com/scannetpp/scannetpp`
- `https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation`

The **iPhone** modality has sparse, noisy device LiDAR depth (nominally
256x192) aligned to RGB, and ARKit-aligned poses. The **DSLR** modality has
high-quality held-out reference images and laser-scan-aligned metric COLMAP
calibration. Only use supplied aligned poses for mesh comparisons.
The original DSLR cameras can be fisheye, so prefer the official
`resized_undistorted_images` with matched transforms.
Anonymization masks must be honored in NVS quality comparisons.

The real-world adapter is **not** implemented yet; it will follow the first
asset audit and explicit validation of pose/mesh alignment. The Replica
converter should NOT be pointed at ScanNet++ directory structures.

## Benchmark integrity

Oracle completion sees the reference mesh; it is privileged and must not
be evaluated as a deployable system on hidden benchmark test scenes.
A meaningful next result will compare data-preparation overhead, training
wall time, geometry accuracy, and held-out PSNR across several available
ScanNet++ scenes. Avoid publishing only a photometric number: metric
geometry and spatial coverage matter to Sceneweave.
