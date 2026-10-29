# Sceneweave P1c controlled Gaussian initialization

## Why this experiment exists

The initial P1c control was not scale matched. Original-seed isotropic
KNN scales had median 46.94 mm for the oracle completion and 30.32 mm for
the jittered-duplicate control, because VkSplat recomputed neighbor distances
from the entire point set. World normalization was also derived from all
points, and the Python training shuffle was not seeded.

This opt-in change isolates these confounds **without replacing the previous
datasets or changing default VkSplat behavior**.

With `--controlled-init`:

1. The first `original_seed_count` seeds in a P1c dataset (or all seeds in
   the B0 measured dataset) alone determine the dataparser world transform.
   That exact transform applies to all Gaussian positions and cameras.
2. The same original prefix alone determines initial KNN Gaussian scales.
3. All **added** Gaussians receive one fixed isotropic scale, specified in
   raw-world metres, converted to normalized units using the shared transform.
4. The Python image shuffle is reproducibly seeded (default 42 in controlled
   mode). VkSplat's native random generator already uses seed 42.
5. Initial seed positions, original seed RGB, the training images, and the
   validation images remain unchanged.

The original prefix count is read from the versioned `p1c` manifest.
For B0 without `p1c`, all its measured seed points become the reference
prefix. C++ extensions **must be rebuilt** after pulling the change.

## Run on Strix Halo

```bash
cd ~/git/vksplat
git switch main
git pull --ff-only origin main

pixi run python -m py_compile \
  sceneweave/run_baseline.py sceneweave/check_controlled_runs.py \
  sceneweave/test_controlled_init.py

pixi run python sceneweave/test_controlled_init.py
pixi run python sceneweave/test_oracle_completion.py
pixi run build
pixi run smoke

BASE=~/git/sceneweave/outputs/replica_p1/sparse_sweep_v1/data/seedstep_16/measured
COMP=~/git/sceneweave/outputs/replica_p1/completion_v1
OUT=~/git/sceneweave/outputs/replica_p1/controlled_v1

for STEPS in 250 1000; do
  for MODE in baseline oracle_completion capacity_control; do
    DATASET="$COMP/$MODE"
    [[ "$MODE" == baseline ]] && DATASET="$BASE"
    bash sceneweave/run_amd.sh ../sceneweave/run_baseline.py \
      --dataset "$DATASET" --image-dir images \
      --steps "$STEPS" --controlled-init \
      --added-scale-m 0.025 --shuffle-seed 42 \
      --output "$OUT/steps_${STEPS}/$MODE"
  done
done

bash sceneweave/run_amd.sh ../sceneweave/check_controlled_runs.py \
  --root "$OUT" --steps 250 1000
```

Check the training console output for:

- `Controlled initialization: reference=34200, added=30000`
  (B0 reports added=0).
- `Controlled appended Gaussian scale: 0.025 m raw -> ... normalized`
  (B2/B3 only).
- The audit script must report identical normalized transforms, held-out image
  lists and shuffle seeds within each iteration budget.

Run each condition in an **empty output directory** for unambiguous
comparison. Saved metrics are `train.json`, `eval_psnr.json`, and the
aggregated `controlled_summary.csv`.

## Interpretation

Compare B2 (oracle completion) and B3 (duplicate capacity control) at 250
steps, before default densification starts (after step 500), then again at
1000 steps with growth. Repeat with at least three different shuffle seeds
(e.g. 41,42,43), using separate output roots, before concluding significance.
The added fixed scale of 0.025 m is a first screening value, not a tuned optimum;
later repeat at 0.01 and 0.04 m with both treatments matched.

This test does not match the **spatial density** or fraction of Gaussians
visible in the training views, nor does it endow oracle geometry with useful
appearance, normals, confidence or opacity policies. The extra data preparation
costs are excluded from reported GPU training seconds. Visibility-aware prior
selection and training-derived appearance remain next-stage ablations, not
part of this control.
