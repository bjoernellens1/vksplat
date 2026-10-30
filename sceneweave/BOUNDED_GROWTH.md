# ScanNet++ bounded-growth screening (P2 preparation)

Baseline: ScanNet++ a492fe77aa `window_200` at 10,000 steps, same 175
training views and 25 validation views, seeded image-order 42.

Existing reference measurements:
- Default VkSplat: 22.7256 dB, 167.99 s training, 2,094,482 splats.
- More aggressive policy: 22.8094 dB, 346.01 s, 5,185,319 splats.

The aggressive configuration changed multiple knobs and is not a validated
speedup. The bounded-growth sweep holds the optimizer 30k-step learning-rate
schedule, input images, initial RGB-D points and training seed fixed.
It varies only the densification schedule:
- `limited`: start 500, stop 6000, interval 200, gradient threshold 0.0003.
- `strict`: start 500, stop 4000, interval 300, gradient threshold 0.0004.
- Optional `default`: start 500, stop 15000, interval 100, threshold 0.0002.

These are screening hypotheses; achieved final Gaussian population is not
an exact cap. The actual caps/quality must be determined empirically.

Run with the VkSplat Pixi launcher on Strix Halo:

```fish
cd ~/git/vksplat
git pull --ff-only origin main
pixi run python -m py_compile sceneweave/bounded_growth_sweep.py sceneweave/run_baseline.py

set DATA ~/git/sceneweave/outputs/scannetpp/a492fe77aa/window_200
set OUT ~/git/sceneweave/outputs/scannetpp/a492fe77aa/bounded_growth_v1

bash sceneweave/run_amd.sh ../sceneweave/bounded_growth_sweep.py \
  --dataset "$DATA" --output "$OUT" --steps 10000 \
  --seed 42 --conditions limited strict
```

Outputs in each condition: `train.log`, `result.json`,
`*/train.json`, `*/eval_psnr.json`. Aggregate: `summary.csv`.

Compare against the existing default run (22.7256 dB, 168 s, 2.09M splats)
using PSNR, training time, VRAM and number of splats. The script does not
automatically ingest the historical baseline. If one limited-growth variant
maintains >=22.5 dB with much less compute, repeat it and default at more
random seeds and then test geometry-guided placement under an equivalent
Gaussian population. Exclude data conversion and startup time from VkSplat's
reported training duration; the runner logs full invocation wall time too.
