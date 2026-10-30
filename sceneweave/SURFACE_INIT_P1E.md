# P1e: Measured-surface Gaussian covariance initialization

This experiment follows the ScanNet++ strict-growth oracle result:

| Condition | PSNR | Training |
|---|---:|---:|
| Measured-only | 22.969 dB | 54.84 s |
| Scene-wide oracle +12k | 22.977 dB | 42.89 s |
| Jittered capacity control +12k | 22.986 dB | 44.81 s |

The oracle did NOT show a compelling geometry-specific advantage over a
matched capacity control at seed 42. The new experiment **does not add any
seed points**, does not retrieve a mesh, and uses no privileged laser scan.

## Three matched initializers

1. `isotropic`: existing random quaternion and equal KNN-derived radii.
2. `random_anisotropic`: identical centers, colors, opacity and tangential
   KNN radii, normal-axis radius 0.25×, random quaternion.
3. `aligned`: same anisotropic radii, but the thin axis follows a local
   normal estimated exclusively from measured original-prefix neighbors.

The normal estimator currently picks the maximum-area cross product from the
12 nearest measured points. This is a **fast heuristic, not robust PCA plane
fitting**. It can be noisy at edges or mixed surfaces. Both anisotropic
conditions use the same validity mask, so the initial eigenvalues match.
Future work: weighted local PCA, confidence gating, and analytic depth-derived
normals. The existing KNN scale estimator remains unchanged.

Run on the same ScanNet++ measured dataset as the earlier strict-growth
benchmark, with 10k steps, 30k-step LR schedule, frozen 175/25 view split,
strict growth start 500 / stop 4000 / every 300 / threshold 0.0004.

## Run in fish (no terminal exit)

```fish
cd ~/git/vksplat
git pull --ff-only origin main

pixi run python sceneweave/test_surface_init_sweep.py
pixi run python -m py_compile sceneweave/run_baseline.py sceneweave/surface_init_sweep.py
pixi run build
pixi run smoke

set DATA ~/git/sceneweave/outputs/scannetpp/a492fe77aa/window_200
set OUT ~/git/sceneweave/outputs/scannetpp/a492fe77aa/surface_init_v1

bash sceneweave/run_amd.sh ../sceneweave/surface_init_sweep.py \
    --dataset "$DATA" \
    --output "$OUT" \
    --steps 10000 \
    --seeds 41 42 43 \
    --normal-scale-ratio 0.25
```

Output: `summary.csv`, `comparison.json`, per-condition `train.json`,
`eval_psnr.json` and full `run.log`. The command rotates run order by
seed to partially mitigate sequence-dependent GPU clocks.

For a fast correctness screen before the full comparison use
`--steps 250 --seeds 42` and a separate output path. Always rebuild the
native extension after pulling; the C++ trainer config and init logic changed.

## Interpretation

- `aligned` versus `random_anisotropic` isolates the estimated orientation
  while holding anisotropic covariance eigenvalues approximately fixed.
- `random_anisotropic` versus `isotropic` measures effect of shape alone.
- All three start with the same measured positions/colors and no guessed
  surfaces. Record initial PSNR, gradient convergence, final Gaussian count,
  full process time and training time, not just iteration count.
- These are fixed-iteration held-out **photometric** comparisons, not equivalent
  quality time-to-threshold measurements. No 3× end-to-end claim follows.
- If alignment does not help, inspect normal quality and evaluate stronger
  local PCA or directly supervised normals before asserting structural
  parameterization is ineffective.
