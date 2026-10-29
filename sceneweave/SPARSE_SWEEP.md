# Sceneweave P1b — Sparse depth initialization sweep

**Research question:** Does privileged geometric correction improve RGB reconstruction when fewer RGB-D frames are used to **initialize** the Gaussian cloud?

The full RGB training set and every-eight-frame held-out validation images remain fixed. This is **not** a sparse-view RGB supervision experiment. The reference mesh is privileged oracle information, not a predictor. It must be geometrically aligned with the measured scene.

## Verified local input: Replica NICE-SLAM room0

The existing room0 trajectory uses **OpenCV optical camera-to-world**, as verified against the original Replica room_0 mesh (median vertex distance 5.7 mm with OpenCV, 832.9 mm after an unnecessary OpenGL flip).

Run from the VkSplat checkout on Strix Halo, after checking out this branch:

```bash
cd ~/git/vksplat
SCENE=/mnt/cps_persistent1_shared/datasets/public/Replica-NICE-SLAM/Replica/room0
MESH=/mnt/cps_persistent1_shared/datasets/public/Replica-Dataset/room_0/mesh.ply
OUT=~/git/sceneweave/outputs/replica_p1/sparse_sweep_v1

# First verify three paired datasets, no GPU training:
bash sceneweave/run_amd.sh ../sceneweave/sparse_sweep.py \
  --scene "$SCENE" --mesh "$MESH" --output "$OUT" \
  --pose-camera opencv --fx 600 --fy 600 --cx 599.5 --cy 339.5 \
  --depth-scale 6553.5 --prepare-only

# Once parity checks pass, run the full first-pass sweep:
bash sceneweave/run_amd.sh ../sceneweave/sparse_sweep.py \
  --scene "$SCENE" --mesh "$MESH" --output "$OUT" \
  --pose-camera opencv --fx 600 --fy 600 --cx 599.5 --cy 339.5 \
  --depth-scale 6553.5
```

Default budgets: seed frame step 1, 4, 16 (from the same 87 training frames), training 250, 500, 1000, 3000 iterations, 3 independent repetitions, measured/oracle. **72 training runs total**. Set `--train-steps 250 500 1000 --repeats 1` for a quick 18-run screen; `--seed-frame-steps 1 4 16` is the default.

Output: `sweep.csv`, individual `result.json`, `stdout.log`, and evaluated `eval_psnr.json`. Existing completed runs resume. An exception aborts on any failed training run. Check `stdout.log` in the run directory for failures.

## Experimental controls and limitations

- Only initializer depth coverage is reduced; RGB supervision/held-out images are held constant. This tests **geometry prior vs. sparse geometric evidence**, not full sparse-view reconstruction.
- Measured/oracle counterparts share the same initial seed **IDs, RGB, count, camera and image lists**. Oracle mode snaps the measured positions toward 500k sampled reference mesh points if the nearest sample is within 15 cm.
- Increasing seed frame stride may *also* reduce initial Gaussian count: report `seed_points` and `seed_frame_count` rather than pretending population is controlled across densities.
- Native VkSplat random state is not synchronized between measured and oracle runs. Three repeats characterize noise but do **not** yet ensure same random quaternions or training shuffle. Fixed-seed support is a prerequisite for strong causal claims.
- The code records GPU training seconds and process wall seconds separately; **mesh oracle preprocessing is not included** in timing and no deployable end-to-end speedup is claimed.
- Training remains RGB-only. No metric depth loss or 3D geometry evaluation. At each matched budget, compare paired mean held-out PSNR and Gaussian growth, not merely iteration throughput.
- A strong next milestone is fixed optimizer RNG, intermediate PSNR checkpoints to estimate actual time-to-quality, and a separate limited-RGB-view experiment.
