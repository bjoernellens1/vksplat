# Sceneweave support in the VkSplat fork

This folder contains **baseline harnesses only**. No Gaussian kernel, optimizer, loss, or densification behavior is changed. The Sceneweave oracle-initialization experiment is a later, separately controlled patch.

## Why a separate harness?

The external Sceneweave project is responsible for priors and experiment policy; VkSplat is responsible for training. For a valid prior-ablation, we first establish no-prior performance on the exact same VkSplat code and device.

The source root contains a `vksplat/` directory, which Python may import as a namespace package if run from the outer root. The launcher runs from the inner Python source directory to ensure that the compiled extension is loaded.

## Quick start on the tested Fedora 44 + Strix Halo setup

The local environment needs a built VkSplat extension installed in `.pixi/envs/default`. **This fork does not ship a Pixi environment manifest or build VkSplat**. Reuse the local one already created, then:

```bash
cd ~/git/vksplat
bash sceneweave/run_amd.sh -c 'import vksplat; print(vksplat.__file__)'
bash sceneweave/run_amd.sh -c 'import vksplat; from pathlib import Path; e=vksplat.VkSplat(); e.initialize(str(Path("shader").resolve())+"/", 0)'

bash sceneweave/run_amd.sh ../sceneweave/run_baseline.py \
  --dataset /mnt/cps_persistent1_shared/datasets/public/mipnerf360_v2_dataset/room \
  --image-dir images_2 --steps 1000 \
  --output ~/git/sceneweave/outputs/vksplat/baseline
```

The run creates a timestamped `*_room` directory under the output directory. Evaluate one completed directory:

```bash
bash sceneweave/run_amd.sh ../sceneweave/eval_psnr.py \
  --run ~/git/sceneweave/outputs/vksplat/baseline/REPLACE_WITH_RUN_DIRECTORY
```

## Validity

- PSNR uses original decoded precision (8-bit or 16-bit) and full RGB, not resized or re-encoded 8-bit previews.
- No separate depth or ground-truth surface metric exists for Mip-NeRF 360 in this harness. **Do not claim metric correctness from this PSNR.**
- No cross-backend speedup attribution. Repeat identical runs with the same VkSplat revision, frame split, AMD driver and fixed training configuration.
- Our earlier baseline: 1000 steps, 9.3 s, 163298 Gaussians, mean PSNR 24.982 dB from a separately written image evaluator. This is a user-supplied prior run, not reproduced by this committed harness; the original source normalization/bit-depth handling must be reconciled before using it in a formal comparison.
- Runtime DRM workaround is deliberately local and opt-out through `VK_SPLAT_SYSTEM_DRM=0`; Radeon ICD forcing can be disabled using `VK_SPLAT_FORCE_RADV=0`. System library compatibility is environment-specific.
- Do not silently install PyTorch/TorchMetrics just for basic PSNR. The trainer's optional evaluation may print a skipped notice.

## Cross-machine benchmark (new)

Run the same benchmark on each computer using the same dataset, resolution, strategy, steps and VkSplat revision. Each run is independent, with a fresh optimizer and its own output folder.

```bash
cd ~/git/vksplat
bash sceneweave/run_amd.sh ../sceneweave/benchmark.py run \
  --dataset /mnt/cps_persistent1_shared/datasets/public/mipnerf360_v2_dataset/room \
  --image-dir images_2 \
  --steps 1000 3000 \
  --repeats 3 \
  --output ~/git/sceneweave/outputs/vksplat/cross_machine
```

The output contains a `benchmark.json` with host/OS, Vulkan enumerated GPUs, git revision, COLMAP calibration hashes, per-run training time, full invocation wall time, Gaussian count, peak reported VRAM, and **RGB-only** held-out mean PSNR. Each run stores `train.log`, `train.json`, `config.json`, `eval_psnr.json`, PLY and validation renders.

Copy the completed `benchmark.json` files together and compare:

```bash
bash sceneweave/run_amd.sh ../sceneweave/benchmark.py compare \
  /path/pc-a/benchmark.json /path/pc-b/benchmark.json
```

Comparison requires matching COLMAP hashes, image directory, strategy, and source revision by default. `--allow-different-inputs` permits a **descriptive** table only. Host-reported Vulkan device names can include software devices; verify the **actual selected training GPU from train.log** before interpreting hardware results. Also verify source image content and split equivalence: the comparator hashes COLMAP metadata, not every training image. The user-selected camera dataset and stage must be identical. Mixed revisions, differing hardware drivers or thermal settings should be disclosed.

**Benchmark timing note:** `invocation_wall_seconds` includes launching, loading, training, writing PLY, validation rendering, and any upstream evaluation attempted, but **excludes** the subsequent NumPy/OpenCV PSNR evaluator; use this field to compare the same harness, not full evaluation-to-report latency. The reported `gpu_training_seconds_reported` comes from VkSplat `train.json` and is *not* an independent synchronized GPU-kernel timing measurement.

This benchmark is **not** yet a continuous training time-to-quality experiment: separate step-count runs reinitialize the optimizer. It provides an initial cross-machine sweep. Repeat runs and report variance, matched PSNR and identical image/pose input. Metric geometry needs an independent evaluation.
