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
