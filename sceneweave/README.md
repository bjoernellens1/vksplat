# Sceneweave support in the VkSplat fork

This folder contains **baseline harnesses only**. No Gaussian kernel, optimizer, loss, or densification behavior is changed. The Sceneweave oracle-initialization experiment is a later, separately controlled patch.

## Why a separate harness?

The external Sceneweave project is responsible for priors and experiment policy; VkSplat is responsible for training. For a valid prior-ablation, we first establish no-prior performance on the exact same VkSplat code and device.

The source root contains a `vksplat/` directory, which Python may import as a namespace package if run from the outer root. The launcher runs from the inner Python source directory to ensure that the compiled extension is loaded.

## Pixi build/install (Linux x86_64)

The fork includes a committed `pixi.toml` for Python 3.12, NumPy, OpenCV, pybind11, setuptools, wheel and tqdm. Keep Vulkan driver/ICD and GLM headers at the OS level.

On Fedora:
```bash
sudo dnf install -y gcc-c++ vulkan-loader-devel vulkan-headers glm-devel pkgconf-pkg-config
cd ~/git/vksplat
bash sceneweave/setup.sh
```

On Ubuntu (22.04 / 24.04 or later, x86_64; actual GPU build still needs validation):
```bash
sudo apt update
sudo apt install -y build-essential libvulkan-dev libglm-dev pkg-config \\
    mesa-vulkan-drivers vulkan-tools libdrm2 libdrm-amdgpu1
cd ~/git/vksplat
bash sceneweave/setup.sh
```

**Required first-time setup:** install the OS packages listed above, ensure `vulkaninfo --summary` detects your physical GPU (not only `llvmpipe`), then run `pixi run setup`. Merely running `pixi run smoke` on a fresh clone will fail because the native extension has not been built. The installer reports distribution-specific prerequisite commands when headers are missing. The runtime launcher discovers the host libdrm locations through `ldconfig` (Fedora: `/lib64` or `/usr/lib64`; Ubuntu: `/lib/x86_64-linux-gnu` or `/usr/lib/x86_64-linux-gnu`). It leaves Vulkan ICD selection to the system by default, so Intel/NVIDIA machines can also use the standard driver. For AMD with problematic ICD selection, set `VK_SPLAT_FORCE_RADV=1`. Set `VK_SPLAT_SYSTEM_DRM=0` only if your host driver is compatible with the Pixi-loaded DRM libraries.

Or perform the steps separately:
```bash
pixi run setup      # resolve dependencies, build and smoke-test
pixi run build      # rebuild editable C++ extension
pixi run smoke      # check that a hardware Vulkan GPU is viable
pixi run benchmark-help
```

These scripts do **not** install PyTorch/TorchMetrics and do not train on a dataset. The system Vulkan device must support VkSplat's requirements. `pixi.lock` should be generated on a working machine and committed to your fork (the repository's original `*.lock` ignore pattern is overridden for `pixi.lock`). On Ubuntu and Fedora the launcher automatically locates system DRM libraries. `VK_SPLAT_SYSTEM_DRM=1` requires them, `0` disables preloading, and the default `auto` preloads them if available. `VK_SPLAT_FORCE_RADV=1` explicitly selects AMD RADV where the manifest exists. This configuration is currently Linux-only; other platforms require their own tested setup.

## Troubleshooting Ubuntu AMD GPU enumeration

A successful `pixi run build` only proves that the Python extension compiled. On the Ubuntu Radeon R9700 machine, the current `pixi run smoke` prints **only llvmpipe**, meaning no viable physical GPU is available to VkSplat *in that process*. This is not yet proof that the R9700 lacks any required Vulkan feature.

First diagnose the host:

```bash
vulkaninfo --summary
ls -l /dev/dri/renderD* 2>/dev/null || true
ls -l /usr/share/vulkan/icd.d/ 
```

If system `vulkaninfo` also lists only llvmpipe, confirm the AMD kernel/DRM driver is loaded and users can access a render node; examine `dmesg`/journal and device permissions. Reinstalling Pixi will not fix an absent host GPU.

If system Vulkan lists the AMD GPU but VkSplat only sees llvmpipe, try the diagnostic RADV override:

```bash
VK_SPLAT_FORCE_RADV=1 pixi run smoke
```

The override is process-local. `VK_SPLAT_SYSTEM_DRM=auto` (default) preloads host `libdrm.so.2` and `libdrm_amdgpu.so.1` when present; `VK_SPLAT_SYSTEM_DRM=0` disables the workaround for isolation tests. `VK_SPLAT_FORCE_RADV=1` requires a matching installed RADV ICD manifest. Other vendor devices should use normal ICD discovery.

Pixi's `gdk-pixbuf` post-link warning is generally unrelated to Vulkan GPU enumeration; do **not** globally enable insecure post-link scripts just to remove that warning.

## Quick start on the tested Fedora 44 + Strix Halo setup

The local environment needs a built VkSplat extension installed in `.pixi/envs/default`. This fork now ships a Pixi manifest and setup scripts. Build once using `bash sceneweave/setup.sh`, then:

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

The output contains a `benchmark.json` with host/OS, Vulkan enumerated GPUs, git revision, COLMAP calibration hashes, per-run training time, full invocation wall time, Gaussian count, peak reported VRAM, and **RGB-only** held-out mean PSNR.

Throughput is reported as **training it/s = configured iterations / VkSplat-reported training seconds**, and **wall it/s = configured iterations / full process invocation seconds** (includes startup, loading, PLY export and rendering). Both are included in the per-run JSON, console output, and cross-machine CSV. The comparator also computes these fields for older benchmark files that only contain timing and step counts. These are *average* throughputs, not GPU kernel-only throughput or steady-state instantaneous rates. Each run stores `train.log`, `train.json`, `config.json`, `eval_psnr.json`, PLY and validation renders.

Copy the completed `benchmark.json` files together and compare:

```bash
bash sceneweave/run_amd.sh ../sceneweave/benchmark.py compare \
  /path/pc-a/benchmark.json /path/pc-b/benchmark.json
```

Comparison requires matching COLMAP hashes, image directory, strategy, and source revision by default. `--allow-different-inputs` permits a **descriptive** table only. Host-reported Vulkan device names can include software devices; verify the **actual selected training GPU from train.log** before interpreting hardware results. Also verify source image content and split equivalence: the comparator hashes COLMAP metadata, not every training image. The user-selected camera dataset and stage must be identical. Mixed revisions, differing hardware drivers or thermal settings should be disclosed.

**Benchmark timing note:** `invocation_wall_seconds` includes launching, loading, training, writing PLY, validation rendering, and any upstream evaluation attempted, but **excludes** the subsequent NumPy/OpenCV PSNR evaluator; use this field to compare the same harness, not full evaluation-to-report latency. The reported `gpu_training_seconds_reported` comes from VkSplat `train.json` and is *not* an independent synchronized GPU-kernel timing measurement.

This benchmark is **not** yet a continuous training time-to-quality experiment: separate step-count runs reinitialize the optimizer. It provides an initial cross-machine sweep. Repeat runs and report variance, matched PSNR and identical image/pose input. Metric geometry needs an independent evaluation.
