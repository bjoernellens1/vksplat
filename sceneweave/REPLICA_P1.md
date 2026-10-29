# Sceneweave P1: Replica -> VkSplat (oracle geometry experiment)

**Implementation state:** first *position-only* oracle initialization via COLMAP text inputs. No neural generator, Gaussian normal anchoring, metric depth loss, or C++ optimizer change. This is a controlled *upper-bound* test, not an end-to-end Sceneweave speedup claim.

## Input format / prerequisites

Expected Replica scene directory:
- `traj.txt`: a row-major flattened 4x4 camera-to-world transform per frame
- `results/frame000000.jpg` (or PNG) and `results/depth000000.png`
- For the oracle branch: a reference mesh file in a format supported by `trimesh`

**Coordinate conventions must be checked on real data before using results:** `--pose-camera opencv` converts camera-to-world OpenGL axes to OpenCV optical axes by right-multiplication with diag(1,-1,-1,1). **Validated against the local NICE-SLAM `room0` and original Replica `room_0` mesh:** `--pose-camera opencv` is required. A direct 20-frame point/mesh vertex check gave median 5.7 mm and 100% within 15 cm; applying `opengl` instead gave 832.9 mm and 9.9%. Other Replica trajectory variants must be verified independently. Depths are assumed Z-depth along the optical axis, not Euclidean ray range. The default `--depth-scale 1000` assumes millimetres stored as unsigned integer; verify your Replica export. Intrinsics must come from the dataset/camera config, never invented.

The converter requires `numpy`, `opencv`, `trimesh`, `scipy` from the project Pixi environment. Run `pixi install`.

## Locate your input

```bash
find /mnt/cps_persistent1_shared/datasets -type f -name traj.txt | head -20
find /mnt/cps_persistent1_shared/datasets -type f -name '*office0*.ply' | head -20
```

Set `SCENE` to the actual Replica scene, `MESH` to its reference triangle mesh, and supply real `fx fy cx cy` from the original camera configuration.

## Produce two identical COLMAP datasets except for seed XYZ

```bash
cd ~/git/vksplat
SCENE=/path/to/Replica/office0
MESH=/path/to/Replica/reference_mesh.ply
OUT=~/git/sceneweave/outputs/replica_p1
# EXAMPLE ONLY: replace all four with real camera calibration
FX=0; FY=0; CX=0; CY=0

bash sceneweave/run_amd.sh ../sceneweave/replica_to_colmap.py \
  --scene "$SCENE" --output "$OUT/measured" --mode measured \
  --pose-camera opencv --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY" \
  --frame-step 20 --pixel-step 12 --max-points 150000

bash sceneweave/run_amd.sh ../sceneweave/replica_to_colmap.py \
  --scene "$SCENE" --output "$OUT/oracle" --mode oracle --mesh "$MESH" \
  --pose-camera opencv --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY" \
  --frame-step 20 --pixel-step 12 --max-points 150000

bash sceneweave/run_amd.sh ../sceneweave/check_replica_pair.py \
  --baseline "$OUT/measured" --oracle "$OUT/oracle"
```

**Do not run with placeholder FX/FY=0.** Replace camera intrinsics before executing. The script refuses invalid/nonpositive focal lengths.

The seed sample, RGB (from training images), image symlinks, and camera poses are shared. In oracle mode, each measured seed XYZ is snapped to a sampled reference mesh surface if within 15 cm. This **does not inject extra object points or privileged appearance**, and does not move distant points; the mesh sampling is an approximation to exact triangle distance. It does provide *privileged geometry* unavailable to the sensor-only baseline. A manifest with `schema_version=1.0.0`, source mode, frame IDs, number of points, and oracle alignment statistics is written to each dataset.

The paired checker asserts equal camera/image metadata, seed count, RGB and IDs, and reports how many positions differ.

## Unmodified VkSplat training

```bash
bash sceneweave/run_amd.sh ../sceneweave/run_baseline.py \
  --dataset "$OUT/measured" --image-dir images --steps 3000 \
  --output "$OUT/runs/measured"

bash sceneweave/run_amd.sh ../sceneweave/run_baseline.py \
  --dataset "$OUT/oracle" --image-dir images --steps 3000 \
  --output "$OUT/runs/oracle"
```

Both paths use the existing COLMAP text reader. The train runner accepts `.txt` or `.bin`, validates the versioned Sceneweave conversion manifest, and prints prior mode. This is **opt-in by dataset selection**; the original Mip-NeRF 360 training path remains unchanged.

### Experimental caveats

- The reference mesh and training camera coordinate frames **must match**. Check reprojected points and a few depth frames before interpreting PSNR.
- VkSplat chooses validation frames deterministically from the image list; this conversion seeds only the chosen training frames (every 8th selected frame held out). Compare using identical splits.
- This first prototype is **RGB-only training**. No depth loss exists in VkSplat here, and high PSNR alone does not prove metric correctness.
- Seed positions are moved but their RGB remains from the original observed depth point; alignment error may degrade performance. Record negative transfer as well as gains.
- Both runs must share image content, count, random seed, Gaussian count, device, driver, learning-rate schedule, and exact trainer revision. The existing training code uses unseeded randomness; repeat with multiple runs and add controlled random seeds before publication-grade causal claims.
- The oracle snapping preprocessing time, depth-fusion time and mesh loading must be recorded and compared. The current training runner times only training; this does **not** yet provide a complete end-to-end speedup claim.
- For a proper *metric* oracle benchmark, add independent depth/mesh quality evaluation and a depth-supervised training branch; neither is implemented by this experiment.
- Future P1.3/P1.4 add oriented Gaussians, geometry anchors and freeze/unfreeze as **separate** ablations, not as part of this position-only experiment.
