# Sceneweave P1c — Oracle geometry completion

**Goal:** determine whether adding *geometry in uncovered regions* has more value than slightly correcting already observed seed positions.

**Inputs:** prepared P1b measured COLMAP dataset (prefer `seedstep_16/measured`), geometrically registered Replica `room_0/mesh.ply`.

**Output conditions:**
- B0: unchanged measured sparse seed dataset.
- B2 `oracle_completion`: original seed points + mesh-sampled points at least a configurable distance from the **sparse seed cloud**.
- B3 `capacity_control`: identical original seed points + same count of jittered sampled observed points.

The B2/B3 additions have identical **neutral RGB (128,128,128)** and matched counts. Training image list/camera metadata/original seed XYZ/RGB are identical. B2 uses privileged mesh geometry; B3 uses no mesh geometry in its added points (though generating both datasets together uses a mesh for B2).

**Limitation:** distance from sparse initializer seeds is only a *coverage proxy*, not explicit observed-frustum/occlusion visibility. Additional points can be directly visible in RGB training views even if absent from the depth initializer; that is intentional for this prototype. B2 lacks predicted textures, learned shape priors, surface normals, or proper visibility filtering. The B3 control is not an optimal density baseline; stronger baseline controls remain future work. Added point preprocessing time is excluded from GPU training time.

## Commands (Strix Halo, VkSplat main)

```bash
cd ~/git/vksplat
pixi run python -m py_compile sceneweave/oracle_completion.py sceneweave/test_oracle_completion.py
pixi run python sceneweave/test_oracle_completion.py

BASE=~/git/sceneweave/outputs/replica_p1/sparse_sweep_v1/data/seedstep_16/measured
MESH=/mnt/cps_persistent1_shared/datasets/public/Replica-Dataset/room_0/mesh.ply
OUT=~/git/sceneweave/outputs/replica_p1/completion_v1

# One-time, CPU-only P1c dataset generation:
bash sceneweave/run_amd.sh ../sceneweave/oracle_completion.py \
  --baseline "$BASE" --mesh "$MESH" --output "$OUT" \
  --add-points 30000 --min-distance 0.08 --seed 0

# B2/B3 paired short training:
for MODE in oracle_completion capacity_control; do
  bash sceneweave/run_amd.sh ../sceneweave/run_baseline.py \
    --dataset "$OUT/$MODE" --image-dir images --steps 1000 \
    --output "$OUT/runs/${MODE}_1000"
done

# Evaluate each generated run directory:
for MODE in oracle_completion capacity_control; do
  RUN=$(find "$OUT/runs/${MODE}_1000" -name train.json -printf '%h\n' | head -1)
  bash sceneweave/run_amd.sh ../sceneweave/eval_psnr.py --run "$RUN"
done
```

Compare B2/B3 against existing P1b `seedstep_16` B0, **only if the inputs and trainer revision match**. For the clearest controls, rerun B0 at the same time, on the same hardware. Three-plus repeated runs with controlled random states are required before claiming a causal benefit.

The script fails if it cannot find enough mesh candidates beyond the distance threshold, rather than silently adding fewer points. Output manifests contain `p1c.version=1.0.0`, seed counts, threshold, mesh path, RNG seed, sample count, candidate fraction, and explicit privilege flag. It will refuse overwriting a nonempty destination.

Next improvements: actual image visibility/covisibility mask, controlled initialization random seeds, matched-count random-in-free-space and surface-near-observation controls, metric held-out geometry evaluation, full cost accounting, and time-to-quality checkpoint curves.
