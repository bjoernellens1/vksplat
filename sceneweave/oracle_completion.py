"""P1c: matched-capacity oracle surface completion versus non-oracle control.

Uses an already prepared measured Replica COLMAP dataset. New points have
neutral colors in BOTH variants; images/cameras and original seeds are identical.
Uncovered means 'far from sparse seed points', NOT proven invisible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree


def read_seeds(path: Path):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 8:
            raise ValueError("Invalid COLMAP points3D record")
        xyz.append([float(v) for v in cols[1:4]])
        rgb.append([int(v) for v in cols[4:7]])
    if not xyz:
        raise ValueError("Empty measured seed cloud")
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def sample_mesh(mesh_path: Path, count: int, seed: int):
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Oracle source must be a triangle mesh")
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.all(np.isfinite(areas)) or areas.sum() <= 0:
        raise ValueError("Invalid mesh triangle areas")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(areas), size=count, p=areas / areas.sum())
    tri = mesh.triangles[indices]
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1 - uv[flip]
    return tri[:, 0] + uv[:, 0, None] * (tri[:, 1] - tri[:, 0]) + uv[:, 1, None] * (tri[:, 2] - tri[:, 0])


def candidates_away_from_seeds(seeds, sampled, minimum_distance):
    distances = cKDTree(seeds).query(sampled, workers=1)[0]
    return sampled[distances >= minimum_distance], distances


def make_additions(seeds, mesh, count, min_distance, sample_multiplier, seed, jitter):
    pool = sample_mesh(mesh, max(count * sample_multiplier, 10000), seed)
    far, distances = candidates_away_from_seeds(seeds, pool, min_distance)
    if len(far) < count:
        raise ValueError(
            f"Only {len(far)} candidate points at least {min_distance} m from "
            f"measured seeds; requested {count}. Reduce count/distance.")
    # The two variants receive the same *number* of neutral-colored additions.
    oracle = far[:count].copy()
    rng = np.random.default_rng(seed + 1)
    control = seeds[rng.choice(len(seeds), count, replace=True)].copy()
    control += rng.normal(0, jitter, size=control.shape)
    return oracle, control, {
        "sampled_mesh_points": len(pool),
        "candidate_far_points": len(far),
        "fraction_far": float(len(far) / len(pool)),
        "median_distance_sample_to_seed_m": float(np.median(distances)),
    }


def write_dataset(base, destination, original, rgb, extra, mode, stats, params):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Refusing nonempty destination: {destination}")
    sparse = destination / "sparse" / "0"
    sparse.mkdir(parents=True)
    (destination / "images").symlink_to((base / "images").resolve(), target_is_directory=True)
    for name in ("cameras.txt", "images.txt"):
        shutil.copy2(base / "sparse" / "0" / name, sparse / name)
    with (sparse / "points3D.txt").open("w") as stream:
        stream.write("# ID X Y Z R G B ERROR TRACK[]\n")
        xyz = np.concatenate([original, extra])
        colors = np.concatenate([rgb, np.full((len(extra), 3), 128, dtype=np.uint8)])
        for i, (p, col) in enumerate(zip(xyz, colors), 1):
            stream.write(f"{i} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
                         f"{int(col[0])} {int(col[1])} {int(col[2])} 0\n")
    manifest = json.loads((base / "sceneweave_conversion.json").read_text())
    # Keep the existing runner's accepted conversion schema while adding a
    # separately versioned experiment description.
    manifest["mode"] = "oracle" if mode == "oracle_completion" else "measured"
    manifest["oracle_privileged"] = mode == "oracle_completion"
    manifest["num_seed_points"] = len(original) + len(extra)
    manifest["dataset"] = str(destination)
    manifest["p1c"] = dict(version="1.0.0", condition=mode,
                          original_seed_count=len(original), added_count=len(extra),
                          color_for_added_points=[128, 128, 128],
                          candidate_definition="sampled mesh points >= threshold from seed positions",
                          caveat="Sparse-seed distance is not visibility or observation proof",
                          **params, **stats)
    (destination / "sceneweave_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_pair(a: Path, b: Path, baseline_count: int, added_count: int):
    for name in ("cameras.txt", "images.txt"):
        assert (a / "sparse" / "0" / name).read_bytes() == (b / "sparse" / "0" / name).read_bytes()
    pa, ca = read_seeds(a / "sparse" / "0" / "points3D.txt")
    pb, cb = read_seeds(b / "sparse" / "0" / "points3D.txt")
    assert pa.shape == pb.shape == (baseline_count + added_count, 3)
    np.testing.assert_array_equal(pa[:baseline_count], pb[:baseline_count])
    np.testing.assert_array_equal(ca, cb)
    assert np.all(ca[baseline_count:] == 128)
    return {"initial_seed_count": baseline_count, "added_count": added_count,
            "final_seed_count": len(pa), "shared_colors": True,
            "shared_initial_seed_xyz": True, "shared_training_and_validation_images": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Prepared sparse measured COLMAP dataset, e.g. seedstep_16/measured")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--add-points", type=int, default=30000)
    parser.add_argument("--min-distance", type=float, default=0.08,
                        help="Candidate mesh points farther than this from any baseline seed")
    parser.add_argument("--sample-multiplier", type=int, default=20)
    parser.add_argument("--control-jitter", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.add_points <= 0 or args.sample_multiplier <= 0 or args.min_distance < 0 or args.control_jitter < 0:
        parser.error("Invalid sample count or distance arguments")
    base, out = args.baseline.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output must be new/empty: {out}")
    src = base / "sparse" / "0"
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (src / name).is_file():
            parser.error(f"Missing {name} in {src}")
    source_manifest = json.loads((base / "sceneweave_conversion.json").read_text())
    if source_manifest.get("mode") != "measured":
        parser.error("P1c must use an unmodified measured-depth seed dataset")
    seeds, colors = read_seeds(src / "points3D.txt")
    oracle, control, stats = make_additions(
        seeds, args.mesh.resolve(), args.add_points, args.min_distance,
        args.sample_multiplier, args.seed, args.control_jitter)
    params = dict(mesh=str(args.mesh.resolve()), add_points=args.add_points,
                  min_distance_m=args.min_distance, sample_multiplier=args.sample_multiplier,
                  control_jitter_m=args.control_jitter, random_seed=args.seed)
    write_dataset(base, out / "oracle_completion", seeds, colors, oracle,
                  "oracle_completion", stats, params)
    write_dataset(base, out / "capacity_control", seeds, colors, control,
                  "capacity_control", stats, params)
    result = validate_pair(out / "oracle_completion", out / "capacity_control",
                           len(seeds), args.add_points)
    print(json.dumps({"pair": result, "candidate_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
