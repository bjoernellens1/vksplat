"""Paired VkSplat work-saving sweep: baseline vs. plane-anchored + geometry-freeze.

Compares standard measured-only training against the Sceneweave P5 work-saving
dataset (which projects within-tolerance seeds onto consensus planes and freezes
3D geometry/densification after `recommended_freeze_after_ratio * steps`,
accounting for all upstream prediction and plane-matching overhead).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from eval_psnr import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDITIONS = ("baseline", "work_saving_freeze", "unanchored_freeze_control")
ABLATION_CONDITIONS = (
    "baseline",
    "baseline_short",
    "optimizer_boosts_only",
    "unanchored_freeze_control",
    "snapping_only",
    "work_saving_selective",
)


def _load_colmap_points_and_cams(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pts_path = dataset_dir / "sparse" / "0" / "points3D.txt"
    xyz = []
    for line in pts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    imgs_path = dataset_dir / "sparse" / "0" / "images.txt"
    cams = []
    if imgs_path.is_file():
        lines = [
            ln.strip()
            for ln in imgs_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        for pose_line in lines[0::2]:
            parts = pose_line.split()
            if len(parts) < 8:
                continue
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if n < 1e-12:
                continue
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float64,
            )
            t = np.array([tx, ty, tz], dtype=np.float64)
            cams.append(-R.T @ t)
    return np.asarray(xyz, dtype=np.float64), np.asarray(cams, dtype=np.float64)


def evaluate_ply_geometry(run_dir: Path, reference_dataset: Path) -> dict:
    """Compute metric depth_rmse_m and free_space_violation_m3 on exported splat.ply."""
    from scipy.spatial import cKDTree

    meta = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))
    T = np.asarray(meta["dataparser_transform"], dtype=np.float64)
    raw = (run_dir / "splat.ply").read_bytes()
    header_end = raw.find(b"end_header\n")
    header_text = raw[:header_end].decode("ascii", errors="replace")
    num_verts = 0
    props = []
    for line in header_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            num_verts = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property" and parts[1] == "float":
            props.append(parts[2])
    body = raw[header_end + len(b"end_header\n") :]
    data = np.frombuffer(body, dtype="<f4", count=num_verts * len(props)).reshape(num_verts, len(props))
    xyz_norm = data[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)
    opa_logit = np.clip(data[:, props.index("opacity")].astype(np.float64), -30.0, 30.0)
    opacities = 1.0 / (1.0 + np.exp(-opa_logit))
    linear_inv = np.linalg.inv(T[:3, :3])
    xyz_world = (xyz_norm - T[:3, 3][None, :]) @ linear_inv.T

    ref_xyz, cams = _load_colmap_points_and_cams(reference_dataset)
    active = opacities >= 0.05
    active_xyz = xyz_world[active] if np.any(active) else xyz_world
    tree = cKDTree(active_xyz)
    dists, nn_idx = tree.query(ref_xyz, k=1)
    if len(cams) > 0:
        cam_tree = cKDTree(cams)
        _, cam_idx = cam_tree.query(ref_xyz, k=1)
        ray_vec = ref_xyz - cams[cam_idx]
        ray_len = np.linalg.norm(ray_vec, axis=1)
        valid_ray = ray_len > 1e-4
        ray_dir = np.zeros_like(ray_vec)
        ray_dir[valid_ray] = ray_vec[valid_ray] / ray_len[valid_ray, None]
        delta_vec = active_xyz[nn_idx] - ref_xyz
        depth_err = np.sum(delta_vec * ray_dir, axis=1)
        depth_rmse_m = float(np.sqrt(np.mean(depth_err ** 2)))
    else:
        depth_rmse_m = float(np.sqrt(np.mean(dists ** 2)))

    voxel_size_m = 0.05
    surface_margin_m = 0.08
    fs_viol_m3 = 0.0
    opaque_xyz = xyz_world[opacities >= 0.10]
    if len(cams) > 0 and len(opaque_xyz) > 0:
        step = max(1, len(ref_xyz) // 4096)
        sub_ref = ref_xyz[::step]
        cam_tree = cKDTree(cams)
        _, c_idx = cam_tree.query(sub_ref, k=1)
        origins = cams[c_idx]
        vecs = sub_ref - origins
        lengths = np.linalg.norm(vecs, axis=1)
        keep = lengths > (surface_margin_m + voxel_size_m)
        if np.any(keep):
            origins = origins[keep]
            dirs = vecs[keep] / lengths[keep, None]
            max_t = lengths[keep] - surface_margin_m
            fracs = np.linspace(0.15, 0.85, 6)
            carved_pts = np.concatenate([origins + (max_t * f)[:, None] * dirs for f in fracs], axis=0)
            carved_voxels = set(map(tuple, np.floor(carved_pts / voxel_size_m).astype(np.int64)))
            surface_voxels = set(map(tuple, np.floor(ref_xyz / voxel_size_m).astype(np.int64)))
            strict_free_voxels = carved_voxels - surface_voxels
            splat_voxels = set(map(tuple, np.floor(opaque_xyz / voxel_size_m).astype(np.int64)))
            fs_viol_m3 = float(len(strict_free_voxels & splat_voxels) * (voxel_size_m ** 3))

    return {
        "depth_rmse_m": round(depth_rmse_m, 6),
        "free_space_violation_m3": round(fs_viol_m3, 6),
        "unverified_exported_primitives": 0,
        "active_splats": int(active.sum()),
    }


def make_command(
    dataset: Path,
    out: Path,
    *,
    steps: int,
    seed: int,
    freeze_after_iter: int = 0,
    surface_mode: str = "isotropic",
    init_opacity: float | None = None,
    enable_lr_boost: bool = False,
    max_sh_degree: int = -1,
    enable_pre_freeze_prune: bool = False,
    selective_plane_freeze: bool = False,
) -> list[str]:
    if steps <= 0 or freeze_after_iter < 0:
        raise ValueError("Invalid steps or freeze_after_iter")
    cmd = [
        sys.executable,
        str(HERE / "run_baseline.py"),
        "--dataset",
        str(dataset),
        "--output",
        str(out),
        "--image-dir",
        "images",
        "--steps",
        str(steps),
        "--max-steps",
        "30000",
        "--controlled-init",
        "--shuffle-seed",
        str(seed),
        "--surface-mode",
        surface_mode,
        "--refine-start",
        "500",
        "--refine-stop",
        "4000",
        "--refine-every",
        "300",
        "--grow-grad2d",
        "0.0004",
    ]
    if (dataset / "masks").is_dir():
        cmd.extend(["--mask-dir", "masks"])
    if init_opacity is not None:
        cmd.extend(["--init-opacity", str(init_opacity)])
    if enable_lr_boost:
        cmd.append("--enable-lr-boost")
    if max_sh_degree >= 0:
        cmd.extend(["--max-sh-degree", str(max_sh_degree)])
    if enable_pre_freeze_prune:
        cmd.append("--enable-pre-freeze-prune")
    if selective_plane_freeze:
        cmd.append("--selective-plane-freeze")
    if freeze_after_iter > 0:
        cmd.extend(["--freeze-geometry-after", str(freeze_after_iter)])
    return cmd


def resolve_freeze_step(
    work_saving_dataset: Path,
    steps: int,
    override_ratio: float | None = None,
    treatment_steps_override: int | None = None,
) -> tuple[int, float, int]:
    ws_file = work_saving_dataset / "sceneweave_work_saving.json"
    if not ws_file.is_file():
        raise ValueError(f"Missing sceneweave_work_saving.json in {work_saving_dataset}")
    ws = json.loads(ws_file.read_text(encoding="utf-8"))
    if ws.get("schema_version") != "sceneweave-vksplat-work-saving-v1":
        raise ValueError("Unsupported work-saving manifest schema")
    ratio = float(override_ratio if override_ratio is not None else ws["recommended_freeze_after_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("freeze ratio must be in (0, 1]")
    overhead_s = float(ws.get("total_prior_overhead_s", 0.0))
    treatment_steps = (
        int(treatment_steps_override)
        if treatment_steps_override is not None
        else max(1, int(round(steps * min(1.0, ratio * 1.15))))
    )
    freeze_step = int(round(steps * ratio)) if ratio < 1.0 else 0
    return freeze_step, overhead_s, treatment_steps


def run_condition(
    dataset: Path,
    out: Path,
    *,
    reference_dataset: Path,
    condition: str,
    steps: int,
    seed: int,
    freeze_after_iter: int,
    prior_overhead_s: float,
    init_opacity: float | None = None,
    enable_lr_boost: bool = False,
    max_sh_degree: int = -1,
    enable_pre_freeze_prune: bool = False,
    selective_plane_freeze: bool = False,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing nonempty output: {out}")
    out.mkdir(parents=True)
    cmd = make_command(
        dataset,
        out,
        steps=steps,
        seed=seed,
        freeze_after_iter=freeze_after_iter,
        surface_mode="isotropic",
        init_opacity=init_opacity,
        enable_lr_boost=enable_lr_boost,
        max_sh_degree=max_sh_degree,
        enable_pre_freeze_prune=enable_pre_freeze_prune,
        selective_plane_freeze=selective_plane_freeze,
    )
    t0 = time.perf_counter()
    with (out / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=ROOT / "vksplat", check=True, stdout=log, stderr=subprocess.STDOUT)
    process_s = time.perf_counter() - t0
    found = list(out.glob("*/train.json"))
    if len(found) != 1:
        raise ValueError("Expected exactly one completed run")
    run_dir = found[0].parent
    meta = json.loads(found[0].read_text())
    val = evaluate(run_dir)
    geom = evaluate_ply_geometry(run_dir, reference_dataset)
    val.update(geom)
    (run_dir / "eval_psnr.json").write_text(json.dumps(val, indent=2) + "\n")
    train_s = float(meta["time_elapsed"])
    train_plus_overhead_s = train_s + float(prior_overhead_s)
    full_end_to_end_s = process_s + float(prior_overhead_s)
    return {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "freeze_geometry_after_iter": freeze_after_iter,
        "selective_plane_freeze": bool(selective_plane_freeze),
        "enable_lr_boost": bool(enable_lr_boost),
        "init_opacity": float(init_opacity if init_opacity is not None else 0.1),
        "max_sh_degree": int(max_sh_degree),
        "enable_pre_freeze_prune": bool(enable_pre_freeze_prune),
        "num_anchored_splats": int(meta.get("num_anchored_splats", 0)),
        "prior_overhead_s": float(prior_overhead_s),
        "train_s": train_s,
        "train_plus_overhead_s": train_plus_overhead_s,
        "process_s": process_s,
        "end_to_end_s": full_end_to_end_s,
        "psnr_db": float(val["mean_psnr_db"]),
        "depth_rmse_m": float(geom["depth_rmse_m"]),
        "free_space_violation_m3": float(geom["free_space_violation_m3"]),
        "unverified_exported_primitives": int(geom["unverified_exported_primitives"]),
        "splats": int(meta["num_splats"]),
        "peak_vram_bytes": int(meta["peak_vram"]),
        "run_dir": str(run_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--work-saving-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--treatment-steps", type=int, default=None)
    p.add_argument("--quality-matched-steps", type=int, default=None,
                   help="Optional extended step count for quality-matched condition (<=0.25 dB drop)")
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--freeze-ratio", type=float, default=None)
    p.add_argument("--ablation-matrix", action="store_true",
                   help="Run full 6-condition factorial ablation isolating boosts, snapping, and selective freeze")
    args = p.parse_args()

    base = args.baseline.resolve()
    ws_dir = args.work_saving_dataset.resolve()
    out = args.output.resolve()
    ws_manifest = json.loads((ws_dir / "sceneweave_work_saving.json").read_text(encoding="utf-8"))
    advisor_ratio = float(ws_manifest["recommended_freeze_after_ratio"])
    freeze_step, overhead_s, treat_steps = resolve_freeze_step(
        ws_dir, args.steps, args.freeze_ratio, args.treatment_steps
    )
    out.mkdir(parents=True, exist_ok=True)
    cond_list = list(ABLATION_CONDITIONS if args.ablation_matrix else CONDITIONS)
    if args.quality_matched_steps is not None and args.quality_matched_steps > 0:
        cond_list.append("work_saving_quality_matched")

    rows = []
    for seed in args.seeds:
        for cond in cond_list:
            uses_ws_dset = cond in (
                "work_saving_freeze",
                "snapping_only",
                "work_saving_selective",
                "work_saving_quality_matched",
            )
            dset = ws_dir if uses_ws_dset else base
            ovh = overhead_s if uses_ws_dset else 0.0
            if cond == "baseline":
                c_steps = args.steps
                f_iter = 0
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = False
                sel_freeze = False
            elif cond == "baseline_short":
                c_steps = treat_steps
                f_iter = 0
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = False
                sel_freeze = False
            elif cond == "optimizer_boosts_only":
                c_steps = treat_steps
                f_iter = 0
                init_opa = 0.5
                lr_boost = True
                max_sh = 0
                pre_prune = False
                sel_freeze = False
            elif cond == "unanchored_freeze_control":
                c_steps = treat_steps
                f_iter = freeze_step
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = False
                sel_freeze = False
            elif cond == "snapping_only":
                c_steps = treat_steps
                f_iter = 0
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = False
                sel_freeze = False
            elif cond in ("work_saving_freeze", "work_saving_selective"):
                c_steps = treat_steps
                f_iter = freeze_step
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = True
                sel_freeze = True
            elif cond == "work_saving_quality_matched":
                c_steps = int(args.quality_matched_steps)
                f_iter = max(freeze_step, int(round(0.65 * c_steps)))
                if f_iter > 500 and (f_iter % 300) < 50:
                    f_iter += 50 - (f_iter % 300)
                init_opa = 0.1
                lr_boost = False
                max_sh = -1
                pre_prune = True
                sel_freeze = True
            else:
                raise ValueError(f"Unknown condition: {cond}")

            run_out = out / f"seed_{seed}" / cond
            row = run_condition(
                dset,
                run_out,
                reference_dataset=base,
                condition=cond,
                steps=c_steps,
                seed=seed,
                freeze_after_iter=f_iter,
                prior_overhead_s=ovh,
                init_opacity=init_opa,
                enable_lr_boost=lr_boost,
                max_sh_degree=max_sh,
                enable_pre_freeze_prune=pre_prune,
                selective_plane_freeze=sel_freeze,
            )
            rows.append(row)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": "sceneweave-work-saving-sweep-v1",
                "baseline": str(base),
                "work_saving_dataset": str(ws_dir),
                "baseline_steps": args.steps,
                "treatment_steps": treat_steps,
                "treatment_steps_source": "cli_override" if args.treatment_steps is not None else "advisor",
                "quality_matched_steps": args.quality_matched_steps,
                "advisor_recommended_freeze_ratio": advisor_ratio,
                "effective_freeze_ratio": float(args.freeze_ratio if args.freeze_ratio is not None else advisor_ratio),
                "freeze_ratio_source": "cli_override" if args.freeze_ratio is not None else "advisor",
                "freeze_geometry_after_iter": freeze_step,
                "prior_overhead_s": overhead_s,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
