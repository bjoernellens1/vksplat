"""GPU tests for V17.12 P1: GPU-resident ADC growth-cap, mutual-exclusion child-count fix,
local cell credits (ADC & MCMC), and actual GPU memory consumption reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "vksplat"
sys.path.insert(0, str(ROOT))
import shader_freshness  # noqa: E402


def _make_clustered_scene(tmp_path: Path, n_per_cell: int = 500, w: int = 64, h: int = 48) -> Path:
    """Create a 2-cluster COLMAP scene where cluster 0 (x < -0.25) and cluster 1 (x > +0.25)
    are both strongly visible in front of all training cameras and produce high 2D gradients.
    """
    import cv2

    img_dir = tmp_path / "images"
    sparse_dir = tmp_path / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    f, cx, cy = 0.9 * w, w / 2.0, h / 2.0
    (sparse_dir / "cameras.txt").write_text(f"# Camera list\n1 PINHOLE {w} {h} {f} {f} {cx} {cy}\n")

    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    checker = (((xx.astype(int) // 3) + (yy.astype(int) // 3)) % 2).astype(np.float32)
    lines = ["# Image list"]
    for i in range(4):
        name = f"frame_{i:04d}.png"
        img = np.stack([
            20 + 220 * checker,
            230 - 200 * checker,
            80 + 120 * np.sin(xx * 0.4 + i),
        ], axis=-1)
        cv2.imwrite(str(img_dir / name), np.clip(img, 0, 255).astype(np.uint8))
        lines += [f"{i + 1} 1.0 0.0 0.0 0.0 {0.06 * (i - 1.5):.4f} 0.0 2.2 1 {name}", ""]
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    pts = ["# 3D point list"]
    pid = 1
    for cell_idx, x_range in ((0, (-0.65, -0.25)), (1, (0.25, 0.65))):
        for _ in range(n_per_cell):
            px = rng.uniform(x_range[0], x_range[1])
            py = rng.uniform(-0.35, 0.35)
            pz = rng.uniform(-0.2, 0.2)
            # Deliberately mismatched uniform grey initial colour to induce strong gradients everywhere
            pts.append(f"{pid} {px:.5f} {py:.5f} {pz:.5f} 128 128 128 0.01 1 0")
            pid += 1
    (sparse_dir / "points3D.txt").write_text("\n".join(pts) + "\n")
    return tmp_path


def test_shader_freshness_and_mtime_invariants():
    assert shader_freshness.check(ROOT / "shader") == []
    slang_mtimes = {p.name: p.stat().st_mtime for p in (ROOT / "slang").glob("*.slang")}
    for phase_spv in (ROOT / "shader" / "generated").glob("default_*.spv"):
        assert phase_spv.stat().st_mtime > slang_mtimes["default.slang"], phase_spv.name
    for phase_spv in (ROOT / "shader" / "generated").glob("mcmc_*.spv"):
        assert phase_spv.stat().st_mtime > slang_mtimes["mcmc.slang"], phase_spv.name


def test_adc_local_cell_credits_zero_readback_and_strict_cell_isolation(tmp_path):
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=500)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        refine_start_iter=2,
        refine_stop_iter=100,
        refine_every=4,
        grow_grad2d=1e-6,
        grow_scale3d=0.05,
        grow_scale2d=0.01,
        refine_scale2d_stop_iter=100,
        prune_opa=0.0,  # isolate pure growth accounting
        prune_scale3d=100.0,
        prune_scale2d=100.0,
        cap_max=50000,
    )
    xyz0 = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert xyz0.shape == (1000, 3)
    # Assign cell 0 to x < 0 (500 splats) and cell 1 to x >= 0 (500 splats)
    cell_ids = np.where(xyz0[:, 0] < 0.0, 0, 1).astype(np.int32)
    assert int(np.sum(cell_ids == 0)) == 500
    assert int(np.sum(cell_ids == 1)) == 500

    credits_init = np.array([2000, 0], dtype=np.int32)
    mod.set_local_cell_credits(cell_ids, credits_init)
    assert mod.get_adc_host_readbacks() == 0

    n_train = int(mod.num_train)
    for step in range(1, 9):
        mod.train_step(step % n_train, step)

    assert mod.get_adc_host_readbacks() == 0, "Normal ADC steps within effective_cap must do 0 host mask readbacks"

    cids_after = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert len(cids_after) == len(xyz_after)

    n_cell0 = int(np.sum(cids_after == 0))
    n_cell1 = int(np.sum(cids_after == 1))
    spawned_cell0 = n_cell0 - 500
    spawned_cell1 = n_cell1 - 500

    assert spawned_cell0 > 0, "Cell 0 (credit=2000) must grow under ADC"
    assert spawned_cell1 == 0, f"Cell 1 (credit=0) must spawn 0 splats, got {spawned_cell1}"
    # Spatial check: cell 1 splats remain at 500 on x > 0.1
    assert int(np.sum(xyz_after[:, 0] > 0.1)) == 500

    rem_credits = np.asarray(mod.get_local_cell_credits(), dtype=np.int32)
    assert rem_credits.shape == (2,)
    assert rem_credits[0] == 2000 - spawned_cell0
    assert rem_credits[1] == 0
    mod.cleanup()


def test_adc_speculative_cap_mutual_exclusion_never_overshoots(tmp_path):
    """Small scale + huge 2D radius previously triggered BOTH dupli and split (2 children/parent)
    and blew past cap_speculative_max. Verify GPU-side mutual exclusion + spec_remaining atomic cap.
    """
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=200)
    cap_spec = 135
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        refine_start_iter=1,
        refine_stop_iter=100,
        refine_every=3,
        grow_grad2d=1e-7,
        grow_scale3d=10.0,           # every splat has is_small == True -> would trigger is_dupli
        grow_scale2d=1e-5,           # every visible splat has running_radii > grow_scale2d -> ALSO triggers is_split!
        refine_scale2d_stop_iter=100,
        prune_opa=0.0,
        prune_scale3d=100.0,
        prune_scale2d=100.0,
        cap_max=50000,
        cap_speculative_max=cap_spec,
        mcmc_protect_structural_groups=True,
    )
    n0 = len(mod.opacities)
    assert n0 == 400
    # Mark 100 splats as speculative group 0 with OBSERVED_TRAINABLE (state=2u), 300 ungrouped (-1, state=5u)
    gids = np.full(n0, -1, dtype=np.int32)
    gids[:100] = 0
    states = np.full(n0, 5, dtype=np.uint32)
    states[:100] = 2
    mod.set_splat_group_ids(gids)
    mod.set_splat_states(states)

    n_train = int(mod.num_train)
    for step in range(1, 10):
        mod.train_step(step % n_train, step)

    gids_after = np.asarray(mod.get_splat_group_ids(), dtype=np.int32)
    spec_total = int(np.sum(gids_after >= 0))
    assert spec_total == cap_spec, f"Expected exact cap_spec={cap_spec}, got {spec_total}"
    assert spec_total <= cap_spec
    assert mod.get_adc_host_readbacks() == 0
    mod.cleanup()


def test_adc_effective_cap_host_trim_exact_child_count(tmp_path):
    """When effective_cap IS exceeded, verify host fallback trim runs and respects exact cap."""
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=200)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        refine_start_iter=1,
        refine_stop_iter=100,
        refine_every=3,
        grow_grad2d=1e-7,
        grow_scale3d=10.0,
        grow_scale2d=1e-5,
        refine_scale2d_stop_iter=100,
        prune_opa=0.0,
        prune_scale3d=100.0,
        prune_scale2d=100.0,
        cap_max=450,
    )
    xyz0 = np.asarray(mod.xyz_ws, dtype=np.float32)
    cids = np.where(xyz0[:, 0] < 0.0, 0, 1).astype(np.int32)
    mod.set_local_cell_credits(cids, np.array([1000, 0], dtype=np.int32))

    n_train = int(mod.num_train)
    for step in range(1, 5):
        mod.train_step(step % n_train, step)

    assert len(mod.opacities) == 450
    assert mod.get_adc_host_readbacks() >= 1
    # Credits refunded for trimmed candidates: exactly 50 splats spawned in cell 0 -> 950 left
    rem = np.asarray(mod.get_local_cell_credits(), dtype=np.int32)
    assert rem[0] == 950 and rem[1] == 0
    mod.cleanup()


def test_mcmc_local_cell_credits_growth_and_relocation_isolation(tmp_path):
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=300)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="mcmc",
        init_scale=0.1,
        init_opacity=0.5,
        scale_reg=0.01,
        opacity_reg=0.01,
        refine_start_iter=1,
        refine_stop_iter=100,
        refine_every=3,
        grow_factor=1.25,
        cap_max=2000,
        mcmc_protect_structural_groups=True,
    )
    xyz0 = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert xyz0.shape == (600, 3)
    cell_ids = np.where(xyz0[:, 0] < 0.0, 0, 1).astype(np.int32)
    n0_cell0 = int(np.sum(cell_ids == 0))
    n0_cell1 = int(np.sum(cell_ids == 1))
    assert n0_cell0 == 300 and n0_cell1 == 300

    # Mark 20 splats in cell 0 as dead (opacity = 1e-4 < min_opacity=0.005) to also exercise relocate_gs!
    so = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()
    cell0_indices = np.where(cell_ids == 0)[0]
    so[cell0_indices[:20], 3] = 1e-4
    mod._set_scales_opacs(so.reshape(-1))

    # Cell 0 gets 500 credits, Cell 1 gets 0 credits (1-view uncorroborated surface)
    mod.set_local_cell_credits(cell_ids, np.array([500, 0], dtype=np.int32))

    # Snapshot/restore roundtrip before training
    snap = mod.snapshot_optimizer_state()
    mod.clear_local_cell_credits()
    assert len(mod.get_local_cell_credits()) == 0
    mod.restore_optimizer_state(snap)
    assert np.array_equal(mod.get_local_cell_credits(), np.array([500, 0], dtype=np.int32))

    n_train = int(mod.num_train)
    for step in range(1, 10):
        mod.train_step(step % n_train, step)

    st = mod.get_mcmc_stats()
    assert st["added_count"] > 0
    assert st["relocated_count"] >= 20

    cids_after = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert len(cids_after) == len(xyz_after)

    n_after_cell0 = int(np.sum(cids_after == 0))
    n_after_cell1 = int(np.sum(cids_after == 1))
    assert n_after_cell0 == 300 + int(st["added_count"])
    assert n_after_cell1 == 300, f"Cell 1 (credit=0) must spawn 0 splats and receive 0 relocations, got {n_after_cell1 - 300}"
    # Verify cell 1 splats on x > 0.1 were neither grown nor relocated away
    assert int(np.sum(xyz_after[:, 0] > 0.1)) == 300

    rem = np.asarray(mod.get_local_cell_credits(), dtype=np.int32)
    assert rem[0] == 500 - int(st["added_count"])
    assert rem[1] == 0
    mod.cleanup()


def test_gpu_memory_stats_positive_and_consistent(tmp_path):
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=200)
    mod = sc.new_trainer(ds, tmp_path / "out")
    mod.train_step(0, 1)
    mem = mod.get_gpu_memory_stats()
    assert isinstance(mem, dict)
    assert mem["device_buffers_mb"] > 0.0
    assert mem["vram_used_mb"] > 0.0
    assert mem["vram_used_mb"] >= mem["device_buffers_mb"]
    assert mem["source"] in ("VK_EXT_memory_budget", "sysfs_drm", "device_buffers_fallback")
    mod.cleanup()


def test_local_cell_credits_topology_sync_append_and_compact(tmp_path):
    """Verify append_splats_preserving_optimizer pads new splats with cell_id=-1 (ineligible until
    reassigned) and compact_retracted remaps splat_cell_ids without GPU out-of-bounds or leakage.
    """
    ds = _make_clustered_scene(tmp_path / "ds", n_per_cell=150)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        refine_start_iter=1,
        refine_stop_iter=100,
        refine_every=4,
        grow_grad2d=1e-6,
        grow_scale3d=0.05,
        prune_opa=0.0,
        cap_max=10000,
        mcmc_protect_structural_groups=True,
    )
    xyz0 = np.asarray(mod.xyz_ws, dtype=np.float32)
    cids0 = np.where(xyz0[:, 0] < 0.0, 0, 1).astype(np.int32)
    mod.set_local_cell_credits(cids0, np.array([500, 0], dtype=np.int32))

    rng = np.random.default_rng(7)
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 40, 3))
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 32, 4))
    cids_post_append = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    assert len(cids_post_append) == 300 + 72
    assert np.all(cids_post_append[300:] == -1)

    mod.retract_group_sparse(3)
    idx_map = np.asarray(mod.compact_retracted(), dtype=np.int32)
    assert int(np.sum(idx_map >= 0)) == 300 + 32
    cids_compacted = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    assert len(cids_compacted) == 332
    assert np.array_equal(cids_compacted[:300], cids0)
    assert np.all(cids_compacted[300:] == -1)

    for step in range(1, 6):
        mod.train_step(step % int(mod.num_train), step)
    cids_final = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    # Cell 1 (credit=0) and appended unassigned (-1) splats spawn 0 children
    assert int(np.sum(cids_final == 1)) == 150
    assert int(np.sum(cids_final == -1)) == 32
    assert int(np.sum(cids_final == 0)) > 150
    assert mod.get_adc_host_readbacks() == 0
    mod.cleanup()

