"""V17.13 P3 GPU tests: 10k-step opacity-reset structural stability, ADC dupli/split/prune/Morton
metadata inheritance, and stabilization LR decay schedule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def _make_checker_scene(tmp_path: Path, n_fg: int = 120, w: int = 64, h: int = 48) -> Path:
    import cv2

    img_dir = tmp_path / "images"
    sparse_dir = tmp_path / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    f, cx, cy = 0.9 * w, w / 2.0, h / 2.0
    (sparse_dir / "cameras.txt").write_text(f"# Camera list\n1 PINHOLE {w} {h} {f} {f} {cx} {cy}\n")

    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    checker = (((xx.astype(int) // 4) + (yy.astype(int) // 4)) % 2).astype(np.float32)
    lines = ["# Image list"]
    for i in range(4):
        name = f"frame_{i:04d}.png"
        img = np.stack(
            [
                30 + 200 * checker,
                220 - 180 * checker,
                70 + 140 * np.sin(xx * 0.35 + i),
            ],
            axis=-1,
        )
        cv2.imwrite(str(img_dir / name), np.clip(img, 0, 255).astype(np.uint8))
        lines += [f"{i + 1} 1.0 0.0 0.0 0.0 {0.05 * (i - 1.5):.4f} 0.0 2.2 1 {name}", ""]
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    pts = ["# 3D point list"]
    for pid in range(1, n_fg + 1):
        px = rng.uniform(-0.45, 0.45)
        py = rng.uniform(-0.32, 0.32)
        pz = rng.uniform(-0.12, 0.12)
        pts.append(f"{pid} {px:.5f} {py:.5f} {pz:.5f} 128 128 128 0.01 1 0")
    (sparse_dir / "points3D.txt").write_text("\n".join(pts) + "\n")
    return tmp_path


def test_unrolled_3000_6000_9000_reset_preserves_walls_zeros_contradicted_and_flushes_floaters(tmp_path):
    """Unrolled training across steps 1..9150 (hitting resets at 3000, 6000, 9000 and post-reset prunes):
    - Unobserved structural surfels (0u) keep exact opacity across resets and prunes.
    - Observed/metric-verified large wall surfels (2u/3u, scale > prune_scale3d, even off-frustum)
      clamp gently to structural_reset_opa (0.25) instead of 0.01 and survive all post-3000 prunes.
    - Contradicted structural surfels (4u) are zeroed on reset and pruned away.
    - Transparent and bloated off-frustum measured floaters (gid=-1, 5u) are flushed.
    """
    ds = _make_checker_scene(tmp_path / "ds", n_fg=80)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        max_steps=10000,
        refine_start_iter=500,
        refine_stop_iter=9500,
        refine_every=100,
        reset_every=3000,
        stop_reset_at=-1,
        pause_refine_after_reset=0,
        prune_opa=0.005,
        prune_scale3d=0.05,  # strict 3D scale threshold so large wall surfels exceed prune_scale3d after step 3000
        grow_grad2d=0.05,    # keep growth modest so 9150 steps run fast
        cap_max=2000,
    )
    mod.set_stabilization_schedule(structural_reset_opa=0.28, lr_final_scale=0.25, max_steps=10000)
    sched = mod.get_stabilization_schedule()
    assert abs(sched["structural_reset_opa"] - 0.28) < 1e-6
    assert abs(sched["feature_lr_final_scale"] - 0.25) < 1e-6
    assert abs(sched["opacity_lr_final_scale"] - 0.25) < 1e-6
    assert abs(sched["scale_lr_final_scale"] - 0.25) < 1e-6
    assert sched["max_steps"] == 10000

    n_fg = len(mod.opacities)
    assert n_fg == 80
    # Mark initial foreground splats as MEASURED_FREE (gid=-1, state=5u); make 15 of them transparent floaters
    # and 15 of them huge off-frustum floaters that must get flushed after step 3000.
    xyz_fg = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    so_fg = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()
    so_fg[:15, 3] = 0.002  # transparent floaters (< prune_opa)
    xyz_fg[15:30] = np.array([12.0, 12.0, -8.0], dtype=np.float32)  # off-frustum bloated floaters
    so_fg[15:30, :3] = 0.85
    so_fg[15:30, 3] = 0.75
    so_fg[30:, 3] = 0.65
    mod._set_xyz_ws(xyz_fg.reshape(-1))
    mod._set_scales_opacs(so_fg.reshape(-1))
    mod.set_splat_group_ids(np.full(n_fg, -1, dtype=np.int32))
    mod.set_splat_states(np.full(n_fg, 5, dtype=np.uint32))

    rng = np.random.default_rng(99)

    def _append_offscreen_group(m: int, gid: int, state_u32: int, scale_tangent: float, opac_val: float, x_off: float):
        xyz = np.column_stack(
            [
                np.full(m, x_off, dtype=np.float32),
                rng.uniform(-0.5, 0.5, size=m).astype(np.float32),
                np.full(m, -6.0, dtype=np.float32),  # behind cameras -> 0 views, pure intrinsic survival test
            ]
        )
        sh = np.zeros((m, 16, 3), dtype=np.float32)
        rot = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (m, 1))
        sc_arr = np.tile(np.array([[scale_tangent, scale_tangent, 0.01]], dtype=np.float32), (m, 1))
        op_arr = np.full(m, opac_val, dtype=np.float32)
        gids_arr = np.full(m, gid, dtype=np.int32)
        st_arr = np.full(m, state_u32, dtype=np.uint32)
        mod.append_splats_preserving_optimizer(xyz, sh, rot, sc_arr, op_arr, gids_arr, st_arr)

    # Group 10: 16 unobserved wall surfels (state=0u, opacity=0.82, huge tangent scale=0.60)
    _append_offscreen_group(16, gid=10, state_u32=0, scale_tangent=0.60, opac_val=0.82, x_off=8.0)
    # Group 11: 20 observed-trainable wall surfels (state=2u, opacity=0.80, huge tangent scale=0.60)
    _append_offscreen_group(20, gid=11, state_u32=2, scale_tangent=0.60, opac_val=0.80, x_off=9.0)
    # Group 12: 20 metric-verified wall surfels (state=3u, opacity=0.75, huge tangent scale=0.60)
    _append_offscreen_group(20, gid=12, state_u32=3, scale_tangent=0.60, opac_val=0.75, x_off=10.0)
    # Group 13: 14 contradicted surfels (state=4u, opacity=0.90) -> must zero at reset/prune and vanish
    _append_offscreen_group(14, gid=13, state_u32=4, scale_tangent=0.20, opac_val=0.90, x_off=11.0)

    n_train = int(mod.num_train)
    # Run steps 1..3000 first so we can inspect immediately after step 3000 reset (before 3100 prune)
    slots_3000 = [s % n_train for s in range(1, 3001)]
    mod.train_steps(slots_3000, 1)

    gids_3000 = np.asarray(mod.get_splat_group_ids(), dtype=np.int32)
    opac_3000 = np.asarray(mod.opacities, dtype=np.float32).reshape(-1)
    # Transparent measured floaters (:15) already pruned at step 600; contradicted group 13 already pruned too!
    assert int(np.sum(gids_3000 == 13)) == 0
    # Inject a fresh contradicted batch right before step 6000 to also test resetOpacity zeroing on 4u directly!
    _append_offscreen_group(12, gid=14, state_u32=4, scale_tangent=0.20, opac_val=0.88, x_off=11.5)
    # Verify Group 10 (0u) stayed untouched at 0.82 and Groups 11/12 (2u/3u) clamped gently to 0.28 (NOT 0.01!)
    assert int(np.sum(gids_3000 == 10)) == 16
    assert int(np.sum(gids_3000 == 11)) == 20
    assert int(np.sum(gids_3000 == 12)) == 20
    np.testing.assert_allclose(opac_3000[gids_3000 == 10], 0.82, atol=1e-6)
    np.testing.assert_allclose(opac_3000[gids_3000 == 11], 0.28, atol=1e-6)
    np.testing.assert_allclose(opac_3000[gids_3000 == 12], 0.28, atol=1e-6)

    # Continue unrolled steps 3001..9150 (crossing post-3000 prunes, step 6000 reset, step 9000 reset, step 9100 prune)
    slots_rest = [s % n_train for s in range(3001, 9151)]
    mod.train_steps(slots_rest, 3001)

    gids_end = np.asarray(mod.get_splat_group_ids(), dtype=np.int32)
    states_end = np.asarray(mod.get_splat_states(), dtype=np.uint32)
    opac_end = np.asarray(mod.opacities, dtype=np.float32).reshape(-1)
    xyz_end = np.asarray(mod.xyz_ws, dtype=np.float32)

    assert len(gids_end) == len(states_end) == len(opac_end) == len(xyz_end)
    # All unobserved (0u) and observed/verified (2u/3u) large structural wall surfels survived 3000, 6000, 9000!
    assert int(np.sum(gids_end == 10)) == 16
    assert int(np.sum(gids_end == 11)) == 20
    assert int(np.sum(gids_end == 12)) == 20
    np.testing.assert_allclose(opac_end[gids_end == 10], 0.82, atol=1e-6)
    np.testing.assert_allclose(opac_end[gids_end == 11], 0.28, atol=1e-6)
    np.testing.assert_allclose(opac_end[gids_end == 12], 0.28, atol=1e-6)

    # Contradicted groups (13, 14) and off-screen bloated measured floaters at x ~ 12.0 were completely flushed
    assert int(np.sum(gids_end == 13)) == 0
    assert int(np.sum(gids_end == 14)) == 0
    assert int(np.sum((gids_end < 0) & (xyz_end[:, 0] > 11.8))) == 0
    mod.cleanup()


def test_adc_duplicate_split_prune_morton_preserves_exact_parent_child_metadata(tmp_path):
    """Simultaneous ADC duplicate + split + prune + MortonSort preserves exact parent->child
    inheritance for group_id, state, cell_id, entity_id, and 8-float splat_anchor with exact lengths.
    """
    ds = _make_checker_scene(tmp_path / "ds", n_fg=160)
    mod = sc.new_trainer(
        ds,
        tmp_path / "out",
        strategy="default",
        refine_start_iter=1,
        refine_stop_iter=100,
        refine_every=4,
        reset_every=3000,
        grow_grad2d=1e-6,
        grow_scale3d=0.06,  # splats <= 0.06 duplicate; splats > 0.06 split!
        grow_scale2d=1.0,
        refine_scale2d_stop_iter=0,
        prune_opa=0.01,
        prune_scale3d=100.0,
        prune_scale2d=100.0,
        cap_max=20000,
    )
    n0 = len(mod.opacities)
    assert n0 == 160

    so = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()
    # Partition 160 splats into:
    # - [0:50]: small scale (0.02 <= grow_scale3d) -> triggers DUPLICATE
    # - [50:100]: large scale (0.14 > grow_scale3d) -> triggers SPLIT
    # - [100:130]: transparent measured splats (opac = 0.001 < prune_opa) -> triggers PRUNE
    # - [130:160]: structural unobserved splats (state = 0u) -> protected from growth & prune
    so[:50, :3] = 0.02
    so[:50, 3] = 0.70
    so[50:100, :3] = 0.14
    so[50:100, 3] = 0.70
    so[100:130, :3] = 0.03
    so[100:130, 3] = 0.001
    so[130:160, :3] = 0.12
    so[130:160, 3] = 0.75
    mod._set_scales_opacs(so.reshape(-1))

    gids0 = np.full(n0, -1, dtype=np.int32)
    gids0[25:50] = 2     #observed structural group 2 (duplicates)
    gids0[75:100] = 3    # verified structural group 3 (splits)
    gids0[130:160] = 4   # unobserved structural group 4 (frozen)

    states0 = np.full(n0, 5, dtype=np.uint32)
    states0[25:50] = 2
    states0[75:100] = 3
    states0[130:160] = 0

    eids0 = (np.arange(n0, dtype=np.int32) % 7) + 1
    cids0 = np.arange(n0, dtype=np.int32) % 8
    credits0 = np.full(8, 500, dtype=np.int32)

    # Encode each parent's exact (gid, state, cid, eid) inside its 8-float anchor (with w_depth=0, w_plane=0)
    # so after duplicate + split + prune + MortonSort every child's anchor must match its inherited metadata!
    anchors0 = np.zeros((n0, 8), dtype=np.float32)
    anchors0[:, 0] = gids0.astype(np.float32)
    anchors0[:, 1] = states0.astype(np.float32)
    anchors0[:, 2] = cids0.astype(np.float32)
    anchors0[:, 4] = eids0.astype(np.float32)

    mod.set_splat_group_ids(gids0)
    mod.set_splat_states(states0)
    mod.set_splat_entity_ids(eids0)
    mod.set_local_cell_credits(cids0, credits0)
    mod.set_splat_anchors(anchors0)

    n_train = int(mod.num_train)
    for step in range(1, 9):
        mod.train_step(step % n_train, step)

    n_final = len(mod.opacities)
    gids_f = np.asarray(mod.get_splat_group_ids(), dtype=np.int32)
    dev_gids_f = np.asarray(mod.get_device_group_ids(), dtype=np.int32)[:n_final]
    states_f = np.asarray(mod.get_splat_states(), dtype=np.uint32)
    cids_f = np.asarray(mod.get_splat_cell_ids(), dtype=np.int32)
    eids_f = np.asarray(mod.get_splat_entity_ids(), dtype=np.int32)
    anchors_f = np.asarray(mod.get_splat_anchors(), dtype=np.float32)
    rem_creds = np.asarray(mod.get_local_cell_credits(), dtype=np.int32)

    # Exact length invariants across all host & device metadata arrays
    assert len(gids_f) == n_final
    assert len(states_f) == n_final
    assert len(cids_f) == n_final
    assert len(eids_f) == n_final
    assert anchors_f.shape == (n_final, 8)
    assert np.array_equal(dev_gids_f, gids_f)

    # Verify both duplicate (group 2 grew > 25) and split (group 3 grew > 25) AND prune (30 transparent pruned) ran
    assert int(np.sum(gids_f == 2)) > 25, "Expected duplicate growth in group 2"
    assert int(np.sum(gids_f == 3)) > 25, "Expected split growth in group 3"
    assert int(np.sum(gids_f == 4)) == 30, "Unobserved group 4 (0u) must neither grow nor prune"
    total_spawned = int(np.sum(credits0 - rem_creds))
    assert total_spawned > 40
    assert n_final == (n0 - 30) + total_spawned

    # Every single surviving parent and spawned duplicate/split child after MortonSort must match its anchor signature!
    np.testing.assert_array_equal(anchors_f[:, 0].astype(np.int32), gids_f)
    np.testing.assert_array_equal(anchors_f[:, 1].astype(np.uint32), states_f)
    np.testing.assert_array_equal(anchors_f[:, 2].astype(np.int32), cids_f)
    np.testing.assert_array_equal(anchors_f[:, 4].astype(np.int32), eids_f)
    mod.cleanup()
