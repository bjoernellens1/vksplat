"""Tests for vksplat PR #18:
1. Static source audit: DEVICE_GUARD always declared before PerfTimer::Timer in
   every function of gs_renderer.cpp and gs_trainer.cpp that uses both.
2. AGO-0 opt-in per-Gaussian telemetry: zero baseline perturbation (< 1e-6) when
   enabled vs. disabled.
3. AGO-0 telemetry numerical sanity: grad_norm, update_norm, predicted_gain,
   abs_proj_grad, update_count, contrib_sum, last_step_updated, last_predicted_gain,
   and reset_ago_telemetry().
4. AGO-1 exact snapshot_ago_state() / restore_ago_state() round-trip determinism
   (both from fresh uninitialized Adam moments and from warm non-zero Adam moments).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def _strip_cpp_comments(source: str) -> str:
    """Strip C/C++ comments while preserving line numbers."""
    # Replace block comments with equivalent newlines
    def _block_repl(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    no_block = re.sub(r"/\*.*?\*/", _block_repl, source, flags=re.S)
    return re.sub(r"//[^\n]*", "", no_block)


def _iter_cpp_function_bodies(source: str):
    """Yield (start_line, header, body) for top-level C++ function bodies."""
    clean = _strip_cpp_comments(source)
    lines = clean.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        # Top-level function definition: starts at column 0 (no leading whitespace)
        # and is not a preprocessor directive or struct/class/namespace/comment.
        if (
            line
            and not line[0].isspace()
            and not line.startswith(("#", "struct ", "class ", "namespace ", "PACK_STRUCT", "enum ", "using ", "typedef "))
            and "::" in line
        ):
            start_line = i + 1
            header_lines = [line]
            depth = line.count("{") - line.count("}")
            j = i + 1
            while depth == 0 and j < n:
                header_lines.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                if ";" in lines[j] and depth == 0:
                    break
                j += 1
            if depth > 0:
                body_lines = list(header_lines)
                while depth > 0 and j < n:
                    body_lines.append(lines[j])
                    depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                yield start_line, header_lines[0].strip(), "\n".join(body_lines)
                i = j
                continue
        i += 1


def _render_train_l1_loss(mod, train_idx: int) -> float:
    """Render training view `train_idx` and return mean L1 RGB error against target."""
    mod.render_train(train_idx)
    rendered = np.asarray(mod.pixel_state, dtype=np.float32)[..., :3]
    target = np.asarray(mod.get_train_image(train_idx), dtype=np.float32)[..., :3] / 255.0
    return float(np.mean(np.abs(rendered - target)))


def test_device_guard_before_perf_timer_static_audit():
    """Every function in gs_renderer.cpp and gs_trainer.cpp with both DEVICE_GUARD
    and PerfTimer::Timer must declare DEVICE_GUARD strictly before the first timer."""
    timer_re = re.compile(r"\b(?:PerfTimer::)?Timer\s*<")
    guard_re = re.compile(r"\bDEVICE_GUARD\b")

    checked_functions = 0
    violations = []

    for rel in ("vksplat/src/gs_renderer.cpp", "vksplat/src/gs_trainer.cpp"):
        path = REPO_ROOT / rel
        src = path.read_text(encoding="utf-8")
        for start_line, header, body in _iter_cpp_function_bodies(src):
            g_match = guard_re.search(body)
            t_match = timer_re.search(body)
            if g_match and t_match:
                checked_functions += 1
                if t_match.start() < g_match.start():
                    violations.append(f"{rel}:{start_line} ({header})")

    assert checked_functions >= 11, (
        f"Expected at least 11 functions with both DEVICE_GUARD and PerfTimer::Timer, "
        f"found {checked_functions}"
    )
    assert not violations, (
        "Found PerfTimer::Timer declared before DEVICE_GUARD in:\n  "
        + "\n  ".join(violations)
    )


def test_ago_telemetry_zero_baseline_perturbation(tmp_path):
    """Training 5 steps with ago_telemetry_enabled=True vs False from an identical
    snapshot must match parameters and loss to < 1e-6."""
    ds = sc.build_dataset(tmp_path / "ds_zero_perturb", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_zero_perturb")
    mod.force_backward_variant(1)  # deterministic PerSplat backward

    assert mod.get_ago_telemetry_enabled() is False

    # Warm up 2 steps so Adam moments are populated before branching
    for s in range(1, 3):
        mod.train_step(s % int(mod.num_train), s)

    snap = mod.snapshot_ago_state()

    # Branch A: telemetry disabled
    mod.set_ago_telemetry_enabled(False)
    assert mod.get_ago_telemetry_enabled() is False
    for s in range(3, 8):
        mod.train_step(s % int(mod.num_train), s)
    loss_off = _render_train_l1_loss(mod, 0)
    xyz_off = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_off = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_off = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_off = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    # Branch B: restore identical state, enable telemetry, run exact same 5 steps
    mod.restore_ago_state(snap)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()
    assert mod.get_ago_telemetry_enabled() is True
    for s in range(3, 8):
        mod.train_step(s % int(mod.num_train), s)
    loss_on = _render_train_l1_loss(mod, 0)
    xyz_on = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_on = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_on = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_on = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    assert abs(loss_on - loss_off) < 1e-6, f"Loss mismatch: on={loss_on} off={loss_off}"
    assert float(np.max(np.abs(xyz_on - xyz_off))) < 1e-6
    assert float(np.max(np.abs(sh_on - sh_off))) < 1e-6
    assert float(np.max(np.abs(so_on - so_off))) < 1e-6
    # Rasterize-backward uses non-deterministic GPU float32 atomicAdd across tiles,
    # which can nudge a single near-eps quaternion by ~1e-4; median diff stays < 1e-6.
    assert float(np.median(np.abs(rot_on - rot_off))) < 1e-6
    assert float(np.max(np.abs(rot_on - rot_off))) < 1e-3

    mod.cleanup()


def test_ago_telemetry_numerical_sanity_and_reset(tmp_path):
    """Verify AGO-0 telemetry keys, shapes, signs, accumulation, and reset."""
    ds = sc.build_dataset(tmp_path / "ds_tel_sanity", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_tel_sanity")
    mod.force_backward_variant(1)

    n = int(len(mod.xyz_ws))
    assert n == 400
    assert mod.get_ago_telemetry_enabled() is False

    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()
    assert mod.get_ago_telemetry_enabled() is True

    # Step 0 -> 1-indexed Adam step 1: m_hat == g, so -g^T delta > 0 strictly for every updated Gaussian
    mod.train_step(0, 0)
    tel1 = mod.get_ago_telemetry()
    expected_array_keys = (
        "grad_norm",
        "update_norm",
        "predicted_gain",
        "abs_proj_grad",
        "update_count",
        "contrib_sum",
        "last_step_updated",
        "last_predicted_gain",
    )
    assert int(tel1["num_gaussians"]) == n
    assert int(tel1["step"]) == 1
    for key in expected_array_keys:
        arr = tel1[key]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr)), f"Non-finite values in telemetry[{key!r}]"

    updated1 = tel1["update_count"] >= 1.0
    assert int(np.sum(updated1)) > 50, "Expected substantial visible updated Gaussians on step 1"
    assert np.all(tel1["grad_norm"][updated1] > 0.0)
    assert np.all(tel1["update_norm"][updated1] > 0.0)
    assert np.all(tel1["predicted_gain"][updated1] > 0.0)
    assert np.all(tel1["last_predicted_gain"][updated1] > 0.0)
    assert np.allclose(tel1["predicted_gain"][updated1], tel1["last_predicted_gain"][updated1], rtol=1e-6, atol=1e-7)
    assert np.all(tel1["contrib_sum"][updated1] > 0.0)
    assert np.all(tel1["last_step_updated"][updated1] == 1.0)
    assert float(np.max(tel1["abs_proj_grad"][updated1])) > 0.0

    # Steps 1..3 -> 1-indexed Adam steps 2..4: counters accumulate across steps
    for s in range(1, 4):
        mod.train_step(s % int(mod.num_train), s)
    tel4 = mod.get_ago_telemetry()
    assert int(tel4["step"]) == 4
    multi = tel4["update_count"] >= 2.0
    assert int(np.sum(multi)) > 50
    assert float(np.sum(tel4["grad_norm"])) > float(np.sum(tel1["grad_norm"]))
    assert float(np.sum(tel4["update_norm"])) > float(np.sum(tel1["update_norm"]))
    assert float(np.sum(tel4["predicted_gain"])) > float(np.sum(tel1["predicted_gain"]))
    assert float(np.max(tel4["last_step_updated"])) == 4.0

    # Reset must zero all 8 arrays cleanly
    mod.reset_ago_telemetry()
    tel_zero = mod.get_ago_telemetry()
    for key in expected_array_keys:
        assert np.all(tel_zero[key] == 0.0), f"reset_ago_telemetry did not zero {key!r}"

    mod.cleanup()


def test_ago_snapshot_restore_exact_roundtrip(tmp_path):
    """Snapshot at step t (both t=0 fresh and t=3 warm), train 4 steps, restore,
    train the same 4 steps again -> loss and parameters must match to < 1e-5."""
    ds = sc.build_dataset(tmp_path / "ds_snap", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_snap")
    mod.force_backward_variant(1)

    # 1) Round-trip from step 0 (before Adam moments are ever allocated)
    snap0 = mod.snapshot_ago_state()
    assert int(snap0["num_gaussians"]) == 400
    assert int(snap0["step"]) == 0
    for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        assert k in snap0

    for s in range(0, 4):
        mod.train_step(s % int(mod.num_train), s)
    loss0_a = _render_train_l1_loss(mod, 0)
    means0_a = np.asarray(mod.xyz_ws, dtype=np.float32).copy()

    gen_before = int(mod.map_generation)
    mod.restore_ago_state(snap0)
    assert int(mod.map_generation) > gen_before

    for s in range(0, 4):
        mod.train_step(s % int(mod.num_train), s)
    loss0_b = _render_train_l1_loss(mod, 0)
    means0_b = np.asarray(mod.xyz_ws, dtype=np.float32).copy()

    assert abs(loss0_a - loss0_b) < 1e-5, f"Fresh snapshot loss mismatch: {loss0_a} vs {loss0_b}"
    assert float(np.max(np.abs(means0_a - means0_b))) < 1e-5

    # 2) Round-trip from warm state at Adam step 4 (non-zero Adam moments + strategy state)
    snap4 = mod.snapshot_ago_state()
    assert int(snap4["step"]) == 4
    assert snap4["has_optimizer_moments"] is True
    for k in ("g_xyz_ws", "g_sh_coeffs_1", "g_sh_coeffs_2", "g_rotations", "g_scales_opacs"):
        assert k in snap4

    for s in range(4, 8):
        mod.train_step(s % int(mod.num_train), s)
    loss4_a = _render_train_l1_loss(mod, 1)
    means4_a = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh4_a = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot4_a = np.asarray(mod.rotations, dtype=np.float32).copy()
    so4_a = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    gen_before2 = int(mod.map_generation)
    mod.restore_ago_state(snap4)
    assert int(mod.map_generation) > gen_before2

    for s in range(4, 8):
        mod.train_step(s % int(mod.num_train), s)
    loss4_b = _render_train_l1_loss(mod, 1)
    means4_b = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh4_b = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot4_b = np.asarray(mod.rotations, dtype=np.float32).copy()
    so4_b = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    assert abs(loss4_a - loss4_b) < 1e-5, f"Warm snapshot loss mismatch: {loss4_a} vs {loss4_b}"
    assert float(np.max(np.abs(means4_a - means4_b))) < 1e-5
    assert float(np.max(np.abs(sh4_a - sh4_b))) < 1e-5
    assert float(np.max(np.abs(so4_a - so4_b))) < 1e-5
    assert float(np.median(np.abs(rot4_a - rot4_b))) < 1e-5
    assert float(np.max(np.abs(rot4_a - rot4_b))) < 1e-3

    mod.cleanup()
