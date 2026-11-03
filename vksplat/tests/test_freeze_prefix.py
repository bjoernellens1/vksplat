"""Frozen-prefix and rendered-depth GPU tests (skipped without the built extension / a Vulkan device).

Run (from vksplat/): VK_SPLAT_FORCE_RADV=1 ../sceneweave/run_amd.sh -m unittest discover -s tests
"""
import ast
import re
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_trace import config_hash, depth_from_render, z_to_ray_range  # noqa: E402
from test_db0_trace import _write_synthetic_colmap  # noqa: E402

try:
    import vksplat
    from simple_trainer import TrainerConfig, join_dir
    _probe = vksplat.VkSplat()
    _probe.initialize(join_dir(str(ROOT), "shader"), -1)
    _probe.cleanup()
    GPU_ERR = None
except Exception as e:  # no extension / no Vulkan device
    GPU_ERR = str(e)

needs_gpu = unittest.skipIf(GPU_ERR, f"vksplat/Vulkan unavailable: {GPU_ERR}")
SH = 32 if GPU_ERR else vksplat.SH_REORDER_SIZE  # freeze_prefix_count granularity
N = 10 * SH  # initial splat count of the synthetic scene (multiple of SH_REORDER_SIZE)
PARAMS = ("xyz_ws", "scales", "rotations", "opacities", "sh_coeffs")


def state(m):
    return {k: np.array(getattr(m, k)) for k in PARAMS}


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def write_custom_ply(path, rows):
    """rows: dicts(xyz, rgb=(r,g,b) DC color, scale=(sx,sy,sz), opacity, quat=(w,x,y,z) optional) -> 3DGS PLY."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.array([[*r["xyz"], *r["rgb"], np.log(r["opacity"] / (1 - r["opacity"])), *np.log(r["scale"]),
                      *r.get("quat", (1, 0, 0, 0))] for r in rows], dtype="<f4")
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(rows)
    head += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    Path(path).write_bytes(head.encode() + data.tobytes())


class GpuCase(unittest.TestCase):
    """Temp dir + module lifecycle: every module created through new_module() is cleaned up."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.live = []
        self.addCleanup(lambda: [self.release(m) for m in list(self.live)])

    def new_module(self, n_pts=N, **over):
        if not (self.tmp / "ds").exists():
            _write_synthetic_colmap(self.tmp / "ds", n_pts=n_pts)
        ds = str(self.tmp / "ds")
        cfg = TrainerConfig(dataset_dir=ds, output_dir=str(self.tmp), image_dir=join_dir(ds, "images"),
                            sparse_dir=join_dir(ds, "sparse/0"), mask_dir="", output_ply=str(self.tmp / "o.ply"),
                            eval_interval=3, **over)
        m = vksplat.VkSplat()
        m.initialize(join_dir(str(ROOT), "shader"), -1)
        self.live.append(m)  # registered before set_train_config so a failing init still frees the device
        m.set_train_config(asdict(cfg))
        return m

    def release(self, m):
        self.live.remove(m)
        m.cleanup()


@needs_gpu
class DepthTest(GpuCase):
    def render(self, rows):
        write_custom_ply(self.tmp / "in.ply", rows)
        _write_synthetic_colmap(self.tmp / "ds")
        m = self.new_module(init_gaussians_ply=str(self.tmp / "in.ply"))
        m.render_val(0, False)
        rgb0 = np.array(m.pixel_state)
        m.render_val(0, True)
        st, pd, z = np.array(m.pixel_state), np.array(m.pixel_depth), np.array(m.depths)
        self.xy = np.array(m.xy_vs)
        return rgb0, st, pd, z

    def test_depth_flag_does_not_change_rgb(self):
        rows = [dict(xyz=(0, 0, 3), rgb=(1, 0, 0), scale=(.5, .5, .5), opacity=.9)]
        rgb0, st, _, _ = self.render(rows)
        np.testing.assert_array_equal(bits(rgb0), bits(st))

    def test_single_opaque_splat(self):
        rows = [dict(xyz=(0, 0, 3), rgb=(1, 0, 0), scale=(3, 3, 3), opacity=.999)]
        _, st, pd, z = self.render(rows)
        depth, alpha = depth_from_render(st, pd, 0.5)
        x, y = np.round(self.xy[0]).astype(int)
        c = depth[y - 2:y + 3, x - 2:x + 3]
        self.assertTrue(np.all(np.isfinite(c)))
        np.testing.assert_allclose(c, z[0, 0], rtol=1e-4)
        self.assertGreater(alpha[y, x], 0.9)

    def test_two_translucent_splats_weighted_mean(self):
        # w1 = a, w2 = (1-a)*a  =>  E[z] = (z1*a + z2*(1-a)*a) / (a + (1-a)*a)
        a = 0.5
        rows = [dict(xyz=(0, 0, 3), rgb=(1, 0, 0), scale=(100, 100, .1), opacity=a),
                dict(xyz=(0, 0, 5), rgb=(0, 1, 0), scale=(100, 100, .1), opacity=a)]
        _, st, pd, z = self.render(rows)
        depth, alpha = depth_from_render(st, pd, 0.5)
        j, i = np.round(self.xy[0]).astype(int)  # gaussians are ~flat (sigma >> image) around here
        near, far = sorted((float(z[0, 0]), float(z[1, 0])))
        self.assertAlmostEqual(float(alpha[i, j]), a + (1 - a) * a, places=2)
        self.assertAlmostEqual(float(depth[i, j]), (near * a + far * (1 - a) * a) / (a + (1 - a) * a), delta=0.02 * far)

    def test_low_alpha_is_nan_and_ray_range(self):
        rows = [dict(xyz=(0, 0, 3), rgb=(1, 0, 0), scale=(.2, .2, .2), opacity=.999)]
        _, st, pd, z = self.render(rows)
        depth, alpha = depth_from_render(st, pd, 0.5)
        self.assertTrue(np.all(np.isnan(depth[alpha < 0.5])))
        self.assertTrue(np.isnan(depth[0, 0]))
        r = z_to_ray_range(np.full((64, 64), 2.0, np.float32), 60, 60, 32, 32)
        self.assertAlmostEqual(float(r[32, 32]), 2.0 * np.sqrt(1 + 2 * (0.5 / 60) ** 2), places=5)
        self.assertGreater(float(r[0, 0]), 2.0)

    def test_train_step_leaves_depth_off(self):
        """train_step must not leave the depth pipeline selected: pixel_depth stays untouched after a depth render."""
        rows = [dict(xyz=(0, 0, 3), rgb=(1, 0, 0), scale=(3, 3, 3), opacity=.999)]
        _, _, pd0, _ = self.render(rows)
        m = self.live[0]
        self.assertGreater(np.abs(pd0).max(), 0)
        m._set_pixel_depth(np.full_like(pd0, -1.0))
        m.train_step(0, 1)
        np.testing.assert_array_equal(np.array(m.pixel_depth), -1.0)


@needs_gpu
class FrozenPrefixTest(GpuCase):
    F = 6 * SH
    # every densification path active: grow (tiny 2D-radius split criterion), prune, periodic opacity reset
    DENSIFY = dict(refine_start_iter=3, refine_every=3, grow_grad2d=1e-9, grow_scale2d=1e-4,
                   refine_scale2d_stop_iter=10**6, reset_every=10, prune_opa=0.02, pause_refine_after_reset=0,
                   refine_stop_iter=25)

    def run_steps(self, m, k, first=1):  # step 0 is degenerate for Adam bias correction (1 - beta^0 = 0)
        for s in range(first, first + k):
            m.train_step(s % m.num_train, s)

    def assert_prefix_identical(self, got, ref, F, msg=""):
        for k in PARAMS:
            np.testing.assert_array_equal(bits(got[k][:F]), bits(ref[k][:F]), err_msg=f"{msg} {k}")

    # ---- validation --------------------------------------------------------------------------------------
    def test_validation(self):
        cases = [
            (dict(freeze_prefix_count=SH + 1), "multiple of"),
            (dict(freeze_prefix_count=-SH), "non-negative multiple"),
            (dict(freeze_prefix_count=N + SH), "exceeds the number of initial splats"),
            (dict(freeze_prefix_count=SH, strategy="mcmc"), "requires the default strategy"),
            (dict(freeze_prefix_count=SH, selective_plane_freeze=True), "incompatible with selective_plane_freeze"),
        ]
        for over, msg in cases:
            with self.subTest(**over):
                with self.assertRaisesRegex(vksplat.RuntimeError, msg):  # C++ std::runtime_error
                    self.new_module(**over)

    # ---- freezing ----------------------------------------------------------------------------------------
    def test_frozen_bit_identical_and_active_changes(self):
        m = self.new_module(freeze_prefix_count=self.F, refine_start_iter=10**6)
        s0 = state(m)
        self.run_steps(m, 12)
        s1 = state(m)
        self.assert_prefix_identical(s1, s0, self.F)
        self.assertTrue(any(np.any(s1[k][self.F:] != s0[k][self.F:]) for k in PARAMS))

    def test_all_frozen_with_densify_and_reset(self):
        """F == N with densification, pruning and opacity reset all enabled: nothing may change at all."""
        m = self.new_module(freeze_prefix_count=N, **self.DENSIFY)
        s0 = state(m)
        self.run_steps(m, 45)
        s1 = state(m)
        self.assertEqual(len(s1["opacities"]), N)
        self.assert_prefix_identical(s1, s0, N)

    def test_raster_backward_pinned(self):
        m = self.new_module(freeze_prefix_count=self.F, refine_start_iter=10**6)
        self.run_steps(m, 2)
        self.assertEqual(m.get_raster_backward_impl(), "PerSplat")
        m = self.new_module(pin_raster_backward_per_splat=True, refine_start_iter=10**6)
        self.run_steps(m, 2)
        self.assertEqual(m.get_raster_backward_impl(), "PerSplat")

    # ---- equivalence of the active splats -----------------------------------------------------------------
    def compare_active(self, ply=None):
        """Run one step with and without the prefix skipped; return (unfrozen, frozen) modules.

        The reference uses the same per-splat raster-backward implementation (pin knob), so only the frozen-prefix
        masking differs. Raster-backward outputs are compared directly (<= 1e-5 of the max |gradient|, observed
        ~1e-8: float-atomic summation order)."""
        extra = dict(init_gaussians_ply=ply) if ply else {}
        a = self.new_module(pin_raster_backward_per_splat=True, refine_start_iter=10**6, **extra)
        b = self.new_module(freeze_prefix_count=self.F, refine_start_iter=10**6, **extra)
        a.train_step(0, 1)
        b.train_step(0, 1)
        for k in ("v_xy_vs", "v_inv_cov_vs_opacity", "v_rgb"):
            ga, gb = np.array(getattr(a, k)), np.array(getattr(b, k))
            self.assertGreater(np.abs(ga[self.F:]).max(), 0, k)
            rel = np.abs(ga[self.F:] - gb[self.F:]).max() / np.abs(ga[self.F:]).max()
            self.assertLessEqual(rel, 1e-5, k)
            self.assertEqual(float(np.abs(gb[:self.F]).max()), 0.0, k + ": frozen splats must get no gradient")
            print(f"\n[equiv] raster grad {k}: max rel diff {rel:.2g}")
        return a, b

    def test_active_update_matches_unfrozen_isotropic(self):
        """Synthetic COLMAP scene (isotropic, identity rotations): every rotation gradient is analytically zero, so
        the rotation update is pure rounding noise (Adam's first step is +-lr and its sign flips with summation
        order, also between two unmodified runs). Rotations are therefore only bounded here; the anisotropic test
        below covers rotation equivalence for real."""
        a, b = self.compare_active()
        sa, sb = state(a), state(b)
        for k in PARAMS:
            x, y = sa[k][self.F:], sb[k][self.F:]
            if k == "rotations":
                self.assertLessEqual(np.abs(x - y).max(), 2 * 0.001 + 1e-6)
                self.assertLess((np.abs(x - y) > 1e-7).mean(), 0.20)
            else:
                np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-7, err_msg=k)

    def test_active_update_matches_unfrozen_anisotropic(self):
        """Random rotations and anisotropic scales: rotation gradients are real. Compares the Adam first moment
        (g_rotations[:, 0]) of the active splats with rtol, and the resulting rotation update."""
        rng = np.random.default_rng(1)
        quats = rng.normal(size=(N, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        rows = [dict(xyz=(*rng.uniform(-1, 1, 2), rng.uniform(3, 5)), rgb=tuple(rng.uniform(-1, 1, 3)),
                     scale=tuple(np.exp(rng.uniform(np.log(.03), np.log(.4), 3))), opacity=rng.uniform(.3, .9),
                     quat=tuple(quats[i])) for i in range(N)]
        write_custom_ply(self.tmp / "aniso.ply", rows)
        _write_synthetic_colmap(self.tmp / "ds")
        a, b = self.compare_active(str(self.tmp / "aniso.ply"))
        ma, mb = (np.array(x.g_rotations)[self.F:, 0] for x in (a, b))
        self.assertGreater(np.abs(ma).max(), 1e-9)
        scale = np.abs(ma).max()
        np.testing.assert_allclose(mb, ma, rtol=1e-4, atol=1e-5 * scale)
        self.assertGreater(float((np.abs(ma) > 1e-3 * scale).mean()), 0.5)  # gradients are real, not rounding noise
        print(f"[equiv] rotation first moment: max |m| = {scale:.3g}, max |diff| / max |m| = {np.abs(ma - mb).max() / scale:.2g}")
        sa, sb = state(a), state(b)
        strong = np.abs(ma) > 1e-3 * scale  # Adam's first step is ~lr*sign(g): only well-conditioned elements
        diff = np.abs(sa["rotations"][self.F:] - sb["rotations"][self.F:])
        self.assertLess(diff[strong].max(), 1e-6)
        print(f"[equiv] rotations (strong grads): max abs diff {diff[strong].max():.2g}, "
              f"weak-gradient elements differing >1e-7: {int((diff[~strong] > 1e-7).sum())}/{int((~strong).sum())}")
        for k in PARAMS:
            if k != "rotations":
                np.testing.assert_allclose(sb[k][self.F:], sa[k][self.F:], rtol=1e-4, atol=1e-6, err_msg=k)

    # ---- densify / prune / reset / reload ----------------------------------------------------------------
    def prefix_ply(self):
        """Prefix rows chosen so that each densification path WOULD touch them without the pin:
        rows 0-2: opacity 0.001 (< prune_opa=0.02, pruned); row 3: huge scale (pruned after reset_every);
        rows 4-11: opacity 0.1 (an unpinned reset clamps to min(0.1, 2*0.02)=0.04); the rest: visible, opacity .5, so
        the 2D-radius criterion (grow_scale2d=1e-4) splits them. Active rows are ordinary visible splats."""
        rng = np.random.default_rng(2)
        rows = []
        for i in range(N):
            r = dict(xyz=(*rng.uniform(-1, 1, 2), rng.uniform(3, 5)), rgb=tuple(rng.uniform(-1, 1, 3)),
                     scale=(.05, .05, .05), opacity=.5)
            if i < 3:
                r["opacity"] = .001
            elif i == 3:
                r["scale"] = (5., 5., 5.)
            elif i < 12:
                r["opacity"] = .1
            rows.append(r)
        write_custom_ply(self.tmp / "prefix.ply", rows)
        _write_synthetic_colmap(self.tmp / "ds")
        return str(self.tmp / "prefix.ply")

    def test_control_densification_changes_unpinned_rows(self):
        """Non-vacuity control: the same PLY and schedule WITHOUT the freeze changes the rows that the frozen run keeps."""
        m = self.new_module(init_gaussians_ply=self.prefix_ply(), **self.DENSIFY)
        s0 = state(m)
        self.run_steps(m, 45)
        s1 = state(m)
        self.assertNotEqual(len(s1["opacities"]), N)
        n = min(len(s1["opacities"]), self.F)
        self.assertTrue(any(np.any(s1[k][:n] != s0[k][:n]) for k in PARAMS))
        self.assertLess(int(np.sum(np.isclose(s1["opacities"], 0.001, atol=1e-4))), 3, "0.001 rows should be pruned")

    def test_prefix_survives_densify_prune_reset_and_reload(self):
        ply = self.prefix_ply()
        m = self.new_module(freeze_prefix_count=self.F, init_gaussians_ply=ply, **self.DENSIFY)
        s0 = state(m)
        self.run_steps(m, 45)  # refines at 6,9,..,24; opacity resets at 10 and 20
        s1 = state(m)
        n1 = len(s1["opacities"])
        self.assertGreater(n1, N, "active splats did not grow: the grow path was not exercised")
        self.assert_prefix_identical(s1, s0, self.F, "after grow/prune/reset")
        self.assertTrue(np.all(s1["opacities"][4:12] == s0["opacities"][4:12]))  # not clamped to 0.04 by the reset
        self.assertTrue(np.all(s1["opacities"][:3] == s0["opacities"][:3]))  # 0.001 rows not pruned
        self.assertTrue(np.all(s1["scales"][3] == s0["scales"][3]))  # huge splat not pruned
        # checkpoint save / reload: write_ply -> init_gaussians_ply keeps the prefix in PLY space
        out = str(self.tmp / "ck.ply")
        m.write_ply(out)
        self.release(m)
        cont = dict(self.DENSIFY, refine_start_iter=99, refine_stop_iter=113)  # a second round of resets/refines
        m2 = self.new_module(freeze_prefix_count=self.F, init_gaussians_ply=out, **cont)
        s2 = state(m2)
        for k in PARAMS:
            np.testing.assert_allclose(s2[k][:self.F], s0[k][:self.F], rtol=1e-6, atol=1e-7, err_msg=k)
        self.run_steps(m2, 13, first=100)
        s3 = state(m2)
        self.assertNotEqual(len(s3["opacities"]), len(s2["opacities"]))
        self.assert_prefix_identical(s3, s2, self.F, "after reload")

    def test_morton_reorders_without_freeze(self):
        """Control: with every learning rate 0 and no freeze, only Morton re-sorting / densification / reset can change
        rows; they do (so the frozen test above would catch them leaking into the prefix)."""
        zero = {k: 0.0 for k in ("means_lr", "means_lr_final", "scales_lr", "opacities_lr", "quats_lr",
                                 "features_dc_lr", "features_rest_lr")}
        for k in zero:
            self.assertIn(k, TrainerConfig.__dataclass_fields__, k)
        # prune_scale*: the synthetic COLMAP seeds are wide; keep them from being pruned away entirely
        m = self.new_module(**zero, **dict(self.DENSIFY, prune_scale3d=1e3, prune_scale2d=1e3))
        s0 = state(m)
        self.run_steps(m, 45)
        s1 = state(m)
        n = min(len(s1["xyz_ws"]), self.F)
        self.assertTrue(any(np.any(s1[k][:n] != s0[k][:n]) for k in PARAMS))


class ConfigHashTest(unittest.TestCase):
    CFG = dict(dataset_dir="/data/ds", image_dir="/data/ds/images", sparse_dir="/data/ds/sparse/0", mask_dir="",
               output_dir="/o", output_ply="/o/x.ply", seed=7, checkpoint_steps=[1, 10, 100])

    def test_old_config_keeps_its_hash(self):
        """Golden digest computed with ckpt_trace.config_hash and TrainerConfig at commit dca7536 (before the
        freeze/depth/pin fields existed)."""
        from simple_trainer import TrainerConfig
        cfg = TrainerConfig(**self.CFG)
        self.assertEqual(config_hash(cfg), "d7d6781c620559d1edc263f82053bd1e99fb5814f22f983a5019e541305dd213")
        cfg.ckpt_depth, cfg.ckpt_depth_min_alpha = True, 0.9  # evaluation-only: excluded
        self.assertEqual(config_hash(cfg), "d7d6781c620559d1edc263f82053bd1e99fb5814f22f983a5019e541305dd213")

    def test_new_knobs_change_hash_when_set(self):
        from simple_trainer import TrainerConfig
        h0 = config_hash(TrainerConfig(**self.CFG))
        self.assertNotEqual(h0, config_hash(TrainerConfig(freeze_prefix_count=32, **self.CFG)))
        self.assertNotEqual(h0, config_hash(TrainerConfig(pin_raster_backward_per_splat=True, **self.CFG)))


SRC = ROOT / "src"


def _function_body(text, signature):
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    begin = i
    while True:
        depth += text[i] == "{"
        depth -= text[i] == "}"
        i += 1
        if depth == 0:
            return text[begin:i]


class HotLoopTest(unittest.TestCase):
    def test_loop_has_no_new_syncs(self):
        tree = ast.parse((ROOT / "simple_trainer.py").read_text())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        names = {n.attr for n in ast.walk(train) if isinstance(n, ast.Attribute)}
        self.assertNotIn("pixel_depth", names)
        self.assertNotIn("depths", names)
        loop = next(n for n in ast.walk(train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and n.target.id == "step")
        src = ast.unparse(loop)
        self.assertNotIn("freeze_prefix_count", src)
        self.assertNotIn("with_depth", src)


class NativeSourceGuardTest(unittest.TestCase):
    """Text-level guards on the C++/Slang sources (no GPU needed)."""

    def test_train_step_disables_depth(self):
        body = _function_body((SRC / "python_bindings.cpp").read_text(), "void train_step(")
        self.assertRegex(body, r"uniforms\.render_depth\s*=\s*0\s*;")

    def test_render_depth_is_scope_guarded(self):
        text = (SRC / "python_bindings.cpp").read_text()
        for fn in ("void render_train(", "void render_val("):
            self.assertIn("DepthFlagGuard", _function_body(text, fn), fn)
        # train_step reset + the guard's constructor and destructor: no other writer of render_depth
        self.assertEqual(len(re.findall(r"\.render_depth\s*=", text)), 3)

    def test_fill_buffer_only_in_pinned_blocks(self):
        body = _function_body((SRC / "gs_trainer.cpp").read_text(), "void VulkanGSTrainer::executeDefaultPostBackward(")
        fills = [m.start() for m in re.finditer(r"vkCmdFillBuffer", body)]
        self.assertEqual(len(fills), 3)  # dupli mask, split mask, keep mask
        for pos in fills:
            depth, enclosing = 0, []
            for i in range(pos - 1, -1, -1):  # headers of the enclosing blocks, innermost first
                depth += body[i] == "}"
                if body[i] == "{":
                    if depth == 0:
                        enclosing.append(body[body.rfind("\n", 0, i) + 1:i])
                    else:
                        depth -= 1
            self.assertTrue(any("pinned > 0" in h for h in enclosing), enclosing)

    def test_frozen_prefix_only_read_where_intended(self):
        def users(root, pattern):
            return {p.name for p in root.glob(pattern) if "frozen_prefix" in p.read_text()}
        self.assertEqual(users(SRC, "*.*pp") | users(SRC, "*.h"), {"gs_renderer.h", "gs_renderer.cpp", "python_bindings.cpp"})
        self.assertEqual(users(ROOT / "slang", "*.slang"), {"utils.slang", "alphablend_shader_bwd_per_splat.slang"})
        rend = (SRC / "gs_renderer.cpp").read_text()
        self.assertEqual(rend.count("frozen_prefix"), _function_body(
            rend, "void VulkanGSRenderer::executeRasterizeBackward(").count("frozen_prefix"))
        binds = (SRC / "python_bindings.cpp").read_text()
        self.assertEqual(binds.count("frozen_prefix"), _function_body(binds, "void train_step(").count("frozen_prefix"))

    def test_training_forward_shader_has_no_depth_code(self):
        """Depth lives in a separate compile-time variant (RENDER_DEPTH); every depth line is behind that macro."""
        src = (ROOT / "slang" / "alphablend_shader.slang").read_text()
        guarded, depth_lines = 0, []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("#if"):
                guarded += s.startswith("#if RENDER_DEPTH")
            elif s.startswith("#endif") and guarded:
                guarded -= 1
            elif re.search(r"depth", s) and not s.startswith("//") and not guarded:
                depth_lines.append(s)
        self.assertEqual(depth_lines, [])
        jobs = (ROOT.parent / "compile_shaders.py").read_text()
        self.assertIn('"rasterize_forward", {"EXPORT_MODE": forward}', jobs)
        self.assertIn('"rasterize_forward_depth", {"EXPORT_MODE": forward, "RENDER_DEPTH": 1}', jobs)


if __name__ == "__main__":
    unittest.main()
