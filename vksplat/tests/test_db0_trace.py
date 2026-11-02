"""DB0 tests: checkpoint schedule, hot-loop AST guard, PLY init roundtrip.

Run: python -m unittest discover -s vksplat/tests   (PLY roundtrip needs the built extension + a Vulkan device)
"""
import ast
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ckpt_trace import log_spaced_steps  # noqa: E402

FORBIDDEN_CALLS = {
    "render_val", "render_train", "pixel_state", "write_ply", "save", "savez", "np.save", "open",
    "item", "synchronize", "get_timing_breakdown", "get_peak_vram_usage", "write_trace", "checkpoint",
}


def _train_loop():
    tree = ast.parse((ROOT / "simple_trainer.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    return next(n for n in ast.walk(train) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "step")


def _is_ckpt_branch(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "step"
            and isinstance(node.test.ops[0], ast.In)
            and isinstance(node.test.comparators[0], ast.Name) and node.test.comparators[0].id == "ckpt_set")


def _unguarded_names(node, out):
    for child in ast.iter_child_nodes(node):
        if _is_ckpt_branch(child):
            continue
        if isinstance(child, ast.Attribute):
            out.append(child.attr)
        elif isinstance(child, ast.Name):
            out.append(child.id)
        _unguarded_names(child, out)
    return out


class ScheduleTest(unittest.TestCase):
    def test_log_spaced(self):
        s = log_spaced_steps(30000, 25)
        self.assertEqual(s, sorted(set(s)))
        self.assertEqual((s[0], s[-1]), (1, 30000))
        self.assertTrue(20 <= len(s) <= 25)
        ratios = np.diff(np.log(s))
        self.assertTrue(np.all(ratios > 0))

    def test_small_and_degenerate(self):
        self.assertEqual(log_spaced_steps(1, 25), [1])
        self.assertEqual(log_spaced_steps(10, 25), list(range(1, 11)))
        self.assertEqual(log_spaced_steps(300, 25)[-1], 300)
        self.assertEqual(log_spaced_steps(0), [])


class HotLoopGuardTest(unittest.TestCase):
    def test_loop_has_guarded_branch(self):
        self.assertTrue(any(_is_ckpt_branch(n) for n in ast.walk(_train_loop())))

    def test_no_eval_or_io_outside_ckpt_branch(self):
        names = set(_unguarded_names(_train_loop(), []))
        self.assertFalse(names & FORBIDDEN_CALLS, names & FORBIDDEN_CALLS)

    def test_guard_detects_violation(self):
        loop = ast.parse("for step in r:\n    module.train_step(i, step)\n    x = module.pixel_state\n").body[0]
        self.assertIn("pixel_state", _unguarded_names(loop, []))


def _write_synthetic_colmap(root, n_img=10, n_pts=300, size=64):
    rng = np.random.default_rng(0)
    (root / "images").mkdir(parents=True)
    (root / "sparse/0").mkdir(parents=True)
    (root / "sparse/0/cameras.txt").write_text(f"1 PINHOLE {size} {size} 60 60 {size/2} {size/2}\n")
    lines = []
    for i in range(n_img):
        name = f"img_{i:03d}.png"
        cv2.imwrite(str(root / "images" / name), rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        a = 2 * np.pi * i / n_img
        # identity rotation, camera slides on a circle (w2c translation); points lie in front.
        lines += [f"{i+1} 1 0 0 0 {np.cos(a)*0.3:.4f} {np.sin(a)*0.3:.4f} 0 1 {name}\n", "\n"]
    (root / "sparse/0/images.txt").write_text("".join(lines))
    pts = rng.uniform(-1, 1, (n_pts, 3)) + [0, 0, 4]
    rgb = rng.integers(0, 255, (n_pts, 3))
    (root / "sparse/0/points3D.txt").write_text("".join(
        f"{j+1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0.1\n" for j, (p, c) in enumerate(zip(pts, rgb))))


def _read_ply(path):
    raw = Path(path).read_bytes()
    head, _, body = raw.partition(b"end_header\n")
    names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
    return names, np.frombuffer(body, dtype="<f4").reshape(-1, len(names))


class PlyRoundtripTest(unittest.TestCase):
    def test_init_from_ply_reproduces_input(self):
        try:
            import vksplat
            from simple_trainer import TrainerConfig, join_dir
            probe = vksplat.VkSplat()
            probe.initialize(join_dir(str(ROOT), "shader"), -1)
        except Exception as e:  # no extension / no Vulkan device
            self.skipTest(f"vksplat/Vulkan unavailable: {e}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_synthetic_colmap(tmp / "ds")

            def init_and_write(init_ply, out):
                cfg = TrainerConfig(dataset_dir=str(tmp / "ds"), output_dir=str(tmp), init_gaussians_ply=init_ply,
                                    image_dir=join_dir(str(tmp / "ds"), "images"), sparse_dir=join_dir(str(tmp / "ds"), "sparse/0"),
                                    mask_dir="", output_ply=str(out))
                m = vksplat.VkSplat()
                m.initialize(join_dir(str(ROOT), "shader"), -1)
                m.set_train_config(asdict(cfg))
                m.write_ply(str(out))
                m.cleanup()

            init_and_write("", tmp / "a.ply")
            # Distinguish PLY init from points3D seeding: fewer Gaussians, altered values.
            names_a, a = _read_ply(tmp / "a.ply")
            a = a[:100] * np.float32(0.9)
            head = (tmp / "a.ply").read_bytes().partition(b"end_header\n")[0].decode().replace("vertex 300", "vertex 100")
            (tmp / "a.ply").write_bytes(head.encode() + b"end_header\n" + a.astype("<f4").tobytes())
            init_and_write(str(tmp / "a.ply"), tmp / "b.ply")
            names_b, b = _read_ply(tmp / "b.ply")
            self.assertEqual(names_a, names_b)
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
