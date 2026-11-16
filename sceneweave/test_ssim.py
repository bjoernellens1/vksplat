"""SSIM + L1 pixel-loss gradient (two-stage kernels, ssim_every, ssim_downsample) against a float64 torch reference.

Every size runs in its own subprocess under a timeout, smallest first (a hung kernel must not take the suite
or the shared GPU down with it); the sizes include ones that are not multiples of the 16x16 tile.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

LAM = 0.2
TOL = 1e-4  # relative to the largest gradient entry


def _reference(render, gt, ds, use_ssim):
    import torch
    import torch.nn.functional as F

    r = torch.tensor(render, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(gt, dtype=torch.float64)
    p1 = torch.relu(r)
    loss = (1 - LAM) * (p1 - g).abs().mean()
    if use_ssim:
        x, y = p1.permute(2, 0, 1)[None], g.permute(2, 0, 1)[None]
        if ds > 1:
            gh, gw = x.shape[2] // ds, x.shape[3] // ds
            x = F.avg_pool2d(x[:, :, :gh * ds, :gw * ds], ds)
            y = F.avg_pool2d(y[:, :, :gh * ds, :gw * ds], ds)
        t = torch.arange(11, dtype=torch.float64) - 5
        k1 = torch.exp(-t ** 2 / (2 * 1.5 ** 2))
        k1 /= k1.sum()
        k = (k1[:, None] * k1[None, :])[None, None].repeat(3, 1, 1, 1)
        c = lambda a: F.conv2d(a, k, padding=5, groups=3)  # noqa: E731
        mu1, mu2 = c(x), c(y)
        s11, s22, s12 = c(x * x) - mu1 ** 2, c(y * y) - mu2 ** 2, c(x * y) - mu1 * mu2
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
        loss = loss + LAM * (1 - m[:, :, 5:-5, 5:-5].mean())
    loss.backward()
    return r.grad.numpy()


def _worker(w, h):
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        ds_dir = sc.build_dataset(Path(td) / "ds", n_images=3, n_points=300, w=w, h=h)
        for ds, every, step in [(1, 1, 0), (2, 1, 0), (1, 2, 1)]:
            mod = sc.new_trainer(ds_dir, Path(td) / f"o{ds}{every}", ssim_downsample=ds, ssim_every=every)
            mod.train_step(0, 0)  # a forward pass, so that the loss gradient has work to do
            gt = np.asarray(mod.get_train_image(0), dtype=np.float32)[..., :3] / 255.0
            render = rng.uniform(-0.1, 1.1, size=(h, w, 3)).astype(np.float32)
            mod._set_pixel_state(np.concatenate([render, np.ones((h, w, 1), np.float32)], axis=2).reshape(-1))
            mod.compute_pixel_loss_grad(0, step)
            got = np.asarray(mod.v_pixel_state, dtype=np.float32).reshape(h, w, 4)[..., :3]
            ref = _reference(render, gt, ds, step % every == 0)
            err = np.abs(got - ref).max() / np.abs(ref).max()
            assert err < TOL, f"{w}x{h} ds={ds} every={every}: relative error {err:.2e}"


@pytest.mark.parametrize("size", [(37, 23), (641, 479), (1200, 680)])
def test_ssim_gradient_matches_reference(size):
    pytest.importorskip("torch")
    proc = subprocess.run([sys.executable, __file__, str(size[0]), str(size[1])], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]


@pytest.mark.parametrize("options", [{}, {"ssim_every": 2}, {"ssim_downsample": 2}, {"ssim_every": 3, "ssim_downsample": 2}])
def test_training_with_ssim_options_runs(options):
    with tempfile.TemporaryDirectory() as td:
        ds_dir = sc.build_dataset(Path(td) / "ds", n_images=3, n_points=500, w=96, h=80)
        mod = sc.new_trainer(ds_dir, Path(td) / "o", **options)
        n = int(mod.num_train)
        for step in range(12):
            mod.train_step(step % n, step)
        assert np.isfinite(np.asarray(mod.xyz_ws)).all()
        assert np.isfinite(np.asarray(mod.opacities)).all()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]))
