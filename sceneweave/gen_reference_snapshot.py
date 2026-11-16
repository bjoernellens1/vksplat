"""Writes the dense-training reference snapshot used by test_sparse_backward.py.

    VKSPLAT_PY_DIR=<vksplat dir of the 4eda56f build> python sceneweave/gen_reference_snapshot.py sceneweave/reference_4eda56f.npz
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import append_scenario as sc

STEPS, N_POINTS = 12, 300


def dense_snapshot() -> dict:
    with tempfile.TemporaryDirectory() as td:
        ds = sc.build_dataset(Path(td) / "ds", n_points=N_POINTS)
        mod = sc.new_trainer(ds, Path(td) / "out")
        n = int(mod.num_train)
        for step in range(STEPS):
            mod.train_step(step % n, step)
        return {k: v for k, v in sc.snapshot(mod).items() if k != "gids"}


if __name__ == "__main__":
    np.savez_compressed(sys.argv[1], **dense_snapshot())
