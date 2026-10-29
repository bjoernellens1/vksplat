"""CPU integration test: matched geometry-completion versus capacity control."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from oracle_completion import read_seeds, validate_pair


class CompletionTest(unittest.TestCase):
    def test_paired_datasets_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "measured"
            sparse = base / "sparse" / "0"
            sparse.mkdir(parents=True)
            (base / "images").mkdir()
            (sparse / "cameras.txt").write_text("1 PINHOLE 4 4 2 2 1.5 1.5\n")
            (sparse / "images.txt").write_text("# empty test camera list\n")
            with (sparse / "points3D.txt").open("w") as f:
                for i in range(20):
                    f.write(f"{i+1} {i*0.001} 0 0 100 110 120 0\n")
            (base / "sceneweave_conversion.json").write_text(
                json.dumps({"schema_version": "1.0.0", "mode": "measured",
                            "num_seed_points": 20, "frame_indices": [],
                            "seed_frame_indices": []}))
            mesh = root / "box.ply"
            trimesh.creation.box(extents=(2, 2, 2)).export(mesh)
            out = root / "completion"
            cmd = [sys.executable, str(Path(__file__).with_name("oracle_completion.py")),
                   "--baseline", str(base), "--mesh", str(mesh), "--output", str(out),
                   "--add-points", "12", "--sample-multiplier", "30",
                   "--min-distance", "0.08", "--seed", "19"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            oracle = out / "oracle_completion"
            control = out / "capacity_control"
            check = validate_pair(oracle, control, 20, 12)
            self.assertEqual(check["final_seed_count"], 32)
            a, _ = read_seeds(oracle / "sparse" / "0" / "points3D.txt")
            b, _ = read_seeds(control / "sparse" / "0" / "points3D.txt")
            self.assertFalse(np.allclose(a[20:], b[20:]))
            self.assertTrue((oracle / "images").is_symlink())
            om = json.loads((oracle / "sceneweave_conversion.json").read_text())
            cm = json.loads((control / "sceneweave_conversion.json").read_text())
            self.assertTrue(om["oracle_privileged"])
            self.assertFalse(cm["oracle_privileged"])
            self.assertEqual(om["p1c"]["version"], "1.0.0")


if __name__ == "__main__":
    unittest.main()
