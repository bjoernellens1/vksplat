"""CPU-only checks for controlled strict-growth prior benchmark command."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prior_strict_sweep import command

class StrictPriorCommandTests(unittest.TestCase):
    def test_matches_original_strict_growth_policy(self):
        cmd=command(Path("/tmp/baseline"),Path("/tmp/results"),steps=10000,seed=42)
        def flag(name):
            return cmd[cmd.index(name)+1]
        self.assertIn("--controlled-init",cmd)
        self.assertEqual(flag("--steps"),"10000")
        self.assertEqual(flag("--mask-dir"),"masks")
        self.assertEqual(flag("--refine-start"),"500")
        self.assertEqual(flag("--refine-stop"),"4000")
        self.assertEqual(flag("--refine-every"),"300")
        self.assertEqual(flag("--grow-grad2d"),"0.0004")
        self.assertEqual(flag("--shuffle-seed"),"42")

if __name__=="__main__":
    unittest.main()
