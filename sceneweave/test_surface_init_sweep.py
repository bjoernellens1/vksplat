"""CPU contract for the Sceneweave measured-surface initializer sweep."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from surface_init_sweep import make_command

class SurfaceInitCommandTest(unittest.TestCase):
    def test_same_geometry_and_training_policy(self):
        cmds={mode:make_command(Path("/tmp/data"),Path("/tmp/out"),
                 mode=mode,steps=10000,seed=42,ratio=.25)
              for mode in ("isotropic","random_anisotropic","aligned")}
        def flag(command,name):return command[command.index(name)+1]
        for mode,command in cmds.items():
            self.assertEqual(flag(command,"--dataset"),"/tmp/data")
            self.assertEqual(flag(command,"--surface-mode"),mode)
            self.assertEqual(flag(command,"--normal-scale-ratio"),"0.25")
            for option,value in (
                ("--shuffle-seed","42"),("--refine-start","500"),
                ("--refine-stop","4000"),("--refine-every","300"),
                ("--grow-grad2d","0.0004"),("--max-steps","30000"),
                ("--mask-dir","masks")):
                self.assertEqual(flag(command,option),value)

    def test_invalid_modes_rejected(self):
        with self.assertRaises(ValueError):
            make_command(Path("/tmp/data"),Path("/tmp/out"),
                         mode="hallucinated",steps=10000,seed=42,ratio=.25)

if __name__=="__main__":
    unittest.main()
