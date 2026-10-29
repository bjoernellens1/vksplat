"""CPU checks for P1c controlled initialization manifest interpretation."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_baseline import controlled_reference_count


class ControlledInitTest(unittest.TestCase):
    def test_original_measured_dataset(self):
        data={"num_seed_points":34200}
        self.assertEqual(controlled_reference_count(data),34200)

    def test_completion_prefix_identical_for_both_variants(self):
        oracle={"num_seed_points":64200,
                "p1c":{"condition":"oracle_completion","original_seed_count":34200}}
        control={"num_seed_points":64200,
                 "p1c":{"condition":"capacity_control","original_seed_count":34200}}
        self.assertEqual(controlled_reference_count(oracle),34200)
        self.assertEqual(controlled_reference_count(control),34200)

    def test_reject_invalid_reference_counts(self):
        for bad in (0,3,64201,-1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    controlled_reference_count({"num_seed_points":64200,
                        "p1c":{"original_seed_count":bad}})


if __name__=="__main__":
    unittest.main()
