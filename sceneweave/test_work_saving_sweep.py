"""Unit tests for VkSplat work-saving sweep and geometry-freeze command construction."""
import json
from pathlib import Path
import tempfile
from work_saving_sweep import make_command, resolve_freeze_step


def test_make_command_and_resolve_freeze_step():
    with tempfile.TemporaryDirectory() as tmp:
        dset = Path(tmp) / "ws_dset"
        dset.mkdir()
        (dset / "sceneweave_work_saving.json").write_text(
            json.dumps({
                "schema_version": "sceneweave-vksplat-work-saving-v1",
                "recommended_freeze_after_ratio": 0.30,
                "total_prior_overhead_s": 0.125,
            })
        )
        step, ovh, treat_steps = resolve_freeze_step(dset, steps=10000)
        assert step == 3000
        assert treat_steps == 3450
        assert abs(ovh - 0.125) < 1e-9

        cmd = make_command(
            dset,
            Path(tmp) / "out",
            steps=treat_steps,
            seed=42,
            freeze_after_iter=step,
            surface_mode="isotropic",
        )
        assert "--freeze-geometry-after" in cmd
        idx = cmd.index("--freeze-geometry-after")
        assert cmd[idx + 1] == "3000"
        assert cmd[cmd.index("--surface-mode") + 1] == "isotropic"


if __name__ == "__main__":
    test_make_command_and_resolve_freeze_step()
    print("work_saving_sweep tests passed.")
