import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "vksplat"
sys.path.insert(0, str(ROOT))
import shader_freshness  # noqa: E402


@pytest.fixture()
def shader_tree(tmp_path):
    shutil.copytree(ROOT / "slang", tmp_path / "slang")
    shutil.copytree(ROOT / "shader", tmp_path / "shader")
    return tmp_path


def test_committed_shaders_are_fresh():
    assert shader_freshness.check(ROOT / "shader") == []


def test_edited_source_is_reported_and_raises(shader_tree):
    with (shader_tree / "slang" / "utils.slang").open("a") as f:
        f.write("\n// edit\n")
    with pytest.raises(RuntimeError, match="utils.slang"):
        shader_freshness.check(shader_tree / "shader")
    stale = shader_freshness.stale_groups(shader_tree / "shader")
    assert any(g.startswith("alphablend_shader.slang") for g in stale)
    assert not any(g.startswith("sum.slang") for g in stale)


def test_modified_spv_and_missing_manifest(shader_tree):
    (shader_tree / "shader" / "generated" / "sum.spv").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="sum.spv"):
        shader_freshness.check(shader_tree / "shader")
    (shader_tree / "shader" / shader_freshness.MANIFEST_NAME).unlink()
    with pytest.raises(RuntimeError, match="manifest.json missing"):
        shader_freshness.check(shader_tree / "shader")


def test_env_override_warns(shader_tree, monkeypatch):
    (shader_tree / "slang" / "sum.slang").write_text("// stale\n")
    monkeypatch.setenv(shader_freshness.ALLOW_ENV, "1")
    with pytest.warns(RuntimeWarning, match="sum.slang"):
        assert shader_freshness.check(shader_tree / "shader")
    monkeypatch.delenv(shader_freshness.ALLOW_ENV)
    with pytest.warns(RuntimeWarning):
        shader_freshness.check(shader_tree / "shader", strict=False)
