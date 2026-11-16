"""Guard against running with SPIR-V that is older than the Slang/GLSL sources.

compile_shaders.py writes ``shader/manifest.json`` (sha256 of every shader source and of every generated
.spv, grouped per compile unit). ``check`` recomputes the hashes and reports every group whose sources or
outputs differ. The extension calls it from ``VkSplat.initialize``.

Set VKSPLAT_ALLOW_STALE_SHADERS=1 to downgrade the error to a warning.
"""

import hashlib
import json
import os
import warnings
from pathlib import Path

MANIFEST_NAME = "manifest.json"
ALLOW_ENV = "VKSPLAT_ALLOW_STALE_SHADERS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path, groups: dict) -> dict:
    """groups: {group_name: (source_paths, output_paths)}; all paths are hashed and stored relative to root."""
    root = Path(root)

    def hashed(paths):
        return {os.path.relpath(p, root): _sha256(Path(p)) for p in sorted(map(str, paths))}

    return {"version": 1, "groups": {name: {"sources": hashed(srcs), "outputs": hashed(outs)}
                                     for name, (srcs, outs) in groups.items()}}


def stale_groups(shader_dir) -> list:
    """Return human-readable reasons for every stale compile unit (empty list: fresh)."""
    shader_dir = Path(shader_dir)
    root = shader_dir.parent
    manifest_path = shader_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{manifest_path} missing: run compile_shaders.py"]
    groups = json.loads(manifest_path.read_text())["groups"]
    stale = []
    for name, entry in groups.items():
        bad = []
        for kind in ("sources", "outputs"):
            for rel, digest in entry[kind].items():
                path = root / rel
                if not path.is_file():
                    bad.append(f"{rel} (missing)")
                elif _sha256(path) != digest:
                    bad.append(f"{rel} ({'changed since compile' if kind == 'sources' else 'differs from compiled output'})")
        if bad:
            stale.append(f"{name}: " + ", ".join(bad))
    return stale


def check(shader_dir=None, strict: bool = True) -> list:
    """Raise RuntimeError naming the stale shaders (warn if strict=False or VKSPLAT_ALLOW_STALE_SHADERS=1)."""
    if shader_dir is None:
        shader_dir = Path(__file__).resolve().parent / "shader"
    stale = stale_groups(shader_dir)
    if stale:
        msg = ("VkSplat shaders are stale; rerun compile_shaders.py (then rebuild if C++ changed):\n  "
               + "\n  ".join(stale))
        if strict and os.environ.get(ALLOW_ENV) != "1":
            raise RuntimeError(msg + f"\nSet {ALLOW_ENV}=1 to run anyway.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return stale
