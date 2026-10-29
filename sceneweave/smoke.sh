#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The launcher's cwd fixes the namespace-package shadowing issue.
# Users on non-Fedora hosts may opt out: VK_SPLAT_SYSTEM_DRM=0 VK_SPLAT_FORCE_RADV=0.
bash "$ROOT/sceneweave/run_amd.sh" - <<'PY'
import vksplat
from pathlib import Path
print("Extension:", getattr(vksplat, "__file__", None))
if not hasattr(vksplat, "VkSplat"):
    raise RuntimeError("Compiled VkSplat extension not imported")
engine = vksplat.VkSplat()
engine.initialize(str(Path("shader").resolve()) + "/", -1)
print("SUCCESS: VkSplat GPU pipeline initialized")
PY
