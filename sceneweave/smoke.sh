#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Perform environment diagnostics before probing the compiled extension.
# Neither a Python import nor an llvmpipe device means GPU training is available.
if command -v vulkaninfo >/dev/null 2>&1; then
  echo "=== System Vulkan devices (may contain software renderers) ==="
  vulkaninfo --summary 2>&1 | grep -E 'deviceName|driverName|driverInfo' || true
else
  echo "WARNING: vulkaninfo missing. Install vulkan-tools." >&2
fi

echo "=== VkSplat Vulkan initialization ==="
if ! bash "$ROOT/sceneweave/run_amd.sh" - <<'PY'
import vksplat
from pathlib import Path

print("Extension:", getattr(vksplat, "__file__", None), flush=True)
if not hasattr(vksplat, "VkSplat"):
    raise RuntimeError("Compiled extension not imported; run pixi run build")
engine = vksplat.VkSplat()
engine.initialize(str(Path("shader").resolve()) + "/", -1)
print("SUCCESS: VkSplat Vulkan pipeline initialized", flush=True)
PY
then
  cat >&2 <<'EOF'

VkSplat could not initialize a viable hardware Vulkan device.
If only llvmpipe appears:
  1. Run: vulkaninfo --summary
  2. Check: ls -l /dev/dri/renderD*
  3. Check: ls /usr/share/vulkan/icd.d/
  4. For AMD RADV, try:
     VK_SPLAT_FORCE_RADV=1 pixi run smoke
  5. If the SYSTEM detects AMD but VkSplat does not, inspect the Pixi
     loader/DRM libraries. Do not modify or remove system drivers blindly.
  6. If the SYSTEM also only detects llvmpipe, repair the host GPU driver
     and device permissions first. VkSplat cannot solve this inside Pixi.
EOF
  exit 1
fi
