#!/usr/bin/env bash
# Reproducible Linux/Pixi VkSplat extension setup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" != "--skip-pixi-install" ]]; then
    command -v pixi >/dev/null || { echo "Pixi not found: https://pixi.sh" >&2; exit 2; }
    pixi install
fi
PYTHON="$ROOT/.pixi/envs/default/bin/python"
[[ -x "$PYTHON" ]] || { echo "Pixi Python not found; run 'pixi install' first" >&2; exit 2; }
[[ -f /usr/include/vulkan/vulkan.h ]] || { echo "Missing Vulkan headers; install system vulkan-headers/vulkan-loader-devel" >&2; exit 2; }
[[ -f /usr/include/glm/glm.hpp ]] || { echo "Missing GLM headers; install system glm-devel" >&2; exit 2; }
echo "Building editable VkSplat extension with $PYTHON"
"$PYTHON" -m pip install -e ./vksplat --no-build-isolation --no-deps -v
echo "Build installed. Run: pixi run smoke"
