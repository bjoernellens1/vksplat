#!/usr/bin/env bash
# Pixi-managed Python build with host Vulkan/GLM SDK. Works on Fedora/Ubuntu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" != "--skip-pixi-install" ]]; then
  command -v pixi >/dev/null || { echo "Install Pixi first: https://pixi.sh" >&2; exit 2; }
  pixi install
fi
PYTHON="$ROOT/.pixi/envs/default/bin/python"
[[ -x "$PYTHON" ]] || { echo "Pixi Python missing; run pixi install" >&2; exit 2; }

install_hint() {
  if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    case "${ID:-}" in
      ubuntu|debian)
        echo "Install prerequisites: sudo apt update && sudo apt install -y build-essential libvulkan-dev libglm-dev pkg-config" >&2 ;;
      fedora)
        echo "Install prerequisites: sudo dnf install -y gcc-c++ vulkan-loader-devel vulkan-headers glm-devel pkgconf-pkg-config" >&2 ;;
      *)
        echo "Install system Vulkan loader/development headers, GLM headers, C++17 compiler and pkg-config." >&2 ;;
    esac
  fi
}
if [[ ! -f /usr/include/vulkan/vulkan.h || ! -f /usr/include/glm/glm.hpp ]]; then
  echo "Vulkan and/or GLM development headers missing." >&2
  install_hint
  exit 2
fi
echo "Building editable VkSplat extension with $PYTHON"
"$PYTHON" -m pip install -e ./vksplat --no-build-isolation --no-deps -v
echo "Build installed. Next: pixi run smoke"
