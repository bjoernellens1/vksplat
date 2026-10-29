#!/usr/bin/env bash
# One command: resolve Pixi, build extension, run Vulkan smoke test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash sceneweave/install.sh
bash sceneweave/smoke.sh
