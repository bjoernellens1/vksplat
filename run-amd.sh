#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
export LD_PRELOAD="/usr/lib64/libdrm.so.2 /usr/lib64/libdrm_amdgpu.so.1${LD_PRELOAD:+ $LD_PRELOAD}"

cd "$ROOT/vksplat"

exec "$ROOT/.pixi/envs/default/bin/python" "$@"
