#!/usr/bin/env bash
# Fedora/Pixi RADV troubleshooting launcher; do not assume this is universal.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${VK_SPLAT_PYTHON:-$ROOT/.pixi/envs/default/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Pixi Python: $PYTHON" >&2
  exit 2
fi
# Pin system DRM libs to prevent older conda-forge libdrm from shadowing Fedora RADV.
if [[ "${VK_SPLAT_SYSTEM_DRM:-1}" == "1" ]]; then
  for lib in /usr/lib64/libdrm.so.2 /usr/lib64/libdrm_amdgpu.so.1; do
    [[ -f "$lib" ]] || { echo "Missing Fedora library $lib" >&2; exit 2; }
  done
  export LD_PRELOAD="/usr/lib64/libdrm.so.2 /usr/lib64/libdrm_amdgpu.so.1${LD_PRELOAD:+ $LD_PRELOAD}"
fi
if [[ "${VK_SPLAT_FORCE_RADV:-1}" == "1" ]]; then
  ICD="/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
  [[ -f "$ICD" ]] || { echo "Missing RADV manifest $ICD" >&2; exit 2; }
  export VK_DRIVER_FILES="$ICD"
fi
# Essential: running from outer repository makes 'vksplat/' a namespace package.
cd "$ROOT/vksplat"
exec "$PYTHON" "$@"
