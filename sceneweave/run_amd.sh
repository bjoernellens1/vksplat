#!/usr/bin/env bash
# Portable Linux Pixi launcher for VkSplat; keeps system Vulkan ICD/DRM libraries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${VK_SPLAT_PYTHON:-$ROOT/.pixi/envs/default/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing Python at $PYTHON; run 'pixi install'." >&2; exit 2; }

# libdrm from conda-forge may be older than the Linux host's AMD graphics stack.
# Resolve SONAMEs via the host dynamic-loader cache, not Fedora-only /usr/lib64.
find_system_lib() {
  local name="$1"
  local tool
  tool="$(command -v ldconfig || true)"
  if [[ -z "$tool" ]]; then
    for candidate in /usr/sbin/ldconfig /sbin/ldconfig; do
      if [[ -x "$candidate" ]]; then tool="$candidate"; break; fi
    done
  fi
  [[ -n "$tool" ]] || return 1
  "$tool" -p 2>/dev/null | awk -v n="$name" '$1 == n && $0 ~ /x86-64/ {print $NF; exit}'
}
SYSTEM_DRM="${VK_SPLAT_SYSTEM_DRM:-auto}" # 0 disable, 1 require, auto if both libs available
if [[ "$SYSTEM_DRM" != "0" ]]; then
  DRM="$(find_system_lib libdrm.so.2 || true)"
  AMDGPU="$(find_system_lib libdrm_amdgpu.so.1 || true)"
  if [[ -f "$DRM" && -f "$AMDGPU" ]]; then
    export LD_PRELOAD="$DRM $AMDGPU${LD_PRELOAD:+ $LD_PRELOAD}"
  elif [[ "$SYSTEM_DRM" == "1" ]]; then
    echo "System libdrm libraries not found (ldconfig); install libdrm2 and libdrm-amdgpu1." >&2
    exit 2
  fi
fi

# Don't unconditionally force an AMD ICD on Intel/NVIDIA hardware.
# Set VK_SPLAT_FORCE_RADV=1 to use the AMD RADV ICD explicitly.
FORCE_RADV="${VK_SPLAT_FORCE_RADV:-0}"
if [[ "$FORCE_RADV" == "1" ]]; then
  ICD="${VK_SPLAT_RADV_ICD:-/usr/share/vulkan/icd.d/radeon_icd.x86_64.json}"
  [[ -f "$ICD" ]] || { echo "RADV ICD not found at $ICD" >&2; exit 2; }
  export VK_DRIVER_FILES="$ICD"
fi

# Import compiled 'vksplat' extension rather than outer 'vksplat/' namespace.
cd "$ROOT/vksplat"
exec "$PYTHON" "$@"
