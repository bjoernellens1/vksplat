"""Continuous quality-vs-time trace support for a single training run (DB0).

Everything here runs only at checkpoint steps (or once at start/end); the hot
training loop in simple_trainer.train() never calls into this module otherwise.
Metrics are NOT computed in-process: renders and Gaussian state are dumped and
scored offline.
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import threading
from dataclasses import asdict
from time import perf_counter

import numpy as np

# Config keys that do not affect the experiment (volatile paths / bookkeeping).
_HASH_EXCLUDE = ("output_dir", "output_ply", "ckpt_dir", "gates_json")


def log_spaced_steps(total_steps, n=25):
    """~n unique integer step counts, log-spaced in [1, total_steps], always ending at total_steps."""
    if total_steps < 1:
        return []
    raw = np.unique(np.round(np.geomspace(1, total_steps, n)).astype(int))
    return sorted({int(s) for s in raw} | {int(total_steps)})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg):
    d = {k: v for k, v in asdict(cfg).items() if k not in _HASH_EXCLUDE}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def dataset_hash(cfg):
    """sha256 over sorted image file names+sizes and the content of the COLMAP model files (no mtimes)."""
    h = hashlib.sha256()
    image_dir = cfg.image_dir
    for name in sorted(os.listdir(image_dir)):
        h.update(f"{name}:{os.path.getsize(os.path.join(image_dir, name))}\n".encode())
    for stem in ("cameras", "images", "points3D"):
        for ext in ("bin", "txt"):
            p = os.path.join(cfg.sparse_dir, f"{stem}.{ext}")
            if os.path.isfile(p):
                h.update(f"{stem}.{ext}:{sha256_file(p)}\n".encode())
    return h.hexdigest()


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _vulkan_devices():
    """Best-effort (name, driverInfo) list from `vulkaninfo --summary`."""
    out = _run(["vulkaninfo", "--summary"]) or ""
    devs, cur = [], None
    for line in out.splitlines():
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key == "deviceName":
            cur = {"name": val}
            devs.append(cur)
        elif key == "driverInfo" and cur is not None:
            cur["driver_info"] = val
    return devs


def build_manifest(cfg, gates_json=None):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    devs = _vulkan_devices()
    hw = [d for d in devs if "llvmpipe" not in d["name"]] or devs
    return {
        "config_sha256": config_hash(cfg),
        "vksplat_git_rev": _run(["git", "rev-parse", "HEAD"], cwd=src_dir),
        "vksplat_git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=src_dir)),
        "dataset_sha256": dataset_hash(cfg),
        "device_name": hw[0]["name"] if hw else None,
        "vulkan_devices": devs,
        "kernel": platform.release(),
        "amdgpu_module_version": _read("/sys/module/amdgpu/version"),
        "driver": hw[0].get("driver_info") if hw else None,
        "gates_sha256": sha256_file(gates_json) if gates_json else None,
    }


class HwmonSampler:
    """Background thread sampling amdgpu hwmon power (W) and temperature (C); tolerant of missing files."""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = perf_counter()
        pat = "/sys/class/drm/card*/device/hwmon/*/"
        self._dirs = [d for d in sorted(glob.glob(pat)) if _read(d + "name") == "amdgpu"]

    @staticmethod
    def _val(path, scale):
        v = _read(path)
        try:
            return float(v) / scale
        except (TypeError, ValueError):
            return None

    def _run(self):
        while not self._stop.is_set():
            for d in self._dirs:
                power = self._val(d + "power1_average", 1e6)
                if power is None:
                    power = self._val(d + "power1_input", 1e6)
                self.samples.append({
                    "t_s": perf_counter() - self._t0, "hwmon": d,
                    "power_w": power, "temp_c": self._val(d + "temp1_input", 1e3),
                })
            self._stop.wait(self.interval_s)

    def start(self):
        if self._dirs:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return self.samples


def _gpu_seconds(breakdown):
    return {k: float(dt) for k, (n, dt) in breakdown.items()}


class CheckpointRecorder:
    """Pauses the train clock at checkpoint steps, dumps val renders + Gaussian state, writes trace.json."""

    def __init__(self, module, cfg, ckpt_dir, t_start, prep_s, lock, sampler):
        self.module, self.cfg, self.ckpt_dir = module, cfg, ckpt_dir
        self.t_start, self.prep_s = t_start, prep_s
        self.lock, self.sampler = lock, sampler
        self.eval_wall_s = 0.0
        self.eval_gpu_s = {}
        self.checkpoints = []
        os.makedirs(ckpt_dir, exist_ok=True)

    def checkpoint(self, step):
        """Called after loop index `step` finished, i.e. step + 1 optimizer steps are done."""
        n_done, m, dtype = step + 1, self.module, np.dtype(self.cfg.ckpt_render_dtype)
        with self.lock:
            # get_timing_breakdown() follows blocking queue submissions: all training GPU work is complete.
            before = _gpu_seconds(m.get_timing_breakdown())
            t_pause = perf_counter()
            train_only_s = t_pause - self.t_start - self.eval_wall_s

            tag = f"ckpt_{n_done:06d}"
            renders = {}
            for i in range(m.num_val):
                m.render_val(i)
                renders[f"val_{i:05d}"] = np.ascontiguousarray(m.pixel_state[:, :, :3], dtype=dtype)
            renders_path = os.path.join(self.ckpt_dir, tag + "_renders.npz")
            np.savez(renders_path, **renders)

            state_path = os.path.join(self.ckpt_dir, tag + ".npz")
            np.savez(
                state_path, xyz=m.xyz_ws, scales=m.scales, quats=m.rotations,
                opacities=m.opacities, sh=m.sh_coeffs,
            )
            peak_vram = m.get_peak_vram_usage()
            after = _gpu_seconds(m.get_timing_breakdown())
            gpu_stage_s = {k: v - self.eval_gpu_s.get(k, 0.0) for k, v in before.items()}
            for k, v in after.items():
                self.eval_gpu_s[k] = self.eval_gpu_s.get(k, 0.0) + v - before.get(k, 0.0)
            self.eval_wall_s += perf_counter() - t_pause
            self.checkpoints.append({
                "step": n_done, "train_only_s": train_only_s, "gpu_stage_s": gpu_stage_s,
                "peak_vram": peak_vram, "num_splats": int(len(m.opacities)),
                "files": {"renders": os.path.basename(renders_path), "state": os.path.basename(state_path)},
            })
            self.write_trace()

    def write_trace(self, final=False):
        samples = self.sampler.samples if not final else self.sampler.stop()
        trace = {
            "prep_s": self.prep_s, "eval_wall_s": self.eval_wall_s,
            "render_dtype": self.cfg.ckpt_render_dtype,
            "gaussian_state_format": {
                "xyz": "(N,3) normalized frame", "scales": "(N,3) linear", "quats": "(N,4) raw wxyz as trained",
                "opacities": "sigmoid-activated", "sh": "(N,16,3)",
            },
            "checkpoints": self.checkpoints, "hwmon_samples": list(samples),
        }
        tmp = os.path.join(self.ckpt_dir, "trace.json.tmp")
        with open(tmp, "w") as f:
            json.dump(trace, f, indent=1)
        os.replace(tmp, os.path.join(self.ckpt_dir, "trace.json"))
