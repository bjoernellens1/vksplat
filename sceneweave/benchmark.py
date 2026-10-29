"""Portable, same-backend VkSplat benchmark: run, evaluate, compare.

Run through the Vulkan launcher (on Fedora: sceneweave/run_amd.sh).
No training code, losses, or optimizer policy are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eval_psnr import evaluate


def command_output(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(dataset: Path):
    os_info = {}
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        for line in os_release.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k in ("ID", "VERSION_ID", "PRETTY_NAME"):
                    os_info[k] = v.strip('"')
    vulkan = command_output("vulkaninfo", "--summary")
    devices = []
    if vulkan:
        devices = re.findall(r"deviceName\s*=\s*(.*)", vulkan)
    relevant = ("sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": platform.processor() or command_output("uname", "-m"),
        "os": os_info,
        "python": sys.version.split()[0],
        "gpu_devices_reported_by_vulkaninfo": devices,
        "vulkan_summary": vulkan,
        "lspci_display": command_output("lspci", "-nn"),
        "vksplat_git_revision": command_output("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "vksplat_git_dirty": bool(command_output("git", "-C", str(ROOT), "status", "--porcelain")),
        "dataset": str(dataset),
        "colmap_sha256": {p: sha256(dataset / p) for p in relevant},
        "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
        "mesa_vulkan_device_select": os.environ.get("MESA_VK_DEVICE_SELECT"),
    }


def run(args):
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset missing: {dataset}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = metadata(dataset)
    records = []
    for steps in args.steps:
        if steps <= 0:
            raise SystemExit("Steps must be positive")
        for repeat in range(args.repeats):
            name = f"steps_{steps}_repeat_{repeat + 1:02d}"
            dest = output / name
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(HERE / "run_baseline.py"),
                "--dataset", str(dataset), "--output", str(dest),
                "--image-dir", args.image_dir, "--steps", str(steps),
                "--strategy", args.strategy, "--cap-max", str(args.cap_max)
            ]
            print(f"=== {name} ===", flush=True)
            start = time.perf_counter()
            with (dest / "train.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT / "vksplat", stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
            wall = time.perf_counter() - start
            subdirs = sorted((p for p in dest.iterdir()
                              if p.is_dir() and (p / "train.json").exists()),
                             key=lambda p: p.stat().st_mtime_ns)
            record = {
                "name": name, "steps": steps, "repeat": repeat + 1,
                "exit_code": proc.returncode, "invocation_wall_seconds": wall,
                "run_directory": str(subdirs[-1]) if subdirs else None,
                "gpu_training_seconds_reported": None,
                "training_iterations_per_second": None,
                "invocation_iterations_per_second": steps / wall if wall > 0 else None,
                "peak_vram_bytes_reported": None,
                "num_splats": None,
                "mean_psnr_db": None,
                "num_validation_images": None,
                "quality_scope": "photometric_only",
            }
            if proc.returncode == 0 and subdirs:
                last = subdirs[-1]
                report = json.loads((last / "train.json").read_text())
                record["gpu_training_seconds_reported"] = report.get("time_elapsed")
                training_seconds = record["gpu_training_seconds_reported"]
                if isinstance(training_seconds, (int, float)) and training_seconds > 0:
                    record["training_iterations_per_second"] = steps / training_seconds
                record["peak_vram_bytes_reported"] = report.get("peak_vram")
                record["num_splats"] = report.get("num_splats")
                try:
                    quality = evaluate(last)
                    (last / "eval_psnr.json").write_text(json.dumps(quality, indent=2) + "\n")
                    record["mean_psnr_db"] = quality["mean_psnr_db"]
                    record["num_validation_images"] = quality["count"]
                except (OSError, ValueError, KeyError) as exc:
                    record["evaluation_error"] = str(exc)
            records.append(record)
            summary = {"schema_version": "1.0.0", "benchmark": "vksplat_same_engine",
                       "mode": "rgb_only", "settings": {"steps": args.steps,
                       "repeats": args.repeats, "strategy": args.strategy,
                       "image_dir": args.image_dir, "cap_max": args.cap_max},
                       "machine": info, "runs": records}
            (output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({k:v for k,v in record.items()
                             if k in ("name","exit_code","invocation_wall_seconds",
                                      "gpu_training_seconds_reported","training_iterations_per_second",
                                      "invocation_iterations_per_second","mean_psnr_db",
                                      "num_splats")}, indent=2), flush=True)
            if proc.returncode != 0:
                raise SystemExit(f"Failed: {name}. See {dest / 'train.log'}")
    print("Wrote", output / "benchmark.json")


def compare(args):
    results = []
    for path in args.files:
        obj = json.loads(path.read_text())
        if obj.get("schema_version") != "1.0.0" or obj.get("benchmark") != "vksplat_same_engine":
            raise SystemExit(f"Not a supported benchmark file: {path}")
        for item in obj["runs"]:
            if item.get("exit_code") != 0 or item.get("mean_psnr_db") is None:
                continue
            results.append({
                "machine": obj["machine"].get("hostname", "unknown"),
                "gpu": "; ".join(obj["machine"].get("gpu_devices_reported_by_vulkaninfo", [])),
                "revision": obj["machine"].get("vksplat_git_revision"),
                "dataset_sha": obj["machine"].get("colmap_sha256"),
                "image_dir": obj["settings"]["image_dir"],
                "strategy": obj["settings"]["strategy"],
                "steps": item["steps"], "repeat": item["repeat"],
                "wall_s": item["invocation_wall_seconds"],
                "training_s": item["gpu_training_seconds_reported"],
                "train_it_s": (item["steps"] / item["gpu_training_seconds_reported"]
                               if isinstance(item.get("gpu_training_seconds_reported"), (int, float))
                               and item["gpu_training_seconds_reported"] > 0 else None),
                "wall_it_s": (item["steps"] / item["invocation_wall_seconds"]
                              if isinstance(item.get("invocation_wall_seconds"), (int, float))
                              and item["invocation_wall_seconds"] > 0 else None),
                "psnr_db": item["mean_psnr_db"],
                "splats": item["num_splats"],
            })
    if not results:
        raise SystemExit("No successful quality-scored runs")
    if not args.allow_different_inputs:
        fingerprints = {(json.dumps(r["dataset_sha"], sort_keys=True), r["image_dir"],
                         r["strategy"], r["revision"]) for r in results}
        if len(fingerprints) != 1:
            raise SystemExit("Inputs, strategy or revision differ; use --allow-different-inputs for descriptive-only table")
    writer = csv.writer(sys.stdout)
    writer.writerow(["machine","gpu","revision","steps","repeat","total_wall_s",
                     "training_s","train_it_s","wall_it_s","psnr_db","gaussians"])
    for r in sorted(results, key=lambda x:(x["steps"],x["machine"],x["repeat"])):
        writer.writerow([r["machine"],r["gpu"],(r["revision"] or "")[:12],
                         r["steps"],r["repeat"],f'{r["wall_s"]:.3f}',
                         "" if r["training_s"] is None else f'{r["training_s"]:.3f}',
                         "" if r["train_it_s"] is None else f'{r["train_it_s"]:.2f}',
                         "" if r["wall_it_s"] is None else f'{r["wall_it_s"]:.2f}',
                         f'{r["psnr_db"]:.3f}',r["splats"]])


def main():
    p = argparse.ArgumentParser(description="Reproducible VkSplat benchmark across PCs")
    subs = p.add_subparsers(dest="action", required=True)
    r = subs.add_parser("run")
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="+", default=[1000, 3000])
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--image-dir", default="images_2")
    r.add_argument("--strategy", choices=["default","mcmc"], default="default")
    r.add_argument("--cap-max", type=int, default=1_000_000)
    c = subs.add_parser("compare")
    c.add_argument("files", type=Path, nargs="+")
    c.add_argument("--allow-different-inputs", action="store_true")
    args = p.parse_args()
    if args.action == "run":
        if args.repeats < 1:
            p.error("repeats must be positive")
        run(args)
    else:
        compare(args)

if __name__ == "__main__":
    main()
