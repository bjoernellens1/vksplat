from pathlib import Path
import json
import cv2
import numpy as np

root = Path.home() / "git/sceneweave/outputs/vksplat"
runs = sorted(p for p in root.iterdir()
              if p.is_dir() and (p / "train.json").exists())
assert runs, "No completed VkSplat run found"

run = runs[-1]
metadata = json.loads((run / "train.json").read_text())
images = metadata["val_images"]

results = []

for i, entry in enumerate(images):
    ref_path = Path(entry["image_path"])
    pred_path = run / f"val_{i:05d}.png"

    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)

    assert ref is not None, f"Missing reference: {ref_path}"
    assert pred is not None, f"Missing render: {pred_path}"
    assert ref.shape == pred.shape, (
        f"Resolution mismatch: {ref.shape} vs {pred.shape}"
    )

    ref = ref.astype(np.float32) / 255.0
    pred = pred.astype(np.float32) / 255.0

    mse = np.mean((ref - pred) ** 2)
    psnr = -10 * np.log10(max(float(mse), 1e-12))
    results.append(float(psnr))

print("Run:", run)
print("Validation images:", len(results))
print(f"Mean PSNR: {np.mean(results):.3f} dB")
print(f"Median PSNR: {np.median(results):.3f} dB")

output = {
    "run": str(run),
    "num_images": len(results),
    "mean_psnr_db": float(np.mean(results)),
    "median_psnr_db": float(np.median(results)),
    "per_image_psnr_db": results,
}

(run / "eval_psnr.json").write_text(
    json.dumps(output, indent=2)
)
