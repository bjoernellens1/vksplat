"""Evaluate saved validation renders without PyTorch; respects 8/16-bit input."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def read_rgb(path: Path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim != 3 or raw.shape[2] not in (3,4):
        raise ValueError(f"Unsupported image channels: {path}: {raw.shape}")
    if raw.dtype==np.uint8:
        scale=255.0
    elif raw.dtype==np.uint16:
        scale=65535.0
    else:
        raise ValueError(f"Unsupported image dtype: {raw.dtype} ({path})")
    rgb=cv2.cvtColor(raw,cv2.COLOR_BGRA2RGB if raw.shape[2]==4 else cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float64)/scale

def evaluate(run:Path):
    metadata=json.loads((run/"train.json").read_text())
    results=[]
    for i,item in enumerate(metadata["val_images"]):
        actual=read_rgb(Path(item["image_path"]))
        rendered=read_rgb(run/f"val_{i:05d}.png")
        if actual.shape!=rendered.shape:
            raise ValueError(f"Image {i} shape mismatch {actual.shape} != {rendered.shape}")
        squared=np.square(actual-rendered)
        mask_path=item.get("mask_path")
        if mask_path:
            mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=actual.shape[:2]:
                raise ValueError(f"Missing or incorrectly shaped eval mask: {mask_path}")
            valid=mask>0
            if not valid.any():
                raise ValueError(f"Empty eval mask: {mask_path}")
            mse=float(np.mean(squared[valid]))
        else:
            mse=float(np.mean(squared))
        psnr=float(-10*np.log10(max(mse,1e-15)))
        results.append({"image_path":item["image_path"],"render_path":f"val_{i:05d}.png","psnr_db":psnr})
    if not results:
        raise ValueError("No validation images")
    return {"evaluation":"decoded_image_psnr","precision":"float64_from_original_8_or_16_bit_images",
            "count":len(results),"mean_psnr_db":float(np.mean([r["psnr_db"] for r in results])),
            "median_psnr_db":float(np.median([r["psnr_db"] for r in results])),
            "photometric_only":True,"mask_handling":"nonzero mask pixels only when mask_path available", "images":results}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True,type=Path)
    args=p.parse_args()
    result=evaluate(args.run)
    dest=args.run/"eval_psnr.json"
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="images"},indent=2))
    print("Saved",dest)

if __name__=="__main__":
    main()
