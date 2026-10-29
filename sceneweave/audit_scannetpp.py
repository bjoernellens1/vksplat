"""Read-only ScanNet++ scene inventory; does not alter dataset or assume calibration.

Run on a ScanNet++ scene directory; surfaces available modalities and files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CANDIDATES={
  "metric_mesh":["scans/mesh_aligned_0.05.ply","scans/mesh_aligned_0.05_semantic.ply"],
  "dslr_colmap":["dslr/colmap/cameras.txt","dslr/colmap/images.txt","dslr/colmap/points3D.txt"],
  "dslr_transforms":["dslr/nerfstudio/transforms_undistorted.json","dslr/nerfstudio/transforms.json"],
  "iphone_registration":["iphone/pose_intrinsic_imu.json"],
  "iphone_colmap":["iphone/colmap/cameras.txt","iphone/colmap/images.txt"],
  "iphone_depth":["iphone/depth.bin"],
  "iphone_rgb":["iphone/rgb.mkv"],
}
def inventory(scene):
    summary={"scene":str(scene.resolve()),"exists":scene.is_dir(),"assets":{}}
    if not scene.is_dir():
        return summary
    for key,options in CANDIDATES.items():
        summary["assets"][key]=[
            {"path":name,"bytes":(scene/name).stat().st_size}
            for name in options if (scene/name).is_file()]
    summary["folders"]={
       name: {"exists":(scene/name).is_dir(),
              "file_count":sum(p.is_file() for p in (scene/name).iterdir()) if (scene/name).is_dir() else 0}
       for name in ("dslr/resized_undistorted_images","dslr/resized_images",
                    "iphone/rgb","iphone/depth","scans")}
    return summary

def main():
    p=argparse.ArgumentParser(description="Audit ScanNet++ scene data before conversion")
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=inventory(a.scene)
    content=json.dumps(data,indent=2)
    print(content)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(content+"\\n")
    if not data["exists"]:raise SystemExit("Scene directory not found")

if __name__=="__main__":main()
