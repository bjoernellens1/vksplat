"""Check that the only difference in the B0/B1 COLMAP inputs is seed geometry."""
from pathlib import Path
import argparse
import json

def point_rows(path):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"): continue
        x=line.split()
        if len(x)<8: raise ValueError(f"Invalid points3D row: {line}")
        rows.append(x)
    return rows

def check(baseline:Path,oracle:Path):
    b=json.loads((baseline/"sceneweave_conversion.json").read_text())
    o=json.loads((oracle/"sceneweave_conversion.json").read_text())
    assert b["schema_version"]==o["schema_version"]=="1.0.0"
    assert b["mode"]=="measured" and o["mode"]=="oracle"
    for k in ("frame_indices","seed_frame_indices","num_seed_points","input_pose_camera_axes","depth_scale_units_per_meter"):
        assert b[k]==o[k],f"Mismatch: {k}"
    for n in ("cameras.txt","images.txt"):
        assert (baseline/"sparse/0"/n).read_bytes()==(oracle/"sparse/0"/n).read_bytes(),n
    p1=point_rows(baseline/"sparse/0/points3D.txt")
    p2=point_rows(oracle/"sparse/0/points3D.txt")
    assert len(p1)==len(p2)
    assert all((a[0],a[4:])==(b[0],b[4:]) for a,b in zip(p1,p2)), "IDs, RGB, or errors differ"
    moved=sum(a[1:4]!=b[1:4] for a,b in zip(p1,p2))
    return {"matched_seed_count":len(p1),"moved_seed_count":moved,
            "identical_camera_and_image_metadata":True,
            "identical_seed_ids_and_colors":True,
            "warning":"Oracle positions are privileged geometry; no metric accuracy claims from RGB-only training"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--oracle",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(check(a.baseline,a.oracle),indent=2))
if __name__=="__main__": main()
