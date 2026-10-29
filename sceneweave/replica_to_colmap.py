"""Convert Replica RGB-D + poses to VkSplat's COLMAP text format.

Two initialization modes share *exactly* the same training images and poses:
- measured: point seeds from a deterministic subset of training depth images
- oracle: points sampled from a provided reference mesh (privileged information!)

The oracle mode is an upper-bound research ablation, NOT a deployable mapper.
No geometry is ever asserted to be verified by the oracle conversion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import cv2

def quaternion_wxyz(R):
    """Stable rotation-matrix to scalar-first quaternion."""
    m = np.asarray(R, dtype=np.float64)
    vals, vecs = np.linalg.eigh(np.array([
        [1+m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2]],
        [m[0,1]+m[1,0], 1-m[0,0]+m[1,1]-m[2,2], m[1,2]+m[2,1], m[0,2]+m[2,0]],
        [m[0,2]+m[2,0], m[1,2]+m[2,1], 1-m[0,0]-m[1,1]+m[2,2], m[0,1]+m[1,0]],
        [m[2,1]-m[1,2], m[0,2]+m[2,0], m[0,1]+m[1,0], m[0,0]+m[1,1]+m[2,2]]
    ]))
    q = vecs[:, np.argmax(vals)]
    out = np.array([q[3],q[0],q[1],q[2]])
    return out * (-1 if out[0]<0 else 1)

def trajectory(path):
    arr=np.loadtxt(path,dtype=np.float64)
    arr=np.atleast_2d(arr)
    if arr.shape[1]!=16: raise ValueError("Trajectory must contain one row of 16 matrix entries per frame")
    mats=arr.reshape(-1,4,4)
    for i,m in enumerate(mats):
        if not np.all(np.isfinite(m)) or not np.allclose(m[3],[0,0,0,1],atol=1e-4):
            raise ValueError(f"Invalid homogeneous pose {i}")
        if np.linalg.norm(m[:3,:3].T@m[:3,:3]-np.eye(3))>0.02 or np.linalg.det(m[:3,:3])<0.98:
            raise ValueError(f"Nonrigid camera rotation {i}")
    return mats

def pair_files(scene):
    rgb=sorted((scene/"results").glob("frame*.jpg"))
    if not rgb: rgb=sorted((scene/"results").glob("frame*.png"))
    depth=sorted((scene/"results").glob("depth*.png"))
    if not rgb or not depth: raise ValueError("Expected Replica results/frame*.jpg or png and depth*.png")
    def number(p):
        import re
        match=re.search(r"(\d+)$",p.stem)
        if not match: raise ValueError(f"Unindexed image: {p}")
        return int(match.group(1))
    r={number(p):p for p in rgb}
    d={number(p):p for p in depth}
    if len(r)!=len(rgb) or len(d)!=len(depth): raise ValueError("Duplicate frame indices")
    shared=sorted(r.keys() & d.keys())
    return [(i,r[i],d[i]) for i in shared]

def measured_points(pairs, poses, fx,fy,cx,cy, depth_scale, frame_stride, pixel_stride, limit):
    points=[]; colors=[]
    for index,rgb_path,depth_path in pairs[::frame_stride]:
        d=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        im=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        if d is None or im is None or d.ndim!=2: raise ValueError(f"Cannot read depth/RGB pair at {index}")
        if d.shape!=im.shape[:2]: raise ValueError(f"RGB/depth resolution mismatch at {index}")
        h,w=d.shape
        yy,xx=np.mgrid[0:h:pixel_stride,0:w:pixel_stride]
        z=d[::pixel_stride,::pixel_stride].astype(np.float64)/depth_scale
        valid=np.isfinite(z)&(z>0.1)&(z<15)
        x=(xx[valid]-cx)*z[valid]/fx
        y=(yy[valid]-cy)*z[valid]/fy
        p=np.stack([x,y,z[valid]],axis=1)
        # Camera axes convention required: input trajectory is camera-to-world OpenCV.
        world=(poses[index][:3,:3]@p.T).T+poses[index][:3,3]
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[::pixel_stride,::pixel_stride][valid]
        points.append(world);colors.append(rgb)
    if not points: raise ValueError("No measured points")
    pts=np.concatenate(points);cols=np.concatenate(colors)
    rng=np.random.default_rng(0)
    if len(pts)>limit:
        ids=rng.choice(len(pts),size=limit,replace=False)
        pts=pts[ids];cols=cols[ids]
    return pts,cols

def mesh_points(path,count):
    try: import trimesh
    except ImportError as e: raise RuntimeError("Install trimesh in Pixi for oracle mesh sampling") from e
    mesh=trimesh.load(path,force="mesh",process=False)
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0:
        raise ValueError("Oracle mesh has no triangles")
    # Deterministic triangle/area sampling, avoiding global random state in trimesh.sample().
    faces=mesh.triangles
    areas=mesh.area_faces.astype(np.float64)
    if not np.all(np.isfinite(areas)) or areas.sum()<=0: raise ValueError("Bad mesh areas")
    rng=np.random.default_rng(0)
    idx=rng.choice(len(areas),size=count,replace=True,p=areas/areas.sum())
    u=rng.random((count,1));v=rng.random((count,1))
    mask=u+v>1;u[mask]=1-u[mask];v[mask]=1-v[mask]
    tri=faces[idx]
    pts=tri[:,0]+u*(tri[:,1]-tri[:,0])+v*(tri[:,2]-tri[:,0])
    # Fixed neutral albedo for baseline oracle: do not leak held-out RGB appearance.
    return pts,np.full((count,3),128,dtype=np.uint8)

def write_colmap(out, selected, poses, K, points, colors, source, depth_scale, seed_indices):
    sparse=out/"sparse"/"0"
    image_dir=out/"images"
    sparse.mkdir(parents=True,exist_ok=True)
    image_dir.mkdir(parents=True,exist_ok=True)
    w,h,fx,fy,cx,cy=K
    (sparse/"cameras.txt").write_text(f"# Camera list\n1 PINHOLE {w} {h} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")
    image_lines=["# Image list with two lines per image"]
    for j,(i,rgb,_depth) in enumerate(selected,1):
        c2w=poses[i]
        w2c=np.linalg.inv(c2w)
        q=quaternion_wxyz(w2c[:3,:3])
        t=w2c[:3,3]
        name=f"{j:06d}{rgb.suffix.lower()}"
        dest=image_dir/name
        if not dest.exists(): dest.symlink_to(rgb.resolve())
        image_lines.extend([f"{j} {' '.join(map(lambda v:f'{v:.15g}',q))} {' '.join(map(lambda v:f'{v:.15g}',t))} 1 {name}",""])
    (sparse/"images.txt").write_text("\n".join(image_lines)+"\n")
    with (sparse/"points3D.txt").open("w") as fh:
        fh.write("# ID X Y Z R G B ERROR TRACK[]\n")
        for k,(p,c) in enumerate(zip(points,colors),1):
            fh.write(f"{k} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")
    manifest={"schema_version":"1.0.0","mode":source,"oracle_privileged":source=="oracle",
        "frame_indices":[i for i,_,_ in selected],"seed_frame_indices":seed_indices,
        "pose_format":"row-major camera-to-world OpenCV","depth_scale_units_per_meter":depth_scale,
        "num_seed_points":len(points),"coordinates":"meters","dataset":str(out),
        "warning":"Oracle mesh is privileged ground truth; measured mode seeds only from TRAIN frames."}
    (out/"sceneweave_conversion.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",type=Path,required=True,help="Replica scene directory with traj.txt and results/")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=["measured","oracle"],required=True)
    p.add_argument("--mesh",type=Path,help="Reference mesh, required only in oracle mode")
    p.add_argument("--fx",type=float,required=True);p.add_argument("--fy",type=float,required=True)
    p.add_argument("--cx",type=float,required=True);p.add_argument("--cy",type=float,required=True)
    p.add_argument("--depth-scale",type=float,default=1000,help="Depth pixel units per metre")
    p.add_argument("--frame-step",type=int,default=20)
    p.add_argument("--pixel-step",type=int,default=12)
    p.add_argument("--max-points",type=int,default=150000)
    p.add_argument("--val-every",type=int,default=8)
    a=p.parse_args()
    if a.mode=="oracle" and a.mesh is None: p.error("--mesh required for oracle mode")
    if any(v<=0 for v in [a.fx,a.fy,a.depth_scale,a.frame_step,a.pixel_step,a.max_points,a.val_every]):
        p.error("Positive intrinsics/scales/strides required")
    scene=a.scene.resolve()
    pairs=pair_files(scene)
    poses=trajectory(scene/"traj.txt")
    if max(i for i,_,_ in pairs)>=len(poses): p.error("Frame index outside pose trajectory")
    sample=cv2.imread(str(pairs[0][1]))
    h,w=sample.shape[:2]
    if not 0<=a.cx<w or not 0<=a.cy<h: p.error("Principal point outside image")
    # Choose an identical frame subset for both modes; held-out frames are not used for seeds.
    selected=pairs[::a.frame_step]
    if len(selected)<9: p.error("Need >=9 selected frames for meaningful holdout")
    training=[item for j,item in enumerate(selected) if j%a.val_every!=0]
    if not training: p.error("No training frames")
    if a.mode=="measured":
        pts,cols=measured_points(training,poses,a.fx,a.fy,a.cx,a.cy,
            a.depth_scale,1,a.pixel_step,a.max_points)
    else: pts,cols=mesh_points(a.mesh.resolve(),a.max_points)
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()): p.error("Output directory must be empty")
    manifest=write_colmap(out,selected,poses,(w,h,a.fx,a.fy,a.cx,a.cy),pts,cols,
                          a.mode,a.depth_scale,[i for i,_,_ in training])
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
