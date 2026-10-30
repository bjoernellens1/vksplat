"""ScanNet++ iPhone RGB-D -> VkSplat COLMAP text, measured geometry only.

Uses aligned_pose and frame-specific RGB intrinsics from pose_intrinsic_imu.json.
Does NOT access reference meshes or create oracle points.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def parse_camera(record, frame_id):
    p=np.asarray(record["aligned_pose"],dtype=np.float64)
    K=np.asarray(record["intrinsic"],dtype=np.float64)
    if p.shape!=(4,4) or K.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(K).all():
        raise ValueError(f"Bad pose/intrinsics: {frame_id}")
    if not np.allclose(p[3],[0,0,0,1],atol=1e-3):
        raise ValueError(f"Invalid pose bottom row: {frame_id}")
    rot=p[:3,:3]
    if np.linalg.norm(rot.T@rot-np.eye(3))>0.025 or np.linalg.det(rot)<0.98:
        raise ValueError(f"Non-rigid pose: {frame_id}")
    if K[0,0]<=0 or K[1,1]<=0:
        raise ValueError(f"Invalid focal length: {frame_id}")
    return p,K

def scale_intrinsics(K, source_wh, destination_wh):
    out=K.copy()
    sx=destination_wh[0]/source_wh[0];sy=destination_wh[1]/source_wh[1]
    out[0,:]*=sx;out[1,:]*=sy
    return out

def select_frames(metadata, max_frames, frame_stride):
    # Sample the complete trajectory to retain meaningful scene coverage.
    ids=sorted(int(k.removeprefix("frame_")) for k in metadata if k.startswith("frame_"))
    ids=ids[::frame_stride]
    if len(ids)>max_frames:
        selected=np.linspace(0,len(ids)-1,max_frames,dtype=int)
        ids=[ids[i] for i in selected]
    if len(ids)<9:raise ValueError("At least 9 selected frames required")
    return ids

def seed_from_depth(depth, color, mask, K, pose, pixel_step, depth_scale, max_depth):
    h,w=depth.shape
    yy,xx=np.mgrid[0:h:pixel_step,0:w:pixel_step]
    z=depth[::pixel_step,::pixel_step].astype(np.float64)/depth_scale
    valid=np.isfinite(z)&(z>0.1)&(z<max_depth)
    valid &= mask[::pixel_step,::pixel_step]
    x=(xx[valid]-K[0,2])*z[valid]/K[0,0]
    y=(yy[valid]-K[1,2])*z[valid]/K[1,1]
    local=np.stack((x,y,z[valid]),axis=1)
    world=local@pose[:3,:3].T+pose[:3,3]
    rgb=color[::pixel_step,::pixel_step][valid]
    return world,rgb

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-frames",type=int,default=176)
    parser.add_argument("--frame-stride",type=int,default=30)
    parser.add_argument("--seed-frame-step",type=int,default=8)
    parser.add_argument("--pixel-step",type=int,default=2)
    parser.add_argument("--max-points",type=int,default=100000)
    parser.add_argument("--rgb-width",type=int,default=640)
    parser.add_argument("--rgb-height",type=int,default=480)
    parser.add_argument("--depth-scale",type=float,default=1000)
    parser.add_argument("--max-depth",type=float,default=8.0)
    parser.add_argument("--mask-valid",choices=("nonzero","zero"),default="nonzero",
                        help="Which pixels in ScanNet++ anonymization mask are valid")
    parser.add_argument("--pose-axes",choices=("opencv","opengl"),required=True,
                        help="Must be validated against the aligned scan mesh")
    args=parser.parse_args()
    if any(v<=0 for v in (args.max_frames,args.frame_stride,args.seed_frame_step,args.pixel_step,
                          args.max_points,args.rgb_width,args.rgb_height,args.depth_scale,args.max_depth)):
        parser.error("All counts, strides and scales must be positive")
    scene=args.scene.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):parser.error("Output must be empty")
    iphone=scene/"iphone"
    meta=json.loads((iphone/"pose_intrinsic_imu.json").read_text())
    ids=select_frames(meta,args.max_frames,args.frame_stride)
    seed_ids=set(ids[i] for i in range(len(ids)) if i%8!=0)[::args.seed_frame_step]
    # Above slicing is performed on the TRAIN frame list, never validation.
    sparse=out/"sparse"/"0";images=out/"images";masks=out/"masks"
    sparse.mkdir(parents=True);images.mkdir();masks.mkdir()
    camera_lines=["# PINHOLE camera for each frame"]
    image_lines=["# COLMAP images: 2 lines per frame"]
    seed_pts=[];seed_rgb=[];used_seed_ids=[]
    for j,idx in enumerate(ids,1):
        key=f"frame_{idx:06d}"
        record=meta[key]
        pose,K=parse_camera(record,key)
        if args.pose_axes=="opengl":
            pose=pose@np.diag([1.,-1.,-1.,1.])
        rgb_path=iphone/"rgb"/f"{key}.jpg"
        depth_path=iphone/"depth"/f"{key}.png"
        mask_path=iphone/"rgb_masks"/f"{key}.png"
        bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        d=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if bgr is None or d is None or mask is None:
            raise ValueError(f"Missing RGB/depth/mask: {key}")
        H,W=bgr.shape[:2]
        if mask.shape!=(H,W) or d.ndim!=2:
            raise ValueError(f"Bad RGB/depth/mask shapes at {key}")
        K_rgb=scale_intrinsics(K,(W,H),(args.rgb_width,args.rgb_height))
        K_depth=scale_intrinsics(K,(W,H),(d.shape[1],d.shape[0]))
        scaled=cv2.resize(bgr,(args.rgb_width,args.rgb_height),interpolation=cv2.INTER_AREA)
        valid=(mask>0) if args.mask_valid=="nonzero" else (mask==0)
        valid_rgb=cv2.resize(valid.astype(np.uint8),(args.rgb_width,args.rgb_height),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        # VkSplat supports separate grayscale masks read as alpha.
        cv2.imwrite(str(images/f"{j:06d}.png"),scaled)
        cv2.imwrite(str(masks/f"{j:06d}.png"),valid_rgb.astype(np.uint8)*255)
        camera_lines.append(f"{j} PINHOLE {args.rgb_width} {args.rgb_height} "
                            f"{K_rgb[0,0]:.10g} {K_rgb[1,1]:.10g} {K_rgb[0,2]:.10g} {K_rgb[1,2]:.10g}")
        w2c=np.linalg.inv(pose)
        x,y,z,w=Rotation.from_matrix(w2c[:3,:3]).as_quat()
        if w<0:x,y,z,w=-x,-y,-z,-w
        t=w2c[:3,3]
        image_lines += [f"{j} {w:.12g} {x:.12g} {y:.12g} {z:.12g} "
                        f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {j} {j:06d}.png",""]
        if idx in seed_ids:
            valid_depth=cv2.resize(valid.astype(np.uint8),(d.shape[1],d.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            rgb_depth=cv2.cvtColor(cv2.resize(bgr,(d.shape[1],d.shape[0]),
                               interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            pts,cols=seed_from_depth(d,rgb_depth,valid_depth,K_depth,pose,
                         args.pixel_step,args.depth_scale,args.max_depth)
            if len(pts):
                seed_pts.append(pts);seed_rgb.append(cols);used_seed_ids.append(idx)
    if not seed_pts:raise ValueError("No valid depth points in selected train frames")
    xyz=np.concatenate(seed_pts);colors=np.concatenate(seed_rgb)
    if len(xyz)>args.max_points:
        choice=np.random.default_rng(0).choice(len(xyz),args.max_points,replace=False)
        xyz=xyz[choice];colors=colors[choice]
    (sparse/"cameras.txt").write_text("\n".join(camera_lines)+"\n")
    (sparse/"images.txt").write_text("\n".join(image_lines)+"\n")
    with (sparse/"points3D.txt").open("w") as f:
        f.write("# ID XYZ RGB ERROR\n")
        for i,(p,c) in enumerate(zip(xyz,colors),1):
            f.write(f"{i} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")
    manifest=dict(schema_version="1.0.0",mode="measured",oracle_privileged=False,
        scene=str(scene),dataset=str(out),frame_indices=ids,seed_frame_indices=used_seed_ids,
        num_seed_points=len(xyz),rgb_resolution=[args.rgb_width,args.rgb_height],
        pose_source="aligned_pose",pose_axes=args.pose_axes,depth_scale_units_per_meter=args.depth_scale,
        mask_valid=args.mask_valid,mask_dir="masks",
        warning="Pose convention and masks must be checked visually and against mesh before research comparisons")
    (out/"sceneweave_conversion.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="frame_indices"},indent=2))

if __name__=="__main__":main()
