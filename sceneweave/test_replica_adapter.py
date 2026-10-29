"""CPU-only tests for the Replica converter (run with Pixi Python)."""
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replica_to_colmap import quaternion_wxyz, trajectory, measured_points, write_colmap
from check_replica_pair import check

class ReplicaAdapterTest(unittest.TestCase):
    def test_identity_quaternion(self):
        q=quaternion_wxyz(np.eye(3))
        np.testing.assert_allclose(q,[1,0,0,0],atol=1e-6)

    def test_rotation_quaternion(self):
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        q=quaternion_wxyz(R)
        np.testing.assert_allclose(abs(q[0]),np.sqrt(.5),atol=1e-6)
        np.testing.assert_allclose(abs(q[3]),np.sqrt(.5),atol=1e-6)
        np.testing.assert_allclose(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(),R,atol=1e-6)

    def test_quaternion_roundtrip_arbitrary_rotations(self):
        for euler in ((0.3,-0.7,1.2),(-2.1,0.4,2.8),(3.14,0,0)):
            R=Rotation.from_euler('xyz',euler).as_matrix()
            q=quaternion_wxyz(R)
            reconstructed=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
            np.testing.assert_allclose(reconstructed,R,atol=1e-6)

    def test_depth_backprojection_and_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"results").mkdir()
            poses=np.repeat(np.eye(4)[None,:,:],2,axis=0)
            np.savetxt(root/"traj.txt",poses.reshape(2,16))
            bgr=np.zeros((4,4,3),dtype=np.uint8);bgr[:]=[1,2,3]
            cv2.imwrite(str(root/"results/frame000000.jpg"),bgr)
            cv2.imwrite(str(root/"results/depth000000.png"),np.full((4,4),1000,dtype=np.uint16))
            pts,cols=measured_points([(0,root/"results/frame000000.jpg",root/"results/depth000000.png")],
                                     poses,2,2,1.5,1.5,1000,1,2,100)
            self.assertEqual(pts.shape,(4,3))
            np.testing.assert_allclose(pts[:,2],1)
            self.assertEqual(len(cols),4)
            for mode in ("measured","oracle"):
                out=root/mode
                xyz=pts.copy()
                if mode=="oracle": xyz[:,0]+=.1
                write_colmap(out,[(0,root/"results/frame000000.jpg",root/"results/depth000000.png")],
                             poses,(4,4,2,2,1.5,1.5),xyz,cols,mode,1000,[0])
                m=json_load(out/"sceneweave_conversion.json")
                m["input_pose_camera_axes"]="opencv"
                (out/"sceneweave_conversion.json").write_text(__import__("json").dumps(m))
            result=check(root/"measured",root/"oracle")
            self.assertEqual(result["moved_seed_count"],4)
            self.assertTrue(result["identical_seed_ids_and_colors"])

def json_load(path):
    import json
    return json.loads(path.read_text())

if __name__=="__main__": unittest.main()
