"""CPU tests for ScanNet++ aligned camera and depth conversion."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from scannetpp_to_colmap import parse_camera,scale_intrinsics,seed_from_depth

class ScanNetPPAdapterTest(unittest.TestCase):
    def test_intrinsic_scaling(self):
        K=np.array([[1425.6107,0,955.303],[0,1425.6107,724.05304],[0,0,1.]])
        down=scale_intrinsics(K,(1920,1440),(256,192))
        self.assertAlmostEqual(down[0,0],190.0814267,places=4)
        self.assertAlmostEqual(down[1,2],96.5404053,places=4)
    def test_depth_backprojection_and_mask(self):
        d=np.full((4,4),1000,dtype=np.uint16)
        rgb=np.zeros((4,4,3),dtype=np.uint8)
        valid=np.ones((4,4),dtype=bool)
        valid[0,0]=False
        K=np.array([[2.,0,1.5],[0,2.,1.5],[0,0,1.]])
        xyz,col=seed_from_depth(d,rgb,valid,K,np.eye(4),1,1000,8.)
        self.assertEqual(xyz.shape,(15,3))
        self.assertEqual(col.shape,(15,3))
        np.testing.assert_allclose(xyz[:,2],1.)
    def test_aligned_pose(self):
        p=np.eye(4);p[:3,3]=[3,4,5]
        r={"aligned_pose":p.tolist(),"intrinsic":np.eye(3).tolist()}
        pose,K=parse_camera(r,"test")
        np.testing.assert_allclose(pose,p)
        np.testing.assert_allclose(K,np.eye(3))

if __name__=="__main__":unittest.main()
