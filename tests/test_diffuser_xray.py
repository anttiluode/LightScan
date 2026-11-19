"""Fast unit tests: no phone, no GPU or network required."""
import unittest
import numpy as np
import cv2
from diffuser_xray import random_pattern, projection, rectify, cam_normalized, render_comparison


class InverseRigTests(unittest.TestCase):
    def test_pattern_reproducible_and_nonempty(self):
        a=random_pattern(np.random.default_rng(19))
        b=random_pattern(np.random.default_rng(19))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(64,64))
        self.assertGreater(np.sum(a),0)

    def test_projection_rectification(self):
        square=np.array([[40,20],[240,28],[230,218],[35,210]],dtype=np.float32)
        M=projection(square,pad=0)
        inp=np.zeros((240,270),np.uint8)
        cv2.circle(inp,(int(square[:,0].mean()),int(square[:,1].mean())),9,255,-1)
        out=rectify(inp,M)
        self.assertEqual(out.shape,(64,64))
        self.assertGreater(float(out[29:35,29:35].mean()),140)

    def test_dark_subtract_is_nonnegative(self):
        raw=np.full((64,64),100,np.uint8)
        dark=np.full((64,64),120,np.uint8)
        self.assertEqual(float(cam_normalized(raw,dark,100).max()),0.)

    def test_three_panel_bounds(self):
        panel=render_comparison(np.full((64,64),.6,np.float32),
                                np.full((64,64),.3,np.float32),
                                np.zeros((64,64),np.uint8))
        self.assertEqual(panel.shape,(150,486,3))

if __name__=='__main__':unittest.main()
