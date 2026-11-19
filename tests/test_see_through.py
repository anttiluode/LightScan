"""No camera, GPU, network or GUI required."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from see_through import (Params,Inverse,make_pattern,glyph_pattern,screen_pixels,
    detect_roi,extract,normalize,synthetic_measure,baseline_fit,baseline_predict,metrics,simulate)


class SeeThroughTests(unittest.TestCase):
    def test_detects_only_projected_light(self):
        black=np.zeros((240,320),np.uint8)
        white=np.zeros_like(black)
        white[80:180,100:240]=110
        (x,y,w,h),stats=detect_roi(black,white)
        self.assertLessEqual(x,100)
        self.assertLessEqual(y,80)
        self.assertGreaterEqual(x+w,240)
        self.assertGreaterEqual(y+h,180)
        self.assertGreaterEqual(stats['signal_p995'],100.)

    def test_rejects_medium_with_no_information(self):
        blank=np.full((140,200),24,np.uint8)
        with self.assertRaisesRegex(ValueError,'signal'):
            detect_roi(blank,blank)

    def test_no_ui_inside_source_zone(self):
        pat=np.eye(8,dtype=np.uint8)
        frame=screen_pixels(960,720,420,pat)
        self.assertEqual(frame.shape,(720,960,3))
        self.assertEqual(int(frame[:60].sum()),0)
        self.assertEqual(int(frame[360,480,0]),235)
        self.assertEqual(int(frame[360,730,0]),0)
        self.assertGreater(int(frame[150:570,270:690].sum()),0)

    def test_inverse_beats_blurry_direct_baseline(self):
        rng=np.random.default_rng(7);g=8;nf=40
        white=synthetic_measure(np.ones(g*g,np.float32),g,nf,rng,3,0)
        dark=np.full((nf,nf),8,np.float32)
        x=[];y=[]
        for i in range(115):
            pat=make_pattern(rng,g,i)
            x.append(pat.ravel())
            cam=synthetic_measure(pat,g,nf,rng,3,1)
            y.append(normalize(cam,dark,white))
        x=np.float32(x);y=np.float32(y)
        model=Inverse(.055).fit(y[:95],x[:95])
        pr=model.predict(y[95:])
        baseline=baseline_fit(y[:95],x[:95],g,nf)
        direct=baseline_predict(y[95:],baseline,g,nf)
        self.assertLess(float(np.abs(pr-x[95:]).mean()),float(np.abs(direct-x[95:]).mean())-.02)
        # The net is only given the CAMERA observation of a previously unseen word.
        word=glyph_pattern('HI',g)
        camera=synthetic_measure(word,g,nf,rng,3,1)
        unseen=model.predict(normalize(camera,dark,white))[0]
        self.assertEqual(unseen.shape,(g*g,))
        self.assertLess(float(np.abs(unseen-word.ravel()).mean()),.23)

    def test_save_reload_does_not_change_output(self):
        rng=np.random.default_rng(9)
        y=rng.random((30,48),dtype=np.float32)
        x=rng.random((30,64),dtype=np.float32)
        model=Inverse().fit(y,x)
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'model.npz'
            model.save(path,Params(grid=8,features=40),(1,2,20,30),np.zeros((40,40)),np.ones((40,40)),{'beats_baseline':True})
            m,params,roi,dark,white,report=Inverse.load(path)
            np.testing.assert_allclose(model.predict(y[:4]),m.predict(y[:4]),atol=1e-5)
            self.assertEqual(params.grid,8)
            self.assertEqual(roi,(1,2,20,30))
            self.assertTrue(report['beats_baseline'])

    def test_simulation_is_labeled_and_has_no_real_camera(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as d:
            args=Namespace(grid=8,features=32,ridge=.05,seed=22,samples=65,holdout=9,sim_blur=2.5)
            result=simulate(args,Path(d))
            self.assertIn('SIMULATED_ONLY',result)
            self.assertTrue((Path(d)/'sim_known.png').is_file())
            self.assertTrue((Path(d)/'sim_recovered.png').is_file())
            self.assertTrue(result['beats_baseline'])


if __name__=='__main__':unittest.main()
