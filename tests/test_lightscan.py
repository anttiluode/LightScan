import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lightscan import Zone,SimCamera,dot_centroid,fit_geometry,pattern_grid,summarize_levels,execute

class LightScanTests(unittest.TestCase):
    def test_dot_localization_and_homography(self):
        z=Zone(960,720,540)
        cam=SimCamera(z)
        dark=cam.capture(z.base())
        uv=pattern_grid(3)
        detected=[]
        for p in uv:
            xy,peak=dot_centroid(cam.capture(z.dot(p)),dark)
            self.assertGreater(peak,40)
            detected.append(xy)
        H,rmse,errs,inliers=fit_geometry(z,uv,detected)
        self.assertLess(rmse,2.5)
        self.assertGreaterEqual(inliers.sum(),8)
        src=np.float32([z.xy((.48,.62))]).reshape(-1,1,2)
        import cv2
        err=np.linalg.norm(cv2.perspectiveTransform(src,H)-cv2.perspectiveTransform(src,cam.h_true))
        self.assertLess(err,3.0)

    def test_exposure_drift_warning(self):
        rows=[{'screen_level':lv,'camera_mean':v,'clipped_fraction':0.0} for lv,v in
              [(0,9),(.25,50),(.5,100),(.75,150),(1,200),(.5,115),(.25,70)]]
        s=summarize_levels(rows)
        self.assertFalse(s['stable'])
        self.assertTrue(any('DRIFT' in w for w in s['warnings']))

    def test_end_to_end_simulated_calibration_and_scan(self):
        with tempfile.TemporaryDirectory() as t:
            args=SimpleNamespace(sim=True,sim_auto_exposure=False,headless=True,
                zone_px=540,screen_width=None,screen_height=None,origin_x=0,origin_y=0,
                url='',camera_mode='shot',settle=0,samples=1,output=t,scan=True,preview=False)
            r=execute(args)
            self.assertLess(r['geometry_rmse_camera_px'],3)
            self.assertTrue(r['stability']['stable'],r['stability'])
            self.assertGreater(r['shadow_scan']['mean_camera_level_difference'],.10)
            self.assertGreater(r['shadow_scan']['hot_pixel_fraction'],.002)
            self.assertTrue((Path(t)/'calibration.json').is_file())
            self.assertTrue((Path(t)/'03_shadow_difference.png').is_file())

if __name__ == '__main__': unittest.main()
