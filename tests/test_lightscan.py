import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lightscan import Screen,Zone,SimCamera,dot_centroid,fit_geometry,pattern_grid,summarize_levels,execute,load_calibration,scan_region,probe_metrics

class LightScanTests(unittest.TestCase):
    def test_guide_banner_never_changes_calibrated_light_square(self):
        zone=Zone(1536,864,420)
        # Create a screen without opening a GUI, then exercise its compositing.
        screen=object.__new__(Screen)
        screen.zone=zone
        screen.headless=False
        screen.guide('STEP 3 / 3  |  HIDDEN SHAPE',
                     'Put a shape behind the diffuser.', 'SPACE when ready')
        pattern=zone.full(.5)
        guided=screen.guided_frame(pattern)
        self.assertTrue(np.array_equal(guided[zone.y0:zone.y0+zone.size,
                                              zone.x0:zone.x0+zone.size],
                                       pattern[zone.y0:zone.y0+zone.size,
                                               zone.x0:zone.x0+zone.size]))
        self.assertTrue(np.all(pattern[:80,:]==0))
        self.assertGreater(int(guided[:80,:].sum()),0)
        self.assertTrue(np.array_equal(guided[:80,:],screen.guided_frame(zone.full(0))[:80,:]))
        self.assertTrue(np.all(guided[80:zone.y0,:]==0))

    def test_scan_pause_uses_full_screen_space_not_hidden_terminal(self):
        zone=Zone(1536,864,420)
        screen=object.__new__(Screen)
        screen.zone=zone
        screen.headless=False
        shown=[]
        screen.show=lambda frame: shown.append(frame)
        with patch('builtins.input', side_effect=AssertionError('terminal prompt used')):
            with patch('lightscan.cv2.waitKey',side_effect=[-1,32]):
                screen.wait_for_space('INSERT DIFFUSER', ['Put blank diffuser in cradle.'])
        self.assertEqual(len(shown),1)
        self.assertEqual(screen.title,'INSERT DIFFUSER')

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

    def test_hadamard_patterns_are_balanced_and_independent(self):
        z=Zone(1536,864,420)
        pats=[z.structured(i,1.) for i in range(16)]
        mask=np.array([p[z.y0:z.y0+z.size,z.x0:z.x0+z.size,0].reshape(4,105,4,105).mean(axis=(1,3)) for p in pats])
        bits=np.where(mask>127,1.,-1.).reshape(16,16)
        np.testing.assert_allclose(bits@bits.T,16*np.eye(16))
        self.assertEqual(int((pats[0]>0).sum()),z.size**2*3)
        for p in pats[1:]:
            self.assertEqual(int((p>0).sum()),z.size**2*3//2)

    def test_saved_geometry_match_and_mismatch(self):
        import json, cv2
        z=Zone(960,720,420)
        cam=SimCamera(z)
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'rig.json'
            data={'schema_version':1,'screen_size_px':[960,720],
                  'zone_px':{'width':960,'height':720,'size':420},
                  'camera_size_px':[800,570],
                  'screen_to_camera_homography':cam.h_true.tolist(),
                  'geometry_rmse_camera_px':0.31}
            f.write_text(json.dumps(data))
            H,report=load_calibration(f,z,(570,800,3))
            np.testing.assert_allclose(H,cam.h_true)
            with self.assertRaisesRegex(ValueError,'zone differs'):
                load_calibration(f,Zone(960,720,400),(570,800,3))
            with self.assertRaisesRegex(ValueError,'camera resolution'):
                load_calibration(f,z,(720,1280,3))

    def test_scan_only_uses_saved_geometry_without_new_empty_calibration(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            z=Zone(960,720,420)
            cam=SimCamera(z)
            ref=Path(d)/'rig_locked.json'
            ref.write_text(json.dumps({'schema_version':1,'screen_size_px':[960,720],
              'zone_px':{'width':960,'height':720,'size':420},
              'camera_size_px':[800,570],
              'screen_to_camera_homography':cam.h_true.tolist(),
              'geometry_rmse_camera_px':.31}))
            args=SimpleNamespace(sim=True,sim_auto_exposure=False,headless=True,
              zone_px=420,screen_width=None,screen_height=None,origin_x=0,origin_y=0,
              url='',camera_mode='shot',settle=0,samples=2,output=str(Path(d)/'scan'),
              scan=False,preview=False,scan_only=True,calibration=str(ref),
              pattern_level=.85,save_frames=True)
            with patch('lightscan.stage_geometry',side_effect=AssertionError('unexpected calibration')):
                with patch('lightscan.stage_exposure',side_effect=AssertionError('unexpected exposure sweep')):
                    result=execute(args)
            self.assertEqual(result['geometry_source'],str(ref))
            self.assertGreater(result['shadow_scan']['rms_signal_to_repeat_ratio'],2)
            self.assertTrue((Path(d)/'scan'/'frames'/'blank_00.png').is_file())
            self.assertTrue((Path(d)/'scan'/'03_roi_mask.png').is_file())

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
                url='',camera_mode='shot',settle=0,samples=1,output=t,scan=True,preview=False,scan_only=False,calibration='',
                pattern_level=.85,save_frames=False)
            r=execute(args)
            self.assertLess(r['geometry_rmse_camera_px'],3)
            self.assertTrue(r['stability']['stable'],r['stability'])
            self.assertGreater(r['shadow_scan']['mean_absolute_difference_in_roi'],.30)
            self.assertGreater(r['shadow_scan']['rms_signal_to_repeat_ratio'],1.5)
            self.assertTrue((Path(t)/'calibration.json').is_file())
            self.assertTrue((Path(t)/'03_shadow_difference.png').is_file())

if __name__ == '__main__': unittest.main()
