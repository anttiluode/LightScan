"""LightScan: screen-to-camera geometric and radiometric calibration.

This is a *measurement rig*, not a CT or depth reconstruction algorithm.
The source screen is controlled by this process. Phone JPEG exposure and tone
mapping must be locked manually in the phone app where supported.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Zone:
    width: int
    height: int
    size: int

    @property
    def x0(self) -> int:
        return (self.width - self.size) // 2

    @property
    def y0(self) -> int:
        return (self.height - self.size) // 2

    def xy(self, uv: tuple[float, float]) -> tuple[int, int]:
        return round(self.x0 + self.size * uv[0]), round(self.y0 + self.size * uv[1])

    def corners(self) -> np.ndarray:
        x, y, s = self.x0, self.y0, self.size
        return np.array([[x, y], [x+s, y], [x+s, y+s], [x, y+s]], np.float32)

    def base(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), np.uint8)

    def full(self, level: float) -> np.ndarray:
        img = self.base()
        v = int(round(np.clip(level, 0., 1.) * 255))
        img[self.y0:self.y0+self.size, self.x0:self.x0+self.size] = (v, v, v)
        return img

    def dot(self, uv: tuple[float, float], brightness: float = .90) -> np.ndarray:
        img = self.base()
        cv2.circle(img, self.xy(uv), max(8, self.size//22),
                   (int(brightness*255),)*3, -1, cv2.LINE_AA)
        return img

    def board(self) -> np.ndarray:
        img = self.base()
        sq = max(5, self.size // 12)
        cx, cy = self.xy((.5, .5))
        for j in range(-4, 4):
            for i in range(-4, 4):
                if (i + j) % 2:
                    x0, y0 = cx + i*sq, cy + j*sq
                    cv2.rectangle(img, (x0, y0), (x0+sq, y0+sq),
                                  (210, 210, 210), -1)
        return img


class Screen:
    def __init__(self, zone: Zone, headless: bool, origin: tuple[int,int]=(0,0)):
        self.zone = zone
        self.headless = headless
        self.name = 'LightScan | illumination screen (Esc to abort)'
        self.current = zone.base()
        if not headless:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.name, *origin)
            cv2.resizeWindow(self.name, zone.width, zone.height)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, frame: np.ndarray):
        self.current = frame
        if not self.headless:
            cv2.imshow(self.name, frame)
            k = cv2.waitKey(40) & 255
            if k == 27:
                raise KeyboardInterrupt('Escape pressed')

    def close(self):
        if not self.headless:
            cv2.destroyAllWindows()


class IPCamera:
    """IP Webcam still endpoint is preferred to avoid MJPEG frame backlog."""
    def __init__(self, url: str, mode: str='shot'):
        self.url = url.rstrip('/')+'/'
        self.mode = mode
        self.cap = None
        if mode == 'video':
            address = urllib.parse.urljoin(self.url, 'video')
            self.cap = cv2.VideoCapture(address)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open video: {address}')
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # backend may ignore
        elif mode != 'shot':
            raise ValueError('Camera mode must be shot or video')

    def capture(self, _pattern: np.ndarray|None=None, _inserted: bool=False) -> np.ndarray:
        if self.mode == 'video':
            for _ in range(2):
                ok, img = self.cap.read()
                if not ok: raise RuntimeError('IP Webcam video stream stopped')
            return img
        address = urllib.parse.urljoin(self.url, 'shot.jpg')
        req = urllib.request.Request(address, headers={'User-Agent': 'LightScan/1'})
        with urllib.request.urlopen(req, timeout=6) as response:
            body = response.read()
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError(f'Could not decode JPEG from {address}')
        return img

    def close(self):
        if self.cap is not None: self.cap.release()


class SimCamera:
    """Synthetic phone and screen geometry: validates pipeline, not inversion."""
    def __init__(self, zone: Zone, ae: bool=False, seed: int=9):
        s = zone.corners()
        d = np.array([[145, 105], [635, 119], [603, 467], [160, 447]],np.float32)
        self.h_true = cv2.getPerspectiveTransform(s, d)
        self.rng = np.random.default_rng(seed)
        self.ae = ae
        self.t = 0
        self.zone = zone
        self.camera_size = (800, 570)

    def capture(self, pattern: np.ndarray|None=None, inserted: bool=False) -> np.ndarray:
        if pattern is None: pattern = self.zone.base()
        warped = cv2.warpPerspective(pattern, self.h_true, self.camera_size,
                                     flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if inserted:
            xx, yy = np.meshgrid(np.arange(800), np.arange(570))
            shadow = np.exp(-.5*(((xx-390)/64)**2+((yy-280)/52)**2))
            gray *= (1-.70*shadow)
        gray = cv2.GaussianBlur(gray,(0,0),1.5)
        mean = gray.mean()/255.
        gain = (0.65 / (0.42+mean)) if self.ae else .78
        signal = 9.0 + gain * gray
        noise = self.rng.normal(0,.9,gray.shape)
        signal = np.uint8(np.clip(signal+noise,0,255))
        self.t += 1
        return cv2.cvtColor(signal,cv2.COLOR_GRAY2BGR)

    def close(self):
        pass


def capture_after(screen: Screen, cam, pattern: np.ndarray, settle:float,
                  samples:int, inserted:bool=False) -> np.ndarray:
    screen.show(pattern)
    if not screen.headless: time.sleep(settle)
    # For /shot.jpg these are new HTTP snapshots, not stale buffered frames.
    imgs = [cam.capture(pattern, inserted) for _ in range(samples)]
    if any(x.shape != imgs[0].shape for x in imgs):
        raise RuntimeError('Camera changed frame dimensions during calibration')
    return np.uint8(np.median(np.stack(imgs),axis=0))


def luminance(img: np.ndarray) -> np.ndarray:
    # Camera-reported encoded pixel intensity, NOT physical linear radiance.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def dot_centroid(frame: np.ndarray, dark: np.ndarray) -> tuple[tuple[float,float],float]:
    delta = luminance(frame) - luminance(dark)
    sm = cv2.GaussianBlur(delta,(0,0),2.)
    peak=float(sm.max())
    if peak < 9:
        raise ValueError(f'No reliable illuminated spot (peak diff {peak:.1f} camera levels)')
    mask=np.uint8(sm>max(8,peak*.44))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask,8)
    if n < 2: raise ValueError('Illuminated spot not detected')
    choices = [(int(stats[i,cv2.CC_STAT_AREA]),i) for i in range(1,n)]
    area,idx=max(choices)
    if area < 10: raise ValueError(f'Illuminated spot too small ({area} px)')
    # Intensity weighted centre within dominant connected component.
    weighted=np.maximum(sm,0)*(labels==idx)
    m=cv2.moments(weighted)
    if m['m00'] < 1e-9: raise ValueError('Spot had zero weight')
    return (float(m['m10']/m['m00']),float(m['m01']/m['m00'])),peak


def fit_geometry(zone:Zone, uv:list[tuple[float,float]], points:list[tuple[float,float]]):
    src=np.array([zone.xy(p) for p in uv],np.float32)
    dst=np.array(points,np.float32)
    H, inliers=cv2.findHomography(src,dst,cv2.RANSAC,4.)
    if H is None: raise ValueError('Cannot estimate screen-to-camera homography')
    pred=cv2.perspectiveTransform(src.reshape(-1,1,2),H).reshape(-1,2)
    errs=np.linalg.norm(pred-dst,axis=1)
    rmse=float(np.sqrt(np.mean(errs**2)))
    return H,rmse,errs,inliers.flatten().astype(bool)


def roi_mask(zone:Zone,H:np.ndarray, shape:tuple[int,...]):
    # Central ~42% of the light table: avoid frame edges and falloff.
    ps=np.array([zone.xy(p) for p in ((.29,.29),(.71,.29),(.71,.71),(.29,.71))],np.float32)
    projected=cv2.perspectiveTransform(ps.reshape(-1,1,2),H).reshape(-1,2)
    mask=np.zeros(shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.int32(np.rint(projected)),255)
    if np.count_nonzero(mask)<200:raise ValueError('Mapped capture zone is too small')
    return mask


def summarize_levels(samples:list[dict]):
    # Tests repeatability. An auto-exposure lock cannot be proved from JPEG levels.
    by_level={}
    for x in samples:by_level.setdefault(x['screen_level'],[]).append(x['camera_mean'])
    base=np.mean(by_level[0.0]); top=np.mean(by_level[1.0]);span=top-base
    flags=[]
    if span<18:flags.append('INSUFFICIENT_RESPONSE: light table too dim or camera too far')
    if span<=0: flags.append('REVERSED_BRIGHTNESS')
    repeated={}
    for lv,vals in by_level.items():
        if len(vals)>1:
            deviation=(max(vals)-min(vals))/max(abs(span),1)
            repeated[str(lv)]=round(float(deviation),4)
            if deviation>.07: flags.append(f'DRIFT_OR_AUTO_EXPOSURE: repeated {lv:g} differs by {deviation:.1%} of response span')
    # Mid-to-high ordering must remain monotonically increasing in a fixed camera.
    means={lv:float(np.median(v)) for lv,v in by_level.items()}
    levels=sorted(means)
    if any(means[b]+3<means[a] for a,b in zip(levels,levels[1:])):
        flags.append('NON_MONOTONIC_BRIGHTNESS')
    if any(x['clipped_fraction']>.02 for x in samples if x['screen_level']>=.75):
        flags.append('CAMERA_CLIPPING: reduce ISO or shutter exposure')
    return {'dark_level':round(float(base),3),'white_level':round(float(top),3),
            'camera_response_span':round(float(span),3),'repeat_deviation':repeated,
            'response_curve':[{'screen_level':float(lv),'camera_mean':round(means[lv],3)} for lv in levels],
            'warnings':flags,'stable':not flags}


def pattern_grid(n:int):
    return [(float(x),float(y)) for y in np.linspace(.16,.84,n) for x in np.linspace(.16,.84,n)]


def write_image(path:Path,img:np.ndarray):
    if not cv2.imwrite(str(path),img): raise RuntimeError(f'Could not write {path}')


def stage_geometry(zone, screen,cam,out,settle,samples):
    print('\nSTAGE 1 / Screen-to-camera mapping (nothing between phone and screen)')
    dark = capture_after(screen,cam,zone.base(),settle,samples)
    uv=pattern_grid(3)
    detections=[]; peaks=[]
    for i,p in enumerate(uv,1):
        frame=capture_after(screen,cam,zone.dot(p),settle,samples)
        xy,peak=dot_centroid(frame,dark)
        detections.append(xy);peaks.append(peak)
        print(f'  illuminated point {i:02d}/09: camera pixel ({xy[0]:.1f}, {xy[1]:.1f}), contrast {peak:.1f}')
    H,rmse,errors,inliers=fit_geometry(zone,uv,detections)
    if rmse>7 or sum(inliers)<7:
        raise RuntimeError(f'Geometry unreliable (RMSE {rmse:.1f} px, {sum(inliers)}/9 inliers). Refocus, straighten phone or reduce reflections.')
    preview=cam.capture(zone.board()) if screen.headless else capture_after(screen,cam,zone.board(),settle,samples)
    overlay=preview.copy()
    crns=cv2.perspectiveTransform(zone.corners().reshape(-1,1,2),H).reshape(-1,2)
    cv2.polylines(overlay,[np.int32(np.rint(crns))],True,(0,255,170),2)
    for i,xy in enumerate(detections):
        cv2.circle(overlay,tuple(np.int32(np.rint(xy))),5,(0,60,255),-1)
    write_image(out/'01_geometry_overlay.png',overlay)
    write_image(out/'01_empty_screen.png',dark)
    print(f'  geometry RMSE: {rmse:.2f} camera pixels; inliers {sum(inliers)}/9')
    return H,rmse,errors,peaks


def stage_exposure(zone,screen,cam,H,out,settle,samples):
    print('\nSTAGE 2 / Camera response & repeatability (no object yet)')
    mask=roi_mask(zone,H,cam.capture(zone.base()).shape)
    levels=[0., .25, .5, .75, 1., .5, .25, .75, 0., .5, 1., 0.]
    result=[]
    for lv in levels:
        frame=capture_after(screen,cam,zone.full(lv),settle,samples)
        values=luminance(frame)[mask>0]
        mean=float(np.mean(values))
        clipped=float(np.mean(values>=250))
        result.append({'screen_level':lv,'camera_mean':mean,'clipped_fraction':clipped})
        print(f'  screen {lv:4.0%} -> camera {mean:6.1f}/255, clipped {clipped:.2%}')
    resp=summarize_levels(result)
    print('  '+('STABILITY CHECK PASSED (not proof of manual lock)' if resp['stable'] else 'WARNINGS: '+', '.join(resp['warnings'])))
    return result, resp


def diff_montage(a:list[np.ndarray], b:list[np.ndarray]):
    # Mean of abs signal changes, preserving spatial location in camera frame.
    x=np.mean(np.stack([luminance(v) for v in a]),axis=0)
    y=np.mean(np.stack([luminance(v) for v in b]),axis=0)
    delta=np.maximum(0.,x-y)
    mx=max(float(np.percentile(delta,99.5)),1.)
    norm=np.uint8(np.clip(delta/mx,0,1)*255)
    hot_fraction = float(np.mean(delta > max(1.5, mx*0.35)))
    return cv2.applyColorMap(norm,cv2.COLORMAP_TURBO), float(np.mean(delta)),mx,hot_fraction


def stage_scan(zone,screen,cam,out,settle,samples,sim):
    """Optional real *2D differential* imaging with a fixed diffuser; no depth claim."""
    print('\nSTAGE 3 / Differential shadow measurement (2D, NOT CT)')
    if not sim:
        screen.show(zone.base())
        input('Install BLANK diffuser only. Keep phone fixed. Press Enter to capture 4x4 reference... ')
    pts=pattern_grid(4)
    base=[]; obj=[]
    for i,p in enumerate(pts,1):
        base.append(capture_after(screen,cam,zone.dot(p),settle,samples,inserted=False))
        print(f'  blank {i:02d}/{len(pts)}')
    if not sim:
        screen.show(zone.base())
        input('Insert hidden object WITHOUT moving camera or diffuser. Press Enter to rescan... ')
    for i,p in enumerate(pts,1):
        obj.append(capture_after(screen,cam,zone.dot(p),settle,samples,inserted=True))
        print(f'  object {i:02d}/{len(pts)}')
    heat,change,display_scale,hot_fraction=diff_montage(base,obj)
    write_image(out/'03_shadow_difference.png',heat)
    write_image(out/'03_blank_mean.png',np.uint8(np.mean(np.stack(base),axis=0)))
    write_image(out/'03_object_mean.png',np.uint8(np.mean(np.stack(obj),axis=0)))
    print(f'  Mean attenuation change: {change:.2f} camera levels; heatmap scale {display_scale:.2f}; hot pixels {hot_fraction:.1%}')
    return {'mean_camera_level_difference':change,'display_scale':display_scale,'hot_pixel_fraction':hot_fraction,
            'interpretation':'Differential 2D transmission/shadow signal only. NOT bead-depth or 3D reconstruction.'}


def screen_dimensions(args):
    if args.sim:
        return 960,720
    if args.screen_width and args.screen_height:
        return args.screen_width,args.screen_height
    try:
        import tkinter
        root=tkinter.Tk();root.withdraw()
        w,h=root.winfo_screenwidth(),root.winfo_screenheight();root.destroy()
        return w,h
    except Exception as exc:
        raise RuntimeError('Could not detect monitor size; pass --screen-width and --screen-height') from exc


def rig_preview(zone:Zone,screen:Screen,cam):
    """One-monitor setup guide: board on monitor, small live camera preview outside zone.

    Setup preview is intentionally destroyed before radiometric measurements.
    """
    name='LightScan | phone preview: Space=calibrate, Q=quit'
    screen.show(zone.board())
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    target_width=min(410,max(240,zone.x0-40))
    cv2.resizeWindow(name,target_width,300)
    cv2.moveWindow(name,20,25)
    print('PREVIEW: adjust phone on stand; inspect focus and framing. Space starts calibration; Q exits.')
    try:
        while True:
            img=cam.capture(zone.board())
            scale=target_width/max(img.shape[1],1)
            disp=cv2.resize(img,(target_width,max(1,int(img.shape[0]*scale))))
            cv2.putText(disp,'SPACE: calibrate | Q: quit',(7,22),cv2.FONT_HERSHEY_SIMPLEX,
                        .55,(0,230,235),2,cv2.LINE_AA)
            cv2.imshow(name,disp)
            k=cv2.waitKey(20)&255
            if k in (ord('q'),27): raise KeyboardInterrupt('Preview cancelled')
            if k in (32,13): break
    finally:
        cv2.destroyWindow(name)
        screen.show(zone.base())


def execute(args):
    w,h=screen_dimensions(args)
    size=min(args.zone_px,w-60,h-60)
    if size < 220: raise ValueError('Light zone must be at least 220 screen pixels')
    zone=Zone(w,h,size)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    screen=Screen(zone,args.headless,(args.origin_x,args.origin_y))
    cam=SimCamera(zone,ae=args.sim_auto_exposure) if args.sim else IPCamera(args.url,args.camera_mode)
    try:
        if args.preview and not screen.headless:
            rig_preview(zone,screen,cam)
        H,rmse,errors,peaks=stage_geometry(zone,screen,cam,out,args.settle,args.samples)
        raw,resp=stage_exposure(zone,screen,cam,H,out,args.settle,args.samples)
        report={
            'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'mode':'simulated' if args.sim else 'physical',
            'camera_source':'synthetic' if args.sim else ('IP Webcam '+args.camera_mode),
            'screen_size_px':[w,h],'zone_px':asdict(zone),
            'screen_to_camera_homography':H.tolist(),
            'geometry_rmse_camera_px':rmse,'geometry_errors_px':errors.tolist(),
            'spot_peaks_camera_levels':peaks,
            'camera_response':raw,'stability':resp,
            'notes':['JPEG pixel values are not calibrated linear radiance.',
                     'A homography maps a SCREEN PLANE, not the depth of a hidden object.',
                     'The camera must be manually locked in the Android camera application; OpenCV cannot reliably do that over IP Webcam.',
                     'Calibration is invalid after moving camera, changing focus, resolution, exposure or screen settings.']}
        if args.scan:
            report['shadow_scan']=stage_scan(zone,screen,cam,out,args.settle,args.samples,args.sim)
        screen.show(zone.base())
        (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(f'\nSaved: {out.resolve() / "calibration.json"}')
        print('STAGE 1+2 COMPLETED. This is not yet 3D scanning.')
        return report
    finally:
        cam.close()
        screen.close()


def parse_args():
    ap=argparse.ArgumentParser(description='LightScan calibration and differential screen probe')
    ap.add_argument('--url',default='',help='Phone IP Webcam base URL on YOUR local network (required without --sim)')
    ap.add_argument('--camera-mode',choices=['shot','video'],default='shot',help='/shot.jpg recommended; /video fallback')
    ap.add_argument('--zone-px',type=int,default=540,help='Size of central square light field in SCREEN pixels. Measure actual cm separately.')
    ap.add_argument('--screen-width',type=int)
    ap.add_argument('--screen-height',type=int)
    ap.add_argument('--origin-x',type=int,default=0)
    ap.add_argument('--origin-y',type=int,default=0)
    ap.add_argument('--settle',type=float,default=.65,help='Wait between pattern change and camera sample; increase for IP/network lag')
    ap.add_argument('--samples',type=int,default=2,help='Frames to median per pattern')
    ap.add_argument('--output',default='runs/rig_001')
    ap.add_argument('--scan',action='store_true',help='Optional 4x4 blank vs object differential 2D probe')
    ap.add_argument('--preview',action='store_true',help='Interactive live camera preview on screen before calibrating')
    ap.add_argument('--sim',action='store_true',help='Synthetic phone feed; does not test real camera')
    ap.add_argument('--sim-auto-exposure',action='store_true',help='Deliberately faulty simulated exposure response')
    ap.add_argument('--headless',action='store_true',help='For simulation tests/CI; no GUI or actual display')
    return ap.parse_args()


if __name__=='__main__':
    args=parse_args()
    if not args.sim and not args.url:
        raise SystemExit('Pass the phone stream base address using --url http://PHONE_IP:8080/')
    if args.headless and not args.sim:
        raise SystemExit('--headless requires --sim (physical patterns need a real display)')
    if not args.sim:
        print('REAL CAPTURE: lock focus/exposure/white balance on PHONE before starting; use dim room.')
        print('No object during stage 1 or stage 2. Phone and monitor must not move.')
        print('Camera network access is LOCAL to this Python process; no images uploaded.')
    try:
        execute(args)
    except KeyboardInterrupt:
        print('\nCalibration cancelled, no changes to camera.')
    except Exception as e:
        print(f'\nCALIBRATION FAILED: {e}')
        raise
