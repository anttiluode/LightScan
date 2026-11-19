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

    def structured(self, index: int, level: float = 0.85) -> np.ndarray:
        """A 4x4 Walsh/Hadamard illumination pattern, one full, 15 half-lit.

        Macrocell 0..15 is illuminated when the parity of index & cell is even.
        The 15 non-DC patterns each illuminate exactly half the active square.
        Their complementary sets make different hidden regions observable.
        """
        if not (0 <= index < 16):
            raise ValueError('Structured pattern index must be 0..15')
        img = self.base()
        bright = int(round(255 * float(np.clip(level, 0, 1))))
        # Partition the active area without gaps or border spill.
        for cell in range(16):
            if ((index & cell).bit_count() & 1) == 0:
                row, col = divmod(cell, 4)
                x1 = self.x0 + (col*self.size)//4
                x2 = self.x0 + ((col+1)*self.size)//4
                y1 = self.y0 + (row*self.size)//4
                y2 = self.y0 + ((row+1)*self.size)//4
                img[y1:y2, x1:x2] = bright
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
    """Full-screen light source with guidance kept ABOVE the calibrated zone.

    The screen is the illumination instrument: do not write UI into the light
    square while measuring. A stable dark banner lives entirely outside it.
    """
    def __init__(self, zone: Zone, headless: bool, origin: tuple[int,int]=(0,0)):
        self.zone = zone
        self.headless = headless
        self.name = 'LightScan | illumination screen (Esc to abort)'
        self.current = zone.base()
        self.title = 'LIGHTSCAN  |  READY'
        self.detail = 'Keep phone and monitor fixed.'
        self.progress = ''
        if not headless:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.name, *origin)
            cv2.resizeWindow(self.name, zone.width, zone.height)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def guide(self, title: str, detail: str='', progress: str=''):
        self.title, self.detail, self.progress = title, detail, progress

    @staticmethod
    def _text(image, msg, x, baseline, max_width, scale=.65, color=(230,240,246), thickness=1):
        """Fit messages on narrow monitors without a new GUI dependency."""
        if not msg or max_width <= 0:
            return
        face = cv2.FONT_HERSHEY_SIMPLEX
        actual_width = cv2.getTextSize(msg, face, scale, thickness)[0][0]
        if actual_width > max_width:
            scale *= max_width/max(1, actual_width)
        cv2.putText(image, msg, (int(x),int(baseline)), face, scale, color,
                    thickness, cv2.LINE_AA)

    def guided_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overlay guidance outside the illuminated square; don't mutate the pattern."""
        image = frame.copy()
        if self.headless:
            return image
        # The effective zone size reserves at least 90 px above the square.
        band = min(82, self.zone.y0-6)
        if band < 48:
            raise ValueError('Illumination zone leaves no space for guidance banner')
        # Black background and dim text reduce stray illumination from the UI.
        # Most importantly, the banner must NOT change between comparable frames.
        cv2.rectangle(image,(0,0),(self.zone.width-1,band),(0,0,0),-1)
        side=22
        self._text(image, self.title, side, 27, self.zone.width-2*side,
                   .68, (72,120,125),1)
        self._text(image, self.detail, side, 51, self.zone.width-2*side,
                   .52, (105,105,105),1)
        self._text(image, self.progress or 'ESC: cancel', side, 71,
                   self.zone.width-2*side, .45, (79,79,79),1)
        return image

    def show(self, frame: np.ndarray):
        self.current = frame
        if not self.headless:
            cv2.imshow(self.name, self.guided_frame(frame))
            k = cv2.waitKey(40) & 255
            if k in (27,ord('q')):
                raise KeyboardInterrupt('Escape/Q pressed')

    def wait_for_space(self, title: str, instructions: list[str]):
        """Visible, keyboard-controlled pause. Never block on an obscured terminal."""
        if self.headless:
            return
        self.guide(title, 'Follow the steps in the middle of the screen.',
                   'CLICK this window if needed, then press SPACE | ESC cancels')
        image=self.zone.base()
        x0, y0, size=self.zone.x0,self.zone.y0,self.zone.size
        margin=max(10,int(size*.05))
        left=x0+margin; right=x0+size-margin
        top=y0+int(size*.14); bottom=y0+int(size*.85)
        cv2.rectangle(image,(left,top),(right,bottom),(25,39,50),-1)
        cv2.rectangle(image,(left,top),(right,bottom),(82,154,168),2)
        w=right-left-30
        self._text(image,'WHAT TO DO',left+15,top+33,w,.68,(133,226,239),2)
        gap=max(28,int((bottom-top-92)/max(1,len(instructions))))
        for i,line in enumerate(instructions):
            self._text(image,line,left+15,top+70+i*gap,w,.65,(244,244,245),1)
        self._text(image,'SPACE / ENTER: continue',left+15,bottom-22,w,
                   .57,(128,223,164),2)
        self.show(image)
        while True:
            key=cv2.waitKey(60)&255
            if key in (32,13):return
            if key in (27,ord('q')):raise KeyboardInterrupt('User cancelled scan')

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
    screen.guide('STEP 1 / 3  |  FIND THE SCREEN',
                 'Nothing between phone and monitor. Do not move either.',
                 'Measuring 9 light positions automatically...')
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
    screen.guide('STEP 2 / 3  |  CAMERA BRIGHTNESS',
                 'Keep the setup EMPTY and still. Light levels will change.',
                 'Checking exposure stability; automatic, no action needed...')
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


def scan_region(zone:Zone,H:np.ndarray,shape:tuple[int,...]):
    """Mask the projected full illumination square, allowing diffuser blur.

    The saved homography is the MONITOR PLANE, not the diffuser plane; this is
    a coarse measurement ROI, not a physical registration of a hidden object.
    """
    corners=cv2.perspectiveTransform(zone.corners().reshape(-1,1,2),H).reshape(-1,2)
    center=np.mean(corners,axis=0)
    expanded=center+(corners-center)*1.12
    mask=np.zeros(shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.int32(np.rint(expanded)),255)
    pixels=int(np.count_nonzero(mask))
    if pixels<1000:
        raise ValueError('Saved calibration projects outside current camera frame; recheck the rig')
    inside=sum(0 <= x < shape[1] and 0 <= y < shape[0] for x,y in corners)
    if inside < 4:
        raise ValueError(f'Illumination zone ({zone.size} screen px) is cropped by the {shape[1]}x{shape[0]} camera frame. Calibration points can still pass because they lie INSIDE the square. Rerun using --zone-px 420, or --zone-px 360 if needed; recalibrate for each zone size.')
    return mask.astype(bool)


def load_calibration(path:Path,zone:Zone,camera_shape:tuple[int,...]):
    """Reuse saved geometry at a different MANUAL exposure, never photometry.

    Fails on resolution, screen or zone changes. Physical movement/focus changes
    cannot be detected perfectly and must be verified by the operator.
    """
    report=json.loads(path.read_text(encoding='utf8'))
    if report.get('schema_version')!=1:
        raise ValueError('Unknown calibration schema')
    if report.get('screen_size_px')!=[zone.width,zone.height]:
        raise ValueError('Screen resolution changed since calibration')
    z=report.get('zone_px',{})
    if [z.get(k) for k in ('width','height','size')]!=[zone.width,zone.height,zone.size]:
        raise ValueError('Calibration zone differs; match --zone-px to the original run')
    if 'camera_size_px' in report and report['camera_size_px'] != [camera_shape[1],camera_shape[0]]:
        raise ValueError('Phone camera resolution changed since calibration')
    H=np.asarray(report.get('screen_to_camera_homography'),dtype=np.float64)
    if H.shape!=(3,3) or not np.all(np.isfinite(H)):
        raise ValueError('Invalid saved screen-to-camera homography')
    _=scan_region(zone,H,camera_shape)
    return H,report


def probe_metrics(blank,object_frames,dark_blank,dark_object,roi,blank_repeat,
                  object_repeat):
    """Contrast in the useful camera ROI, not diluted across the entire frame.

    JPEG camera levels are nonlinear. Dark subtraction is diagnostic only;
    this does NOT establish absolute optical attenuation or radiance.
    """
    a=np.stack([luminance(im) for im in blank]).astype(np.float32)
    b=np.stack([luminance(im) for im in object_frames]).astype(np.float32)
    db=luminance(dark_blank)
    do=luminance(dark_object)
    # Keep signed differences: white and black camera offsets may shift.
    aa=a-db[None]
    bb=b-do[None]
    delta=aa-bb
    signed=np.mean(delta,axis=0)
    mask=roi
    # Average signed changes for each structured illumination.
    per_pattern=[]
    for k in range(len(blank)):
        dd=delta[k][mask]
        v=a[k][mask]
        per_pattern.append({
            'index':k,
            'blank_mean_camera_level':round(float(np.mean(v)),3),
            'blank_signal_above_dark':round(float(np.mean(aa[k][mask])),3),
            'object_signal_above_dark':round(float(np.mean(bb[k][mask])),3),
            'mean_blank_minus_object':round(float(np.mean(dd)),3),
            'rms_difference':round(float(np.sqrt(np.mean(dd**2))),3),
            'p95_absolute_difference':round(float(np.percentile(np.abs(dd),95)),3),
            'clipped_fraction_blank':round(float(np.mean(v>=250)),5)})
    reference_noise=np.sqrt(np.mean((luminance(blank[0])[mask]-
                                     luminance(blank_repeat)[mask])**2))
    object_noise=np.sqrt(np.mean((luminance(object_frames[0])[mask]-
                                  luminance(object_repeat)[mask])**2))
    repeat_rms=float(max(reference_noise,object_noise))
    # Signal/noise comparison uses repeat controls, not an arbitrary threshold.
    # Red = illumination was reduced by the object; blue = illumination grew.
    # Keep signed changes instead of silently erasing inverse-scattering effects.
    scale=max(float(np.percentile(np.abs(signed[mask]),99.5)),1.)
    image=np.zeros((*mask.shape,3),np.uint8)
    image[...,2][mask]=np.uint8(255*np.clip(signed[mask]/scale,0.,1.))
    image[...,0][mask]=np.uint8(255*np.clip(-signed[mask]/scale,0.,1.))
    avg=np.mean(a,axis=0)
    obj_avg=np.mean(b,axis=0)
    effective_diff=float(np.sqrt(np.mean(np.mean(delta,axis=0)[mask]**2)))
    return image, {
        'roi_pixels':int(np.count_nonzero(mask)),
        'blank_dark_mean_camera_level':round(float(np.mean(db[mask])),3),
        'object_dark_mean_camera_level':round(float(np.mean(do[mask])),3),
        'mean_blank_camera_level_in_roi':round(float(np.mean(a[:,mask])),3),
        'mean_object_camera_level_in_roi':round(float(np.mean(b[:,mask])),3),
        'mean_signed_difference_in_roi':round(float(np.mean(delta[:,mask])),3),
        'mean_absolute_difference_in_roi':round(float(np.mean(np.abs(delta[:,mask]))),3),
        'rms_mean_difference_in_roi':round(effective_diff,3),
        'repeat_rms_camera_levels':round(repeat_rms,3),
        'rms_signal_to_repeat_ratio':round(effective_diff/max(repeat_rms,0.15),2),
        'max_blank_clipped_fraction':round(float(max(x['clipped_fraction_blank'] for x in per_pattern)),5),
        'patterns':per_pattern,
        'interpretation':('2D camera differential signal from structured illuminations; '
            'ROI signal and repeat noise only. NOT quantitative optical attenuation, '
            'not an inverse reconstruction, not 3D CT.')},avg,obj_avg


def stage_scan(zone,screen,cam,out,settle,samples,sim,H,pattern_level=.85,save_frames=False):
    """Strong 4x4 Hadamard probes, repeated-white controls, explicit camera ROI."""
    print('\nSTAGE 3 / Strong structured-light differential scan (2D, NOT CT)')
    pts=[zone.structured(i,pattern_level) for i in range(16)]
    if not sim:
        screen.wait_for_space('STAGE 3 / 3  |  BLANK DIFFUSER', [
            'Install BLANK diffuser; NO hidden shape.',
            'Camera and diffuser must stay fixed.',
            'Use manual ISO/exposure; do not change them during scan.',
            'Press SPACE when ready.'])
    screen.guide('STAGE 3 / 3  |  MEASURING BLANK',
                 'Bright structured patterns. Do not move the diffuser.',
                 'Recording blank and repeated white reference...')
    dark_blank=capture_after(screen,cam,zone.base(),settle,samples,inserted=False)
    roi=scan_region(zone,H,dark_blank.shape)
    blank=[]
    for k,pattern in enumerate(pts):
        frame=capture_after(screen,cam,pattern,settle,samples,inserted=False)
        blank.append(frame)
        v=luminance(frame)[roi]
        if k in (0,1,7,15):
            print(f'  blank {k+1:02d}/16 | ROI mean {np.mean(v):.1f}/255 '
                  f'| max {np.max(v):.0f} | clipped {(v>=250).mean():.2%}')
    blank_repeat=capture_after(screen,cam,pts[0],settle,samples,inserted=False)
    if not sim:
        screen.wait_for_space('STAGE 3 / 3  |  INSERT HIDDEN OBJECT', [
            'Keep the diffuser and camera FIXED.',
            'Put BLACK shape behind the diffuser.',
            'Do not change ISO, shutter, focus or monitor settings.',
            'Press SPACE to compare against the blank frames.'])
    screen.guide('STAGE 3 / 3  |  MEASURING OBJECT',
                 'Do not move camera or diffuser; scanning 16 broad patterns.',
                 'Recording object and repeated white reference...')
    dark_object=capture_after(screen,cam,zone.base(),settle,samples,inserted=True)
    object_frames=[]
    for k,pattern in enumerate(pts):
        frame=capture_after(screen,cam,pattern,settle,samples,inserted=True)
        object_frames.append(frame)
        v=luminance(frame)[roi]
        if k in (0,1,7,15):
            print(f'  object {k+1:02d}/16 | ROI mean {np.mean(v):.1f}/255 '
                  f'| max {np.max(v):.0f} | clipped {(v>=250).mean():.2%}')
    object_repeat=capture_after(screen,cam,pts[0],settle,samples,inserted=True)
    heat,report,blank_mean,obj_mean=probe_metrics(blank,object_frames,dark_blank,
                              dark_object,roi,blank_repeat,object_repeat)
    write_image(out/'03_shadow_difference.png',heat)
    write_image(out/'03_blank_mean.png',np.uint8(blank_mean))
    write_image(out/'03_object_mean.png',np.uint8(obj_mean))
    write_image(out/'03_roi_mask.png',np.uint8(roi)*255)
    if save_frames:
        frames=out/'frames'; frames.mkdir(exist_ok=True)
        for prefix,images in [('blank',blank),('object',object_frames)]:
            for k,img in enumerate(images):write_image(frames/f'{prefix}_{k:02d}.png',img)
        for label,img in [('blank_dark',dark_blank),('object_dark',dark_object),
                           ('blank_repeat',blank_repeat),('object_repeat',object_repeat)]:
            write_image(frames/f'{label}.png',img)
        # Source patterns are generated deterministically from index/level/zone.
    print(f'  Blank ROI camera mean: {report["mean_blank_camera_level_in_roi"]:.2f}/255')
    print(f'  Blank-minus-object mean (ROI): {report["mean_signed_difference_in_roi"]:.2f} camera levels')
    print(f'  Mean ABS difference (ROI): {report["mean_absolute_difference_in_roi"]:.2f} levels')
    print(f'  Repeat white-pattern RMS: {report["repeat_rms_camera_levels"]:.2f} levels')
    print(f'  Signal / repeat RMS: {report["rms_signal_to_repeat_ratio"]:.2f} (diagnostic only)')
    if report['max_blank_clipped_fraction']>.02:
        print('  WARNING: >2% clipping in blank; lower ISO or exposure and rerun scan')
    if report['mean_blank_camera_level_in_roi']<20:
        print('  WARNING: blank scan is very dim; increase light safely, avoid clipping')
    return report


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
    screen.guide('SETUP  |  PHONE PREVIEW',
                 'Adjust the phone image, then press SPACE in the small preview.',
                 'Q or ESC cancels. Keep the space between phone and screen empty.')
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
    # Keep the effective size/zone deterministic across calibration and scan.
    size=min(args.zone_px,w-60,h-180)
    if size<220:raise ValueError('Light zone must be at least 220 screen pixels')
    zone=Zone(w,h,size)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    screen=Screen(zone,args.headless,(args.origin_x,args.origin_y))
    cam=SimCamera(zone,ae=args.sim_auto_exposure) if args.sim else IPCamera(args.url,args.camera_mode)
    try:
        if args.preview and not screen.headless and not args.scan_only:
            rig_preview(zone,screen,cam)
        if args.scan_only:
            if not args.calibration:
                raise ValueError('--scan-only requires --calibration path/to/calibration.json')
            frame=cam.capture(zone.base())
            H,cal=load_calibration(Path(args.calibration),zone,frame.shape)
            print(f'REUSED SAVED GEOMETRY from {args.calibration}; no naked-screen calibration.')
            print('Only geometry reused. Prior ISO/exposure brightness response is INVALID now.')
            print('Verify phone/resolution/zoom/focus/position have NOT changed.')
            report={
                'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
                'mode':'simulated' if args.sim else 'physical',
                'camera_source':'synthetic' if args.sim else ('IP Webcam '+args.camera_mode),
                'screen_size_px':[w,h],'zone_px':asdict(zone),
                'camera_size_px':[frame.shape[1],frame.shape[0]],
                'screen_to_camera_homography':H.tolist(),
                'geometry_source':str(args.calibration),
                'geometry_rmse_camera_px':cal.get('geometry_rmse_camera_px'),
                'notes':['Reused monitor-plane homography only. Current exposure is NOT calibrated.',
                         'Do NOT infer depth or absolute optical absorption from this scan.']}
        else:
            H,rmse,errors,peaks=stage_geometry(zone,screen,cam,out,args.settle,args.samples)
            raw,resp=stage_exposure(zone,screen,cam,H,out,args.settle,args.samples)
            frame=cam.capture(zone.base())
            report={
                'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
                'mode':'simulated' if args.sim else 'physical',
                'camera_source':'synthetic' if args.sim else ('IP Webcam '+args.camera_mode),
                'screen_size_px':[w,h],'zone_px':asdict(zone),
                'camera_size_px':[frame.shape[1],frame.shape[0]],
                'screen_to_camera_homography':H.tolist(),
                'geometry_rmse_camera_px':rmse,'geometry_errors_px':errors.tolist(),
                'spot_peaks_camera_levels':peaks,
                'camera_response':raw,'stability':resp,
                'notes':['JPEG pixel values are not calibrated linear radiance.',
                         'A homography maps a SCREEN PLANE, not the depth of a hidden object.',
                         'The camera must be manually locked in the Android camera application.',
                         'Calibration invalid after moving camera, changing focus, zoom, resolution or screen geometry.']}
        if args.scan or args.scan_only:
            report['shadow_scan']=stage_scan(zone,screen,cam,out,args.settle,args.samples,
                            args.sim,H,args.pattern_level,args.save_frames)
        screen.guide('FINISHED  |  SAVING RESULTS','Writing calibration and scan files...',
                     'The screen will show a completion message.')
        screen.show(zone.base())
        (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(f'\nSaved: {out.resolve() / "calibration.json"}')
        print('STAGE 3 COMPLETED. Differential measurement, NOT 3D scanning.' if (args.scan or args.scan_only) else
              'STAGES 1-2 COMPLETED. No inverse reconstruction yet.')
        if not screen.headless:
            if args.scan or args.scan_only:
                info=report['shadow_scan']
                summary=[f'Blank mean: {info["mean_blank_camera_level_in_roi"]:.1f} / 255',
                         f'Abs diff: {info["mean_absolute_difference_in_roi"]:.2f} | repeat RMS: {info["repeat_rms_camera_levels"]:.2f}',
                         'Shadow images + report saved. NOT a 3D reconstruction.']
            else:
                summary=[f'Geometry: {report["geometry_rmse_camera_px"]:.2f} px',
                         'Calibration files saved.','This does NOT reconstruct 3D.']
            screen.wait_for_space('FINISHED  |  RESULTS SAVED',summary)
        return report
    finally:
        cam.close();screen.close()


def parse_args():
    ap=argparse.ArgumentParser(description='LightScan calibration and differential screen probe')
    ap.add_argument('--url',default='',help='Phone IP Webcam base URL on YOUR local network (required without --sim)')
    ap.add_argument('--camera-mode',choices=['shot','video'],default='shot',help='/shot.jpg recommended; /video fallback')
    ap.add_argument('--zone-px',type=int,default=420,help='Square light field size in screen pixels (default: 420; reduce if the camera crops its corners).')
    ap.add_argument('--screen-width',type=int)
    ap.add_argument('--screen-height',type=int)
    ap.add_argument('--origin-x',type=int,default=0)
    ap.add_argument('--origin-y',type=int,default=0)
    ap.add_argument('--settle',type=float,default=.65,help='Wait between pattern change and camera sample; increase for IP/network lag')
    ap.add_argument('--samples',type=int,default=2,help='Frames to median per pattern')
    ap.add_argument('--output',default='runs/rig_001')
    ap.add_argument('--scan',action='store_true',help='Scan with 16 broad 4x4 Walsh/Hadamard patterns after fresh calibration')
    ap.add_argument('--scan-only',action='store_true',help='Skip bare-screen calibration; reuse saved homography and scan diffuser')
    ap.add_argument('--calibration',default='',help='Saved rig_locked/calibration.json for --scan-only (geometry only)')
    ap.add_argument('--pattern-level',type=float,default=.85,help='Peak pattern white level, 0.1..1 (reduce if clipping)')
    ap.add_argument('--save-frames',action='store_true',help='Save all raw blank/object frames for later inverse solver')
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
    if not .1<=args.pattern_level<=1.0:
        raise SystemExit('--pattern-level must be in 0.1..1.0')
    if not args.sim:
        print('REAL CAPTURE: lock focus/exposure/white balance on PHONE before starting; use dim room.')
        print('Scan-only mode: blank diffuser may already be installed.' if args.scan_only else
              'No object during stage 1 or stage 2. Phone and monitor must not move.')
        print('Camera network access is LOCAL to this Python process; no images uploaded.')
    try:
        execute(args)
    except KeyboardInterrupt:
        print('\nCalibration cancelled, no changes to camera.')
    except Exception as e:
        print(f'\nCALIBRATION FAILED: {e}')
        raise
