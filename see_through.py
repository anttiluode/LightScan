"""SEE THROUGH — a camera-only optical inverse for images shown behind a fixed diffuser.

No PyTorch, no 3D claims. It learns a measured linear inverse from known screen
patterns to camera images, and tests on physically captured images it did not fit.
The local IP Webcam stream is accessed by THIS Python process only.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Params:
    grid: int = 12
    features: int = 40
    zone: int = 420
    level: int = 235
    ridge: float = 0.055
    seed: int = 2026


def make_pattern(rng: np.random.Generator, grid: int, index: int) -> np.ndarray:
    """Mostly balanced, independently varied checker cells; occasional broader probes."""
    kind = index % 5
    if kind in (0, 1, 2):
        x = (rng.random((grid, grid)) >= .5).astype(np.uint8)
    elif kind == 3:
        c = max(2, grid // 3)
        small = (rng.random((c, c)) >= .5).astype(np.uint8)
        x = cv2.resize(small, (grid, grid), interpolation=cv2.INTER_NEAREST)
    else:
        x = np.zeros((grid, grid), np.uint8)
        for _ in range(2):
            a, b = map(int, rng.integers(0, grid, size=2))
            cv2.rectangle(x, (a,b), (min(grid-1,a+int(rng.integers(2,grid//2+2))),
                                  min(grid-1,b+int(rng.integers(2,grid//2+2)))),1,-1)
    return x


def glyph_pattern(text: str, grid: int) -> np.ndarray:
    """Hold-out text rendered at 5x screen resolution, reduced to the calibrated grid."""
    text = (text or 'HI').upper()[:8]
    w = grid * 8
    src = np.zeros((w, w), np.uint8)
    scale = min(1.9, (w * .88)/max(1, cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.,2)[0][0]))
    # A second size pass avoids truncation for long strings.
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(src,text,((w-tw)//2,(w+th)//2),cv2.FONT_HERSHEY_SIMPLEX,
                scale,255,max(2,int(round(scale*2))),cv2.LINE_AA)
    return (cv2.resize(src,(grid,grid),interpolation=cv2.INTER_AREA)>70).astype(np.uint8)


def screen_pixels(screen_w:int,screen_h:int,zone:int,grid_pattern:np.ndarray,
                  level:int=235) -> np.ndarray:
    """Only the central illumination zone changes; everything outside stays black."""
    out=np.zeros((screen_h,screen_w,3),np.uint8)
    h=min(zone,screen_h-160,screen_w-50)
    x=(screen_w-h)//2; y=(screen_h-h)//2
    pat=cv2.resize(grid_pattern,(h,h),interpolation=cv2.INTER_NEAREST)
    out[y:y+h,x:x+h]=np.repeat((pat * level)[:,:,None],3,axis=2)
    return out


def detect_roi(dark: np.ndarray, white: np.ndarray, *, cutoff:float=.16,
               minimum:int=8) -> tuple[int,int,int,int,dict]:
    """Locate the region changed by the controlled light square, not UI/browser graphics."""
    if dark.shape != white.shape:
        raise ValueError('Camera resolution changed during the black/white measurement')
    if dark.ndim==3:
        d=cv2.cvtColor(dark,cv2.COLOR_BGR2GRAY).astype(np.float32)
        w=cv2.cvtColor(white,cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        d=dark.astype(np.float32);w=white.astype(np.float32)
    diff=np.maximum(w-d,0)
    p90=float(np.percentile(diff,90));peak=float(np.percentile(diff,99.5))
    if peak<minimum:
        raise ValueError(f'Only {peak:.1f}/255 white-minus-black signal. '
                         'Raise manual exposure or thin the diffuser; no inversion possible yet.')
    smooth=cv2.GaussianBlur(diff,(0,0),5)
    thresh=max(4.,peak*cutoff)
    mask=(smooth>thresh).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((19,19),np.uint8))
    count,lab,stats,ctr=cv2.connectedComponentsWithStats(mask,8)
    if count<2:raise ValueError('Camera did not see a reliable light area')
    k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    x,y,wid,hei=map(int,stats[k,:4])
    if wid<15 or hei<15:raise ValueError('Visible illumination is too small')
    px=max(8,int(wid*.12));py=max(8,int(hei*.12))
    x0=max(0,x-px);y0=max(0,y-py)
    x1=min(diff.shape[1],x+wid+px);y1=min(diff.shape[0],y+hei+py)
    return (x0,y0,x1-x0,y1-y0),{'signal_p90':round(p90,2),
           'signal_p995':round(peak,2),'rect':[x,y,wid,hei],
           'roi':[x0,y0,x1-x0,y1-y0],
           'camera_size':[int(diff.shape[1]),int(diff.shape[0])]}


def extract(frame: np.ndarray, roi: tuple[int,int,int,int], size:int) -> np.ndarray:
    x,y,w,h=roi
    cut=frame[y:y+h,x:x+w]
    if cut.size==0:raise ValueError('Camera ROI falls outside the captured frame')
    if cut.ndim==3:cut=cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
    return cv2.resize(cut,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)


class Inverse:
    """Tikhonov-regularized direct inverse, solved in sample space for speed."""
    def __init__(self, reg: float=.055):
        self.reg=float(reg)

    def fit(self, measurements:np.ndarray, sources:np.ndarray):
        a=np.asarray(measurements,np.float64)
        b=np.asarray(sources,np.float64)
        if a.ndim!=2 or b.ndim!=2 or len(a)!=len(b) or len(a)<8:
            raise ValueError('Expected >=8 corresponding 2D camera/source samples')
        self.mean_a=a.mean(axis=0)
        self.mean_b=b.mean(axis=0)
        self.a=a-self.mean_a
        bb=b-self.mean_b
        kernel=self.a@self.a.T
        strength=max(1e-7,float(np.trace(kernel))/len(a)*self.reg)
        self.dual=np.linalg.solve(kernel+strength*np.eye(len(a)),bb)
        self.strength=strength
        return self

    def predict(self, measurements:np.ndarray) -> np.ndarray:
        if not hasattr(self,'a'):raise RuntimeError('Inverse is not fitted')
        x=np.asarray(measurements,np.float64)
        if x.ndim==1:x=x[None,:]
        if x.shape[1]!=self.a.shape[1]:raise ValueError('Incorrect camera feature dimension')
        return np.clip((x-self.mean_a)@self.a.T@self.dual+self.mean_b,0,1).astype(np.float32)

    def save(self,path:Path,params:Params,roi,black,white,stats):
        np.savez_compressed(path, mean_a=self.mean_a,mean_b=self.mean_b,
            a=self.a,dual=self.dual,reg=self.reg,strength=self.strength,
            roi=np.array(roi),dark=np.asarray(black,np.float32),
            white=np.asarray(white,np.float32),params=np.array(json.dumps(vars(params))),
            stats=np.array(json.dumps(stats)))

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as d:
            params=Params(**json.loads(str(d['params'])))
            model=cls(float(d['reg']))
            for key in ('mean_a','mean_b','a','dual'):
                setattr(model,key,d[key].copy())
            model.strength=float(d['strength'])
            roi=tuple(int(t) for t in d['roi'])
            dark=d['dark'].copy();white=d['white'].copy()
            stats=json.loads(str(d['stats']))
        return model,params,roi,dark,white,stats


def normalize(measured:np.ndarray,dark:np.ndarray,white:np.ndarray)->np.ndarray:
    """Global camera-level scaling. sRGB/JPEG is *not* calibrated irradiance."""
    scale=max(4.,float(np.percentile(white-dark,90)))
    return np.clip((measured-dark)/scale,-.3,1.5).astype(np.float32).ravel()


def baseline_fit(train_y:np.ndarray,train_x:np.ndarray, grid:int,feat:int):
    """A fair best-fit aligned-camera baseline using training samples only."""
    q=np.stack([cv2.resize(z.reshape(feat,feat),(grid,grid),interpolation=cv2.INTER_AREA).ravel()
               for z in train_y])
    ma=q.mean();mb=train_x.mean()
    slope=float(np.sum((q-ma)*(train_x-mb))/(np.sum((q-ma)**2)+1e-10))
    return slope,float(mb-slope*ma)


def baseline_predict(y:np.ndarray,coeff:tuple[float,float],grid:int,feat:int):
    a,b=coeff
    q=np.stack([cv2.resize(z.reshape(feat,feat),(grid,grid),interpolation=cv2.INTER_AREA).ravel()
                for z in np.asarray(y)])
    return np.clip(q*a+b,0,1)


def metrics(pred:np.ndarray,target:np.ndarray)->dict:
    mae=float(np.mean(np.abs(pred-target)))
    mse=float(np.mean((pred-target)**2))
    return {'mae':round(mae,4),'rmse':round(np.sqrt(mse),4)}


def synthetic_measure(source:np.ndarray,grid:int,features:int,rng:np.random.Generator,
                      sigma:float=6.,noise:float=1.5):
    """Synthetic diffusion with a weak broad halo; deliberately not a real optical model."""
    src=np.float32(cv2.resize(source.reshape(grid,grid),(features,features),interpolation=cv2.INTER_NEAREST))
    sm=cv2.GaussianBlur(src,(0,0),sigma)
    halo=cv2.GaussianBlur(src,(0,0),sigma*2)
    sm=.80*sm+.20*halo
    return np.clip(8.+sm*135+rng.normal(0,noise,sm.shape),0,255).astype(np.float32)


class Phone:
    def __init__(self,url:str):self.url=url.rstrip('/')+'/shot.jpg'
    def capture(self)->np.ndarray:
        req=urllib.request.Request(self.url,headers={'User-Agent':'LightScan-SeeThrough/1','Cache-Control':'no-cache'})
        with urllib.request.urlopen(req,timeout=8) as response:
            data=response.read()
        im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is None:raise RuntimeError('IP Webcam did not return a usable shot.jpg')
        return im


class Window:
    def __init__(self,w:int,h:int,zone:int,level:int):
        self.w=w;self.h=h;self.zone=min(zone,w-80,h-230);self.level=level
        if self.zone<160:raise ValueError('Desktop too small for the light area')
        self.name='SEE THROUGH | phone + diffuser'
        cv2.namedWindow(self.name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name,w,h)
        cv2.moveWindow(self.name,0,0)
        cv2.setWindowProperty(self.name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        self.current=np.zeros((h,w,3),np.uint8)
    def frame(self,source:np.ndarray|None=None,title:str='',sub:str=''):
        if source is None:source=np.zeros((12,12),np.uint8)
        frame=screen_pixels(self.w,self.h,self.zone,source,self.level)
        cv2.putText(frame,title,(24,43),cv2.FONT_HERSHEY_SIMPLEX,.72,(132,213,234),2,cv2.LINE_AA)
        cv2.putText(frame,sub[:115],(24,75),cv2.FONT_HERSHEY_SIMPLEX,.50,(135,155,170),1,cv2.LINE_AA)
        self.current=frame
        cv2.imshow(self.name,frame)
        return cv2.waitKey(35)&0xff
    def wait(self,title:str,sub:str):
        self.frame(title=title,sub=sub+' | SPACE: continue  ESC: cancel')
        while True:
            key=cv2.waitKey(70)&0xff
            if key in (13,32):return
            if key in (27,ord('q')):raise KeyboardInterrupt('User cancelled')
    def show_pred(self,source:np.ndarray,camera:np.ndarray,pred:np.ndarray,
                  title:str,score:str):
        # Previews ONLY outside the active square. They are removed before capture.
        img=self.current.copy()
        cv2.rectangle(img,(0,0),(self.w-1,84),(0,0,0),-1)
        cv2.putText(img,'LIVE THROUGH DIFFUSER  |  T TYPE  R RANDOM  C CLEAR  S SAVE  ESC QUIT',
                    (24,38),cv2.FONT_HERSHEY_SIMPLEX,.60,(150,220,224),1,cv2.LINE_AA)
        mid=(self.h+self.zone)//2; top=min(self.h-126,mid+14)
        if top<0:return
        cols=[('CAMERA',camera),('RECOVERED',pred),('KNOWN',source)]
        side=min(122,max(80,(self.h-top)-21))
        left=max(4,(self.w-(3*side+38))//2)
        for i,(lbl,array) in enumerate(cols):
            arr=np.asarray(array,np.float32)
            arr=np.uint8(np.clip(arr*255,0,255))
            arr=cv2.resize(arr,(side,side),interpolation=cv2.INTER_NEAREST)
            xx=left+i*(side+19)
            img[top:top+side,xx:xx+side]=cv2.cvtColor(arr,cv2.COLOR_GRAY2BGR)
            cv2.putText(img,lbl,(xx,top-5),cv2.FONT_HERSHEY_SIMPLEX,.40,(160,205,218),1,cv2.LINE_AA)
        cv2.putText(img,score,(24,self.h-17),cv2.FONT_HERSHEY_SIMPLEX,.48,(115,225,174),1,cv2.LINE_AA)
        cv2.imshow(self.name,img)
        return cv2.waitKey(1)&0xff
    def close(self):cv2.destroyAllWindows()


def capture_screen(win:Window,cam:Phone,pat:np.ndarray,settle:float,frames:int=1):
    win.frame(pat,'MEASURING THROUGH THE DIFFUSER','Keep phone and diffuser fixed. No camera preview in the light area.')
    time.sleep(max(0.,settle))
    shots=[cam.capture() for _ in range(frames)]
    if any(s.shape!=shots[0].shape for s in shots):raise RuntimeError('Camera resolution changed')
    return np.uint8(np.median(np.stack(shots),axis=0))


def get_desktop():
    import tkinter
    root=tkinter.Tk();root.withdraw();w,h=root.winfo_screenwidth(),root.winfo_screenheight();root.destroy()
    return int(w),int(h)


def acquisition(args,win:Window,cam:Phone,params:Params,out:Path):
    g=params.grid;n=params.features
    blackpat=np.zeros((g,g),np.uint8)
    whitepat=np.ones((g,g),np.uint8)
    win.wait('SEE THROUGH THE DIFFUSER','Put the diffuser between MONITOR and Samsung; keep it fixed')
    d0=capture_screen(win,cam,blackpat,args.settle,args.frames)
    w0=capture_screen(win,cam,whitepat,args.settle,args.frames)
    d1=capture_screen(win,cam,blackpat,args.settle,args.frames)
    bbox,geometry=detect_roi(d0,w0)
    roi=tuple(bbox)
    dark=extract(d0,roi,n)
    white=extract(w0,roi,n)
    drift=float(np.mean(np.abs(extract(d1,roi,n)-dark)))
    dynamic=float(np.percentile(white-dark,90))
    clipped=float(np.mean(extract(w0,roi,n)>248))
    geometry.update({'dynamic_camera_levels':round(dynamic,2),
                     'black_repeat_difference':round(drift,3),
                     'white_clip_fraction':round(clipped,4)})
    if dynamic<6:raise RuntimeError('Diffuser light difference too faint for useful inversion; raise exposure or thin diffuser')
    if clipped>.025:raise RuntimeError('More than 2.5% clipped; lower ISO or shutter and restart')
    print('CAMERA RESPONSE: ',json.dumps(geometry,indent=2))
    if drift>dynamic*.14:
        print('WARNING: black frames drift. Keep exposure manual, switch off monitor dynamic brightness, or increase --settle.')
    print(f'  Automated ROI: {roi}. Capture will use the illuminated region only.')
    rng=np.random.default_rng(params.seed)
    patterns=[];observations=[]
    # Fixed sample count + separate, never-fit tests.
    for i in range(args.samples+args.holdout):
        src=make_pattern(rng,g,i)
        frame=capture_screen(win,cam,src,args.settle,args.frames)
        y=normalize(extract(frame,roi,n),dark,white)
        observations.append(y);patterns.append(src.ravel().astype(np.float32))
        if (i+1)%12==0 or i+1==args.samples+args.holdout:
            print(f'  Camera/source pairs: {i+1}/{args.samples+args.holdout}')
    YY=np.stack(observations);XX=np.stack(patterns)
    model=Inverse(params.ridge).fit(YY[:args.samples],XX[:args.samples])
    pred=model.predict(YY[args.samples:]);target=XX[args.samples:]
    base=baseline_fit(YY[:args.samples],XX[:args.samples],g,n)
    basepred=baseline_predict(YY[args.samples:],base,g,n)
    report={'trained_pairs':args.samples,'heldout_pairs':args.holdout,
            'inverse':metrics(pred,target),'direct_camera_baseline':metrics(basepred,target),
            'physical_camera':geometry,'model_type':'regularized linear inverse',
            'meaning':'Measures 2D screen images behind a FIXED diffuser; not depth or CT'}
    report['beats_baseline']=report['inverse']['mae']<report['direct_camera_baseline']['mae']
    out.mkdir(parents=True,exist_ok=True)
    model.save(out/'inverse.npz',params,roi,dark,white,report)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    # Save actual held-out data and visible reconstruction example.
    last=int(len(target)-1)
    cv2.imwrite(str(out/'test_camera.png'),np.uint8(np.clip(YY[-1].reshape(n,n),0,1)*255))
    cv2.imwrite(str(out/'test_recovered.png'),np.uint8(pred[-1].reshape(g,g)*255))
    cv2.imwrite(str(out/'test_known.png'),np.uint8(target[-1].reshape(g,g)*255))
    print('HELD-OUT PHYSICAL VALIDATION:',json.dumps(report,indent=2))
    return model,roi,dark,white,report


def live_loop(args,win:Window,cam:Phone,model:Inverse,params:Params,roi,dark,white,out:Path,report):
    g=params.grid
    src=glyph_pattern('HI',g)
    pending='HI';is_typing=False
    drawing=False
    rng=np.random.default_rng(params.seed+500)
    grab=0
    def mouse(event,x,y,flags,data):
        nonlocal drawing,src,is_typing
        if is_typing:return
        if event==cv2.EVENT_LBUTTONDOWN:drawing=True
        if event==cv2.EVENT_LBUTTONUP:drawing=False
        if drawing and event in (cv2.EVENT_MOUSEMOVE,cv2.EVENT_LBUTTONDOWN):
            x0=(win.w-win.zone)//2;y0=(win.h-win.zone)//2
            if x0<=x<x0+win.zone and y0<=y<y0+win.zone:
                col=min(g-1,int((x-x0)*g/win.zone))
                row=min(g-1,int((y-y0)*g/win.zone))
                cv2.circle(src,(col,row),1,1,-1)
    cv2.setMouseCallback(win.name,mouse)
    # The camera response to previously unseen glyphs is checked for real;
    # there is NO access to src in model.predict().
    while True:
        # During a real capture, display the SAME fixed header as calibration.
        # Otherwise the header's light could leak into the diffuser measurement.
        if is_typing:
            win.frame(src,'TYPE TEXT (ENTER to show behind the diffuser)', f'Text: {pending}')
            key=cv2.waitKey(90)&0xff
            if key in (27,ord('q')):
                is_typing=False
            elif key==13:
                src=glyph_pattern(pending, g);is_typing=False
            elif key in (8,127):pending=pending[:-1]
            elif 32<=key<=126:pending=(pending+chr(key))[:8]
            continue
        prekey=win.frame(src,'MEASURING THROUGH THE DIFFUSER','Keep phone and diffuser fixed. No camera preview in the light area.')
        time.sleep(max(0.,args.live_settle))
        photo=cam.capture()
        feat=normalize(extract(photo,roi,params.features),dark,white)
        recovered=model.predict(feat)[0].reshape(g,g)
        camera_aligned=feat.reshape(params.features,params.features)
        gt=src.astype(np.float32)
        error=float(np.mean(np.abs(recovered-gt)))
        good=report.get('beats_baseline',False)
        score=f'LIVE recovered MAE {error:.3f}   |   held-out: {"BEATS BASELINE" if good else "NOT VALIDATED"}'
        showkey=win.show_pred(gt,camera_aligned,recovered,'LIVE',score)
        key=cv2.waitKey(max(1,int(args.preview_ms)))&0xff
        if key==255:key=showkey
        if key==255:key=prekey
        if key in (27,ord('q')):break
        if key==ord('t'):
            pending='';is_typing=True
            print('Type letters in the full-screen LightScan window, then ENTER')
        elif key==ord('c'):src=np.zeros((g,g),np.uint8)
        elif key==ord('r'):src=make_pattern(rng,g,int(rng.integers(0,5)))
        elif key==ord('s'):
            grab+=1
            p=out/f'live_{grab:03d}'
            cv2.imwrite(str(p.with_name(p.name+'_camera.png')),np.uint8(np.clip(camera_aligned,0,1)*255))
            cv2.imwrite(str(p.with_name(p.name+'_recovered.png')),np.uint8(recovered*255))
            cv2.imwrite(str(p.with_name(p.name+'_known.png')),np.uint8(gt*255))
            print(f'  saved {p.name}: camera, recovered, known')
    return


def simulate(args,out:Path):
    p=Params(grid=args.grid,features=args.features,ridge=args.ridge,seed=args.seed)
    rng=np.random.default_rng(p.seed)
    dark=np.full((p.features,p.features),8,np.float32)
    white=synthetic_measure(np.ones(p.grid*p.grid),p.grid,p.features,rng,noise=0)
    srcs=[];ys=[]
    for i in range(args.samples+args.holdout):
        x=make_pattern(rng,p.grid,i)
        cam=synthetic_measure(x,p.grid,p.features,rng,sigma=args.sim_blur)
        srcs.append(x.ravel().astype(np.float32))
        ys.append(normalize(cam,dark,white))
    X=np.stack(srcs);Y=np.stack(ys)
    m=Inverse(p.ridge).fit(Y[:args.samples],X[:args.samples])
    recovered=m.predict(Y[args.samples:]);tar=X[args.samples:]
    b=baseline_fit(Y[:args.samples],X[:args.samples],p.grid,p.features)
    basepred=baseline_predict(Y[args.samples:],b,p.grid,p.features)
    report={'inverse':metrics(recovered,tar), 'direct_camera_baseline':metrics(basepred,tar),
            'trained_pairs':args.samples,'heldout_pairs':args.holdout,
            'SIMULATED_ONLY':'Gaussian blur, does NOT prove real diffuser inversion'}
    report['beats_baseline']=report['inverse']['mae']<report['direct_camera_baseline']['mae']
    out.mkdir(parents=True,exist_ok=True)
    (out/'sim_results.json').write_text(json.dumps(report,indent=2)+'\n')
    cv2.imwrite(str(out/'sim_known.png'),np.uint8(tar[0].reshape(p.grid,p.grid)*255))
    cv2.imwrite(str(out/'sim_recovered.png'),np.uint8(recovered[0].reshape(p.grid,p.grid)*255))
    print(json.dumps(report,indent=2))
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description='See through a FIXED diffuser using a Samsung phone and controlled monitor')
    ap.add_argument('--url',default='http://192.168.1.157:8080',help='Your LAN IP Webcam address')
    ap.add_argument('--grid',type=int,default=8,help='Source pixel grid: 8 is easiest for strong diffusers; raise to 12 for more detail')
    ap.add_argument('--features',type=int,default=40,help='Camera sample features along one side')
    ap.add_argument('--zone',type=int,default=420,help='Size of the light square in screen pixels')
    ap.add_argument('--level',type=int,default=235,help='White light level 1..255')
    ap.add_argument('--samples',type=int,default=108,help='Known random patterns for fitting; increase to 200 for finer grids')
    ap.add_argument('--holdout',type=int,default=16,help='New patterns for verification')
    ap.add_argument('--settle',type=float,default=.85,help='Delay after changing light; prevent delayed IP snapshots')
    ap.add_argument('--live-settle',type=float,default=.45)
    ap.add_argument('--preview-ms',type=int,default=60)
    ap.add_argument('--frames',type=int,default=1,help='Median frames per pattern')
    ap.add_argument('--ridge',type=float,default=.055,help='Tikhonov regularization strength')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--output',default='runs/see_through')
    ap.add_argument('--load',default='',help='Use an earlier inverse.npz for the SAME unchanged rig')
    ap.add_argument('--sim',action='store_true',help='Synthetic optics validation, no phone/GUI')
    ap.add_argument('--sim-blur',type=float,default=6.0)
    args=ap.parse_args(argv)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not (8<=args.grid<=24 and 16<=args.features<=128 and 0<args.level<=255 and
            args.samples>=12 and args.holdout>=4 and args.frames>=1):
        ap.error('Use grid 8..24, features 16..128, valid levels/frames and sufficient training/test samples')
    if args.sim:return simulate(args,out)
    w,h=get_desktop();cam=Phone(args.url)
    win=Window(w,h,args.zone,args.level)
    try:
        if args.load:
            m,p,roi,dark,white,report=Inverse.load(Path(args.load))
            if (p.zone!=args.zone):
                raise ValueError('--zone must match the saved model')
            print('Loaded previously measured inverse. If phone/diffuser/focus moved, RECAPTURE instead.')
            win.wait('SEE THROUGH: REUSING MODEL', 'Only reuse if camera, focus, exposure and diffuser are unchanged')
        else:
            p=Params(grid=args.grid,features=args.features,zone=win.zone,
                     level=args.level,ridge=args.ridge,seed=args.seed)
            m,roi,dark,white,report=acquisition(args,win,cam,p,out)
        state='PASS' if report.get('beats_baseline') else 'INVERSE NOT VALIDATED'
        print(f'\n{state}: starting the interactive see-through test')
        win.wait('READY: SEE THROUGH THE DIFFUSER',
                 f'Comparison on previously unseen patterns: {state}. Press Space for LIVE.')
        live_loop(args,win,cam,m,p,roi,dark,white,out,report)
    finally:
        win.close()


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('\nUser cancelled')
