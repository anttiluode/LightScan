"""LightScan: learn a 2D inverse of a *fixed* physical diffuser with known screen images.

Not 3D tomography; not a promise to recover information erased by scattering.
Uses LightScan's verified screen geometry and full-screen illumination controller.
Run with --sim to exercise the acquisition/training pipeline without hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import time

import cv2
import numpy as np


GRID = 64


def random_pattern(rng: np.random.Generator, n: int = GRID) -> np.ndarray:
    """No text in training: high-contrast bars, lines, ellipses, checker and blobs."""
    img = np.zeros((n, n), np.uint8)
    mode = int(rng.integers(0, 5))
    if mode == 0:
        for _ in range(int(rng.integers(1, 5))):
            a = tuple(map(int, rng.integers(0, n, 2)))
            b = tuple(map(int, rng.integers(0, n, 2)))
            cv2.line(img, a, b, 255, int(rng.integers(4, 14)))
    elif mode == 1:
        for _ in range(int(rng.integers(1, 6))):
            centre = tuple(map(int, rng.integers(6, n-6, 2)))
            axes = tuple(map(int, rng.integers(4, 20, 2)))
            cv2.ellipse(img, centre, axes, float(rng.integers(0, 180)), 0, 360, 255, -1)
    elif mode == 2:
        for _ in range(int(rng.integers(1, 5))):
            x, y = map(int, rng.integers(0, n-5, 2))
            cv2.rectangle(img, (x,y), (min(n-1,x+int(rng.integers(6,29))),min(n-1,y+int(rng.integers(6,29)))),255,-1)
    elif mode == 3:
        cell = int(rng.choice([8,16]))
        g = (rng.random((n//cell,n//cell))>.5).astype(np.uint8)*255
        img = cv2.resize(g,(n,n),interpolation=cv2.INTER_NEAREST)
    else:
        noise = rng.random((n,n)).astype(np.float32)
        smooth = cv2.GaussianBlur(noise,(0,0),float(rng.uniform(3,9)))
        img = np.uint8(smooth>float(np.quantile(smooth,rng.uniform(.35,.7))))*255
    return img


def projection(quad: np.ndarray, target: int=GRID, pad:float=.10) -> np.ndarray:
    """Warp camera coordinates to a padded square centered on the calibrated monitor zone."""
    q = np.asarray(quad,np.float32).reshape(4,2)
    mid=np.mean(q,axis=0)
    q=mid+(q-mid)*(1+pad)
    dst=np.array([[0,0],[target-1,0],[target-1,target-1],[0,target-1]],np.float32)
    return cv2.getPerspectiveTransform(q,dst)


def camera_quad(zone:Zone,H:np.ndarray)->np.ndarray:
    return cv2.perspectiveTransform(zone.corners().reshape(4,1,2),H).reshape(4,2)


def rectify(gray:np.ndarray, matrix:np.ndarray, n:int=GRID)->np.ndarray:
    if gray.ndim==3:
        gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    return cv2.warpPerspective(gray,matrix,(n,n),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def source_frame(zone:Zone,pat:np.ndarray)->np.ndarray:
    screen=zone.base()
    screen[zone.y0:zone.y0+zone.size,zone.x0:zone.x0+zone.size]=cv2.cvtColor(
        cv2.resize(pat,(zone.size,zone.size),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    return screen


def cam_normalized(cam:np.ndarray,dark:np.ndarray,scale:float)->np.ndarray:
    return np.clip((cam.astype(np.float32)-dark.astype(np.float32))/max(scale,1.),0.,1.)


def train_inverse(inputs: np.ndarray, targets: np.ndarray, *, epochs:int=32, batch:int=16,
                  seed:int=123, quiet:bool=False, device_name:str='auto'):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Block(nn.Module):
        def __init__(self,a,b):
            super().__init__()
            self.l=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(min(8,b),b),nn.GELU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(min(8,b),b),nn.GELU())
        def forward(self,x):return self.l(x)
    class InverseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=Block(1,16);self.b=Block(16,32);self.c=Block(32,64)
            self.d=Block(64,96)
            self.uc=Block(96+64,64);self.ub=Block(64+32,32);self.ua=Block(32+16,16)
            self.out=nn.Conv2d(16,1,1)
        def forward(self,x):
            a=self.a(x)
            b=self.b(F.avg_pool2d(a,2))
            c=self.c(F.avg_pool2d(b,2))
            d=self.d(F.avg_pool2d(c,2))
            x=self.uc(torch.cat((F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
            x=self.ub(torch.cat((F.interpolate(x,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
            x=self.ua(torch.cat((F.interpolate(x,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
            return torch.sigmoid(self.out(x))
    if inputs.shape!=targets.shape or inputs.ndim!=3 or inputs.shape[1:]!=(GRID,GRID):
        raise ValueError('Expected matching (N,64,64) arrays')
    n=len(inputs)
    if n<12:raise ValueError('At least 12 pairs needed for train + held-out validation')
    torch.manual_seed(seed)
    device='cuda' if (device_name=='auto' and torch.cuda.is_available()) else ('cpu' if device_name=='auto' else device_name)
    idx=np.random.default_rng(seed).permutation(n)
    nval=max(4,int(np.ceil(n*.2)))
    tr=idx[nval:]; va=idx[:nval]
    X=torch.from_numpy(inputs.astype(np.float32)).unsqueeze(1).to(device)
    Y=torch.from_numpy(targets.astype(np.float32)).unsqueeze(1).to(device)
    model=InverseNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015)
    if device=='cuda':torch.backends.cudnn.benchmark=True
    started=time.monotonic()
    best=np.inf; best_state=None
    for epoch in range(epochs):
        model.train()
        for batch_ix in np.array_split(np.random.default_rng(seed+epoch).permutation(tr),max(1,int(np.ceil(len(tr)/batch)))):
            x=X[batch_ix]; y=Y[batch_ix]
            pred=model(x)
            loss=F.l1_loss(pred,y)+.2*F.binary_cross_entropy(pred,y)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            val=float(np.mean([F.l1_loss(model(x),y).item() for x,y in zip(X[va].split(batch),Y[va].split(batch))]))
        if val<best:
            best=val
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if not quiet and (epoch==0 or (epoch+1)%max(1,epochs//8)==0 or epoch==epochs-1):
            print(f'  epoch {epoch+1}/{epochs}: held-out MAE {val:.4f} | {time.monotonic()-started:.1f}s')
    model.load_state_dict(best_state);model.eval()
    # Fit a physically unambitious, location-preserving affine baseline on TRAIN only.
    x=inputs[tr].astype(np.float64).ravel();y=targets[tr].astype(np.float64).ravel()
    slope=float(np.mean((x-x.mean())*(y-y.mean()))/(np.var(x)+1e-9))
    intercept=float(y.mean()-slope*x.mean())
    affine=np.clip(slope*inputs[va]+intercept,0,1)
    baseline=float(np.mean(np.abs(affine-targets[va])))
    constant=float(np.mean(np.abs(targets[va]-targets[tr].mean(axis=0))))
    return model,{'device':str(device),'train':int(len(tr)),'holdout':int(len(va)),
                  'model_mae':float(best),'affine_baseline_mae':baseline,
                  'mean_pattern_baseline_mae':constant,
                  'passes_affine_baseline':bool(best<baseline),'seconds_train':round(time.monotonic()-started,2),
                  'slope':slope,'intercept':intercept},device


def synthetic_camera(pat:np.ndarray, rng:np.random.Generator, sigma:float=3.0)->np.ndarray:
    out=cv2.GaussianBlur(pat.astype(np.float32),(0,0),sigma)
    out=5+out*.55+rng.normal(0,2,out.shape)
    return np.clip(out,0,255).astype(np.uint8)


def save_image(path:Path,src:np.ndarray):
    if not cv2.imwrite(str(path),src):raise RuntimeError(f'Cannot save {path}')


def capture_live_pairs(args, zone:Zone, screen:Screen, cam:IPCamera, roi_mat:np.ndarray, out:Path):
    from lightscan import capture_after
    screen.wait_for_space('DIFFUSER X-RAY  |  READY',[
        'Keep the DIFFUSER installed for all training and live tests.',
        'Lock exposure, focus, white balance. Do NOT move phone.',
        'The app will display known shapes, then learn from camera images.',
        'Press SPACE to record black and white references.'])
    screen.guide('DIFFUSER X-RAY  |  REFERENCES','Do not move the phone or diffuser.','White + black camera response')
    dark=rectify(capture_after(screen,cam,zone.base(),args.settle,args.frames),roi_mat)
    white=rectify(capture_after(screen,cam,source_frame(zone,np.full((GRID,GRID),255,np.uint8)),args.settle,args.frames),roi_mat)
    response=white.astype(np.float32)-dark.astype(np.float32)
    bright=float(np.percentile(response,90))
    clipped=float(np.mean(white>248))
    print(f'  Diffuser white-minus-dark P90 {bright:.1f}/255; clipped {clipped:.1%}')
    if bright<6:raise RuntimeError('Diffuser signal too weak. Increase manual exposure or ISO, or use a thinner diffuser.')
    if clipped>.02:raise RuntimeError('Diffuser region clipping. Lower exposure/ISO and restart.')
    cam_scale=max(10,float(np.percentile(response,99)))
    rng=np.random.default_rng(args.seed)
    X=[];Y=[]
    screen.guide('DIFFUSER X-RAY  |  LEARNING THE OPTICS',
                 'Keep phone and diffuser fixed. Do not use browser camera preview.',
                 'Displaying geometric training shapes (no letters).')
    first=None
    for i in range(args.samples):
        pat=random_pattern(rng)
        frame=capture_after(screen,cam,source_frame(zone,pat),args.settle,args.frames)
        crop=rectify(frame,roi_mat)
        X.append(cam_normalized(crop,dark,cam_scale));Y.append(pat.astype(np.float32)/255)
        if (i+1)%25==0 or i==0:
            print(f'  paired patterns {i+1}/{args.samples}')
        if first is None:first=crop
    X=np.stack(X);Y=np.stack(Y)
    np.savez_compressed(out/'pairs.npz',camera=X.astype(np.float16),source=Y.astype(np.uint8),dark=dark,
                        source_scale=cam_scale,training_seed=args.seed)
    return X,Y,dark,cam_scale


def render_comparison(crop:np.ndarray,pred:np.ndarray,pat:np.ndarray) -> np.ndarray:
    """A compact strip *below* the luminous zone, not nested inside it."""
    cols=[crop,pred,pat]
    out=np.zeros((150,486,3),np.uint8)
    for i,x in enumerate(cols):
        v=np.uint8(np.clip(x*255,0,255)) if x.dtype.kind=='f' else x
        v=cv2.resize(v,(142,120),interpolation=cv2.INTER_NEAREST)
        out[22:142,7+162*i:149+162*i]=cv2.cvtColor(v,cv2.COLOR_GRAY2BGR)
        cv2.putText(out,['CAMERA','INVERSE','KNOWN SOURCE'][i],(8+162*i,15),cv2.FONT_HERSHEY_SIMPLEX,
                    .36,(170,215,226),1,cv2.LINE_AA)
    return out


def live_loop(args, zone,screen,cam,warp,dark,scale,model,device,out):
    import torch
    canvas=np.zeros((GRID,GRID),np.uint8)
    rng=np.random.default_rng(args.seed+842)
    canvas[:]=random_pattern(rng)
    drawing=False; panel=True
    def mouse(ev,x,y,flags,param):
        nonlocal drawing, canvas
        if ev==cv2.EVENT_LBUTTONDOWN: drawing=True
        if ev==cv2.EVENT_LBUTTONUP: drawing=False
        if ev==cv2.EVENT_RBUTTONDOWN:canvas[:]=0
        if drawing and ev in (cv2.EVENT_LBUTTONDOWN,cv2.EVENT_MOUSEMOVE):
            if zone.x0<=x<zone.x0+zone.size and zone.y0<=y<zone.y0+zone.size:
                px=int((x-zone.x0)/zone.size*GRID);py=int((y-zone.y0)/zone.size*GRID)
                cv2.circle(canvas,(px,py),2,255,-1)
    cv2.setMouseCallback(screen.name,mouse)
    screen.guide('DIFFUSER X-RAY  |  LIVE TEST',
                 'LEFT DRAG: draw  |  T: secret text  |  R: new random  |  C: clear',
                 'P: toggle feedback strip  |  S: save evidence  |  ESC: quit')
    frame_count=0
    while True:
        stage=source_frame(zone,canvas)
        screen.show(stage)
        if args.live_settle>0:time.sleep(args.live_settle)
        crop=rectify(cam.capture(),warp)
        x=cam_normalized(crop,dark,scale)
        with torch.inference_mode():
            t=torch.from_numpy(x[None,None]).to(device)
            pred=model(t)[0,0].detach().cpu().numpy()
        if panel:
            # This panel is below the 420px luminous square; keep other previews CLOSED.
            strip=render_comparison(x,pred,canvas)
            top=min(zone.height-150,zone.y0+zone.size+22)
            left=max(0,(zone.width-486)//2)
            stage[top:top+150,left:left+486]=strip
            screen.show(stage)
        k=cv2.waitKey(30)&255
        if k in (27,ord('q')):break
        if k==ord('p'):panel=not panel
        if k==ord('c'):canvas[:]=0
        if k==ord('r'):canvas[:]=random_pattern(rng)
        if k==ord('t'):
            canvas[:]=0
            cv2.putText(canvas,'HI!',(5,42),cv2.FONT_HERSHEY_SIMPLEX,1.3,255,3,cv2.LINE_AA)
        if k==ord('s'):
            frame_count+=1
            save_image(out/f'live_{frame_count:03d}_camera.png',np.uint8(x*255))
            save_image(out/f'live_{frame_count:03d}_inverse.png',np.uint8(pred*255))
            save_image(out/f'live_{frame_count:03d}_source.png',canvas)
            print(f'  saved live comparison {frame_count:03d}')


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.sim:
        rng=np.random.default_rng(args.seed)
        pats=np.stack([random_pattern(rng) for _ in range(args.samples)])
        dark=np.full((GRID,GRID),5,np.uint8)
        cam=np.stack([synthetic_camera(p,rng,sigma=args.sim_blur) for p in pats])
        X=np.stack([cam_normalized(c,dark,140) for c in cam])
        Y=pats.astype(np.float32)/255
        print('SIMULATION: synthetic Gaussian blur; NOT proof of seeing through real diffuser.')
        warp=None;zone=None;screen=None;camera=None;scale=140
    else:
        from lightscan import Zone, Screen, IPCamera, load_calibration, screen_dimensions
        from types import SimpleNamespace
        w,h=screen_dimensions(SimpleNamespace(sim=False,screen_width=args.screen_width,screen_height=args.screen_height))
        if w<args.zone_px+60 or h<args.zone_px+180:
            raise RuntimeError('Zone does not fit screen with guidance area')
        zone=Zone(w,h,args.zone_px)
        camera=IPCamera(args.url,'shot')
        screen=Screen(zone,False)
        try:
            first=camera.capture()
            H,_=load_calibration(Path(args.calibration),zone,first.shape)
            warp=projection(camera_quad(zone,H),pad=args.crop_pad)
            X,Y,dark,scale=capture_live_pairs(args,zone,screen,camera,warp,out)
        except Exception:
            camera.close();screen.close();raise
    try:
        model,metrics,device=train_inverse(X,Y,epochs=args.epochs,batch=args.batch,seed=args.seed,
                                           device_name=args.device)
    except BaseException:
        if not args.sim:
            camera.close(); screen.close()
        raise
    print('\nHELD-OUT TEST (not training data):')
    for k in ('device','train','holdout','model_mae','affine_baseline_mae','mean_pattern_baseline_mae','passes_affine_baseline'):
        print(f'  {k}: {metrics[k]}')
    if not metrics['passes_affine_baseline']:
        print('WARNING: model does not beat a direct affine camera baseline on held-out shapes.')
    (out/'metrics.json').write_text(json.dumps({**metrics,'note':'2D projection, no 3D or depth claim; measured held-out shapes only.'},indent=2)+'\n')
    import torch
    torch.save({'state_dict':model.state_dict(),'metrics':metrics,'crop_pad':args.crop_pad},out/'inverse_model.pt')
    if args.sim:
        with torch.inference_mode():
            sample=torch.from_numpy(X[:1,None]).to(device)
            pred=model(sample)[0,0].cpu().numpy()
        save_image(out/'sim_camera.png',np.uint8(X[0]*255))
        save_image(out/'sim_inverse.png',np.uint8(pred*255))
        save_image(out/'sim_source.png',np.uint8(Y[0]*255))
        print(f'  simulation files in {out.resolve()}')
        return metrics
    try:
        screen.wait_for_space('DIFFUSER X-RAY  |  MODEL TRAINED',[
            f'Held-out MAE {metrics["model_mae"]:.3f}; affine baseline {metrics["affine_baseline_mae"]:.3f}',
            'If the model loses to baseline, reconstruction is NOT validated.',
            'Keep camera fixed. Try never-seen letters and random drawings.',
            'SPACE: enter live mode; ESC: cancel.'])
        live_loop(args,zone,screen,camera,warp,dark,scale,model,device,out)
    finally:
        camera.close();screen.close()
    return metrics


def args_parser(argv=None):
    ap=argparse.ArgumentParser(description='Calibrated monitor-to-camera diffuser inverse experiment (2D)')
    ap.add_argument('--url',default='http://192.168.1.157:8080/')
    ap.add_argument('--calibration',default='runs/rig_locked/calibration.json')
    ap.add_argument('--zone-px',type=int,default=420)
    ap.add_argument('--screen-width',type=int,default=None)
    ap.add_argument('--screen-height',type=int,default=None)
    ap.add_argument('--samples',type=int,default=240)
    ap.add_argument('--settle',type=float,default=.8,help='Pattern settle for /shot.jpg; 0.14s can mismatch camera frames')
    ap.add_argument('--frames',type=int,default=2,help='Median JPEG snapshots per pattern')
    ap.add_argument('--live-settle',type=float,default=.15)
    ap.add_argument('--epochs',type=int,default=32)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--crop-pad',type=float,default=.10)
    ap.add_argument('--seed',type=int,default=13)
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--output',default='runs/diffuser_xray')
    ap.add_argument('--sim',action='store_true')
    ap.add_argument('--sim-blur',type=float,default=2.5)
    return ap.parse_args(argv)


if __name__=='__main__':
    try:run(args_parser())
    except KeyboardInterrupt:print('\nCancelled.')
