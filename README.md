# LightScan — monitor + phone camera inverse-imaging laboratory

## Milestone 2 — Learn to read through a real diffuser

**[Interactive public demonstration](https://anttiluode.github.io/LightScan/)** · [Standalone HTML source](index.html) · [Physical inverse experiment](diffuser_xray.py)

We now have a new question: **Can a Samsung IP Webcam learn the fixed optical transform imposed by a real diffuser and reconstruct a *new* 2D screen image?** This is not CT, 3D volume recovery, or real-time detection of arbitrary hidden physical objects. The previous `lightscan.py` calibration and structured probes remain unchanged and reproducible.

- **The web page (`index.html`)** is an interactive, entirely offline-in-the-browser *simulation* of 2D Gaussian scattering and Wiener-style regularized inversion. Drag on the known-screen panel or try the buttons. Its 'inverse' uses known synthetic blur; it does *not* use the phone or a trained neural model. It is designed to make loss of information visible.
- **The PC experiment (`diffuser_xray.py`)** shows known training patterns through the **actual** fixed diffuser, records `/shot.jpg` with an intentional frame-settle interval, and trains a compact PyTorch U-Net. It reserves 20% of *non-letter* patterns for held-out validation and compares reconstruction MAE against a simple affine pixel baseline and a mean-pattern baseline. Real-world inference is unverified until you run it and report those scores.
- **No optical feedback in the illuminated square.** The full-screen display places any three-panel preview below the calibrated 420-pixel measurement area. Avoid opening IP Webcam's live browser view while capturing; the recursive screen image you discovered is a separate optical feedback experiment, not evidence of inversion. Moving the diffuser or changing phone focus, resolution or screen layout invalidates your transfer calibration.

### Run using the physical rig

Prerequisites: S20 Ultra running IP Webcam on the same local network, manual exposure and white balance fixed, diffuser mounted, monitor and phone unmoved, and a saved empty-monitor calibration from `--zone-px 420` at `runs/rig_locked/calibration.json`. The previous photometric calibration does **not** apply to a different ISO or shutter speed; `diffuser_xray.py` measures new black and white references through the diffuser.

```bat
cd /d E:\DocsHouse\1065\LightScan
git pull --ff-only
python3.13 -m pip install -r requirements-xray.txt
python3.13 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python3.13 diffuser_xray.py --url http://192.168.1.157:8080/ --calibration runs\rig_locked\calibration.json --zone-px 420 --samples 240 --settle 0.8 --output runs\diffuser_inverse_001
```

**Important:** the camera IP is a **private LAN address** and must work from your PC; the public site cannot directly connect to it. If `torch.cuda.is_available()` is false, use an appropriate CUDA build of PyTorch for your system, or let the script use the CPU. Never assume a 20-second training time: it depends on hardware, sample count and model. The default snapshot delay is deliberately more conservative than the `0.14 s` Gemini prototype. If pairs look out of sync, increase `--settle 1.5`.

- The full-screen window prompts for **Space** at startup, then records paired camera/source images without further terminal interaction. The PC screen displays the complete light zone in the calibrated position; no dragging a separate light window into place.
- Training captures only geometric shapes, not letters. It prints **held-out MAE** and both baselines, then launches the live drawing test. On-screen: left drag draws; **T** puts never-trained text on the monitor; **R** generates a new random shape; **C** clears; **P** toggles the feedback strip below the light zone; **S** saves a camera/inverse/source comparison; **Esc** quits. If keyboard input is lost, click the main full-screen illumination window.
- Results in the output directory: `pairs.npz` (paired patterns), `metrics.json` (held-out MAE, baselines), `inverse_model.pt` (weights and metadata), and manually saved `live_*.png` comparisons. These contain local camera data; do not upload images containing private information inadvertently.
- **If the model doesn't beat the affine baseline on held-out patterns, it hasn't demonstrated a useful optical inverse.** This is a diagnostic result, not a reason to silently retrain on the same test images. Consider lower spatial resolution, wider illumination, a more transmitting diffuser, stronger calibration or a measured shift-invariant transfer model.

### Reproduce without hardware

```bat
python3.13 diffuser_xray.py --sim --samples 80 --epochs 8 --output runs\inverse_sim
python3.13 -m unittest discover -s tests -v
```

Simulation uses a simple synthetic 2D Gaussian blur and simulated camera noise, and is labeled as such. Synthetic validation does **not** establish real-world inverse scattering performance. The optical feedback's recursive rectangles and delayed cursor traces are visually interesting, but can contaminate a training set if camera previews appear in the actual light zone.

---


**Milestone 1 (existing calibration and structured-light scans): real hardware measurements, not 3D CT.**

This repo is the first physical measurement instrument on the way toward a low-cost optical inverse problem: a PC monitor emits known illumination patterns, a fixed Samsung phone running **IP Webcam** captures their appearance, and LightScan measures geometric mapping and checks photometric repeatability before the new pilot 2D inverse-learning experiment described above.

Built from the research direction of [Moon](https://github.com/anttiluode/Moon) (compact physical forward rendering), [Varjoluotain](https://github.com/anttiluode/Varjoluotain) (occluder-assisted inversion), and [HeadAsResonator](https://github.com/anttiluode/HeadAsResonator) (measured transfer functions). **It does not yet copy Moon's forward shader or run a PyTorch 3D inverse solver.**

## First physical setup

```text
                 Samsung S20 Ultra (IP Webcam)
                            📷
                            │   ~30–45 cm, rigid stand
                            ▼
                      [camera sees]
                  +------------------+
                  |   light patterns |   PC monitor, upright
                  |   15 x 15 cm     |   centered, black background
                  +------------------+
```

- Fix the phone so it cannot move while capturing. Keep the camera aimed near-perpendicular to the monitor. The rectangular active area is specified in **screen pixels**, not automatically centimetres: measure the real displayed size, and adjust `--zone-px`.
- **Initially put nothing between phone and monitor.** Focus on the plane of the future diffuser/object; use manual focus/ISO/shutter/white balance if the Android IP Webcam app supports them. Lock screen brightness, disable automatic monitor brightness/HDR/contrast or dynamic image processing if possible.
- Use a darkened room. Avoid glare and specular reflections. Clean the screen with a dry, screen-safe microfiber cloth; never pour liquids onto the monitor.
- Later, make a **freestanding cardboard spacer/cradle**, holding frosted plastic a few centimetres in front of the display. Never tape wet wax, hot material, sharp objects or heavy objects onto the monitor.
- The phone and computer must be on the same reachable local network. The IP address is local to your LAN; this cloud assistant cannot connect to it.

## Windows quick start

Check from **your PC**, in a browser, that the phone's IP Webcam service works at `http://PHONE_IP:8080/`. In particular try the snapshot URL `http://PHONE_IP:8080/shot.jpg` or the video stream `http://PHONE_IP:8080/video`.

```bat
cd /d D:\anttiluode\LightScan
git pull --ff-only
python3.13 -m pip install -r requirements.txt
python3.13 lightscan.py --url http://PHONE_IP:8080/ --preview --output runs\first_rig
```

The capture defaults to the `/shot.jpg` endpoint because new snapshots usually avoid lagged MJPEG queues. For a camera stream whose snapshot endpoint is unavailable, add `--camera-mode video`.

The `--preview` option shows a small live phone view to help aim and manually focus. **Press Space** in the preview window to begin taking measurements. The preview closes before calibration so that its light cannot contaminate the measurements. On a single-monitor setup, this is particularly useful.

### On-screen guide: no need to see Command Prompt

The full-screen illumination window now includes a **guide across the top**, above the calibrated light square. It displays Stage 1 (screen geometry), Stage 2 (camera brightness) and Stage 3 (blank diffuser / hidden object). Guidance is drawn **outside the measured light square** using a black background and dim text, so it does not replace any of the intended light patterns. The banner stays unchanged during comparable camera captures, avoiding false shadow/geometry signals from changing UI text. Keep the phone fixed and avoid putting objects between it and the screen during Stages 1 and 2.

With `--scan`, you will see two prominent **on-screen pause screens**:

1. **INSERT BLANK DIFFUSER** — fit just the plain frosted sheet to the cradle; click the full-screen LightScan window if necessary, then press **Space** (or Enter) to capture 16 reference light spots.
2. **INSERT HIDDEN SHAPE** — add a black paper shape between the screen and the diffuser **without moving the diffuser or camera**; press **Space** for 16 more measurements.

There is no longer any `input()` or terminal wait during Stage 3. **Esc or Q cancels**, and the final results screen stays visible until you press Space to close. The console still records the detailed log, but no longer needs to be visible. On small monitors, the light zone is limited to leave room for the top guide.

The UI banner is constant in position and outside the active zone, but can still emit a little stray light; treat every new physical set-up as uncalibrated until the repeatability checks pass.

The tool takes 9 sequential light-spot measurements to determine a robust screen-plane homography, then 12 brightness exposures including repeated levels (and checks for dynamic response, clipping and drift). It writes:

- `runs/first_rig/calibration.json`: screen geometry, camera mapping, response samples and warnings
- `01_geometry_overlay.png`: camera view with fitted screen quadrilateral and measured spot positions
- `01_empty_screen.png`: dark-screen reference

**Share the report's `stability` and `geometry_rmse_camera_px` results, plus a screenshot of the geometry overlay.** These tell us whether we should adjust the rig or continue to physics.

If the 9 spots appear stale/late on the camera feed, increase `--settle 1.5`. A low geometry RMSE says the screen plane is mapped; it does **not** mean the phone is focused on the right depth.

### Optional first *real* differential measurement

After successful empty calibration, rerun with `--scan`:

```bat
python3.13 lightscan.py --url http://PHONE_IP:8080/ --scan --output runs\diffuser_001
```

At the first prompt install a **blank translucent diffuser** into the fixed cradle. After 16 probes, add a dark shape behind it (between the monitor and diffuser) **without moving the diffuser or camera**. Another 16 probes produce a shadow difference image and camera-frame residual. This is a real-world *2D* measurement, **not an image recovered through arbitrary opaque objects or a 3D depth estimate**.

Note: if the object goes between monitor and diffuser, screen illumination can be heavily blocked; this is the easier calibration case, but it is already worth measuring. Later we test unseen-pattern prediction and depth discrimination separately.

### Simulate the complete hardware pipeline before using phone

```bat
python3.13 lightscan.py --sim --headless --scan --output runs\sim_001
python3.13 -m unittest discover -s tests -v
```

The simulator generates an angled monitor view, a camera-like tone response and a synthetic shadow, then checks measured geometry and repeatability. The simulation is **not** evidence of reconstructing a real hidden bead or of faithfully modeling multiple scattering.

## Why we don't simply apply `pixel^2.2` to the camera

Most phone streaming apps emit 8-bit JPEG after proprietary exposure, white balance, HDR, gamma and sharpening. Display brightness itself is also nonlinear. `sRGB^2.2` is not a universal calibration. The program therefore records a *camera-display system response curve* and repeated levels; until calibrated by a stronger radiometric reference, pixels are explicitly described as **camera levels**, not radiance or optical density. Passing the repeat test is necessary but **does not prove that automatic exposure is disabled**.

## Physical success gates

1. **Geometry:** spot centres reproducible, ≥7/9 RANSAC inliers, ≤7 px reprojection RMSE; verify overlay visually.
2. **Radiometry:** sufficient dynamic range; monotonic response, no significant clipping or repeated-level drift. Keep exposure/ISO/focus fixed.
3. **Medium measurement:** blank diffuser pattern is repeatable, inserted physical object produces an informative spatial difference.
4. **Inverse milestone (NEW PILOT SCRIPT, UNVALIDATED IN PHYSICAL DATA):** fit a model to some light positions and predict held-out ones better than a simple 2D baseline. Only then attempt a sparse 3D reconstruction with physically labelled material parameters.
5. **Hidden-object challenge (NOT IMPLEMENTED):** blind, independently known bead position and depth; repeated samples and no-leak controls.

**Safety:** do not shine high-power lights at eyes or use UV sources; keep liquids, melted wax and pressure away from electronics. The project is an optical experiment, not medical imaging.

## Stage 3B — strong-light diffuser scan without repeating empty-screen calibration

Use this if you have **already successfully calibrated the empty monitor** and
have not moved the phone, changed the focus/zoom, or changed screen/camera
resolution. For a diffuser, set **manual ISO and shutter on the phone** until
its glow is measurable but not clipped; quality 90–95% reduces JPEG artifacts.
Changing ISO/exposure is fine here because we reuse **geometry only**. Previous
brightness-response values do not remain valid at the new exposure.

Keep the diffuser cradle installed before running. LightScan's full-screen
instructions tell you when to insert the hidden shape; no terminal is needed.

```bat
cd /d E:\DocsHouse\1065\LightScan
git pull --ff-only
python3.13 lightscan.py --url http://192.168.1.157:8080/ --zone-px 420 --scan-only --calibration runs\rig_locked\calibration.json --settle 1.5 --samples 3 --pattern-level 0.85 --save-frames --output runs\diffuser_bright_001
```

- `--scan-only` skips Stage 1/2 so the bare monitor is never exposed to the
  brighter diffuser-optimized camera settings.
- `--calibration` must point to the **real, saved** `rig_locked/calibration.json`.
  A screen/zone/camera resolution mismatch aborts. Changing exposure leaves
  geometry unchanged, but **any physical camera or focus movement needs a new
  geometric calibration** at a safe lower exposure.
- `--zone-px 420` must match the successful empty-screen run.
- The sixteen patterns are one whole-square white exposure and 15 half-lit
  4×4 Hadamard masks (rather than 16 tiny dots). If the full-white frame clips,
  use `--pattern-level 0.65` or lower manual ISO/shutter and restart the scan.
- The on-screen instructions ask for **blank diffuser first**, then inserting
  a dark shape between monitor and diffuser *without moving either*.
- The report now prints per-pattern **blank camera mean within the mapped area**,
  difference magnitude, the repeated-white-pattern RMS measurement noise, and
  clipped fraction. `0.07` previously referred to the average over the whole
  camera image and by itself did not establish that the shadow was absent.
- `03_shadow_difference.png` shows **red** for less detected light with the
  object, **blue** for more. `03_roi_mask.png` shows the evaluated camera area.
- `--save-frames` writes original JPEG-decoded camera frames as PNG into
  `frames/` to support later inversion, without discarding image detail.

The 16 frames are not yet inverted into a hidden object. They are measured
camera responses to specified light patterns. The next milestone is a
validated forward model that predicts held-out light patterns, then actual
reconstruction. Reference-frame repeatability measures noise *within* a scan;
it cannot by itself exclude a change in illumination or diffuser placement
while inserting the object. For a negative control, repeat this run **without
adding any object** during the second phase and compare the resulting difference.

The camera response after JPEG is nonlinear. Subtracting dark frames does not
make these pixel values physically calibrated radiance. Do not report
quantitative absorption or 3D position from them yet.
