# LightScan — See Through the Diffuser

**One experiment.** Put a fixed diffuser in front of a PC monitor, point a Samsung running IP Webcam at it, and ask whether the camera signal can reconstruct **new images shown on the monitor**. No 3D/CT claims, no hidden-object scan, no old calibration file needed.

[**Try the public optical sandbox**](https://anttiluode.github.io/LightScan/) · [Code](see_through.py) · [Prior experiments](docs/previous_experiments.md)

## On your Windows PC

Keep the Samsung fixed with its manual focus, exposure, ISO and white balance locked. Leave your diffuser box between the screen and phone. Close the browser's recursive IP camera preview. Keep the monitor brightness fixed.

```bat
cd /d E:\DocsHouse\1066\LightScan
git pull --ff-only
python3.13 -m pip install -r requirements.txt
python3.13 see_through.py
```

`see_through.py` uses your IP Webcam at `http://192.168.1.157:8080/shot.jpg` by default. For a different address use `--url http://YOUR_PHONE:8080`. It needs **NumPy + OpenCV**, not PyTorch or a GPU. The screen displays instructions; **Space** starts the measurement.

The script automatically finds the illuminated camera region from black and white screen references. It then shows known patterns **behind the actual diffuser**, waits for the camera, and learns a regularized optical inverse. A separate set of new patterns is captured but **never fitted**, and their error is compared with a direct-blurry-camera baseline.

When capture finishes, **Space** opens the live test. The only data sent to the inverse is the phone camera frame; the known screen image is displayed separately for verification.

| Key | Action |
|---|---|
| Mouse drag on light square | Draw a new shape |
| `T`, type a word, `Enter` | Display fresh, **never-trained text** behind diffuser |
| `R` | Random image | 
| `C` | Clear |
| `S` | Save camera / reconstructed / original images to `runs/see_through` |
| `Esc` | Quit |

The default spatial resolution is **8 × 8**. This is deliberately coarse, to test whether any recoverable information survives a strong diffuser instead of using a neural image generator to invent convincing details. For more detail after a successful run:

```bat
python3.13 see_through.py --grid 12 --samples 220 --output runs\fine_scan
```

For more stable or brighter samples, adjust the **phone's manual exposure**, not automatic exposure; a `--settle 1.5` option is useful if the phone returns images from a prior pattern. Changing exposure or moving the phone or diffuser requires fitting again.

**Reopen a previously learned optical inverse** *only if the phone, focus, lens, diffuser, screen position and exposure are unchanged*:

```bat
python3.13 see_through.py --load runs\see_through\inverse.npz
```

Results: `inverse.npz` (your measured inverse), `results.json` (held-out error vs baseline), `test_*.png` and any saved live comparisons. Captured phone images stay on the PC unless you share them.

### If you see nothing

- If the program reports weak white-minus-black signal, too much spatial information may have been lost. Try brighter, manually controlled exposure or a thinner diffuser; avoid clipping. It cannot recover data the sensor never received.
- If new letters remain illegible, try `--grid 8` (default), verify the phone still sees a varying glow, and avoid moving the diffuser or changing screen brightness.
- `/shot.jpg` is a network JPEG snapshot. **Real-time means it updates as fast as the camera responds**, not a guaranteed 60 FPS. In particular, 100+ matched camera measurements cannot honestly be promised in a few seconds.
- A calibration metric beating a baseline on random patterns does **not** guarantee that live unseen text will work; the live images are the important falsification test.

### Simulation and tests

```bat
python3.13 see_through.py --sim --output runs\synthetic_inverse
python3.13 -m unittest discover -s tests -v
```

The simulated inverse uses a Gaussian blur, **not actual physical measurements**. It cannot validate the Samsung/diffuser experiment.

## Where this came from

[Varjoluotain](https://github.com/anttiluode/Varjoluotain) taught us to control the probe and measure which hidden patterns remain observable. [HeadAsResonator](https://github.com/anttiluode/HeadAsResonator) motivated measuring and inverting an actual physical transfer. [Moon](https://github.com/anttiluode/Moon) explores compact forward models. This project currently uses a simpler, explicitly measured 2D inverse rather than pretending to perform full volumetric multiple-scattering tomography.
