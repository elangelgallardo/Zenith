#!/usr/bin/env python3
# touch_calibrate.py
import sys, time, json, os
from statistics import mean
try:
    from evdev import InputDevice, ecodes, list_devices
except Exception as e:
    print("evdev not installed:", e); sys.exit(2)

if len(sys.argv) < 2:
    print("usage: sudo python3 touch_calibrate.py /dev/input/event7")
    sys.exit(1)

devpath = sys.argv[1]
dev = InputDevice(devpath)
print("Using device:", devpath, dev.name)
print("You'll be asked to touch four positions when prompted.")
print("Press Ctrl-C anytime to abort.\n")

def collect_samples(prompt, samples=20, delay=0.15):
    print(prompt)
    vals = []
    cnt = 0
    last_x = None
    last_y = None
    start = time.time()
    while cnt < samples:
        for ev in dev.read_loop():
            if ev.type == ecodes.EV_ABS:
                if ev.code in (ecodes.ABS_MT_POSITION_X, ecodes.ABS_X):
                    last_x = ev.value
                elif ev.code in (ecodes.ABS_MT_POSITION_Y, ecodes.ABS_Y):
                    last_y = ev.value
            elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_TOUCH and ev.value == 1:
                if last_x is not None and last_y is not None:
                    vals.append((int(last_x), int(last_y)))
                    cnt += 1
                    print(f"  sample {cnt}/{samples}: x={last_x} y={last_y}")
                    time.sleep(delay)
            # safety: break if too long waiting for a touch
            if time.time() - start > 60 and cnt == 0:
                print("No touches detected for 60s — aborting.")
                sys.exit(2)
            if cnt >= samples:
                break
    xs = [x for x, y in vals]; ys = [y for x, y in vals]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "mean_x": int(mean(xs)), "mean_y": int(mean(ys))}

try:
    out = {}
    # Ask user to touch near LEFT (roughly left edge)
    out['left'] = collect_samples("Touch and hold NEAR THE LEFT EDGE of the screen repeatedly (20 samples).")
    time.sleep(0.4)
    out['right'] = collect_samples("Touch and hold NEAR THE RIGHT EDGE of the screen repeatedly (20 samples).")
    time.sleep(0.4)
    out['top'] = collect_samples("Touch and hold NEAR THE TOP EDGE of the screen repeatedly (20 samples).")
    time.sleep(0.4)
    out['bottom'] = collect_samples("Touch and hold NEAR THE BOTTOM EDGE of the screen repeatedly (20 samples).")

    # Derive x_min/x_max and y_min/y_max to map raw -> preview pixels
    x_min = int(min(out['left']['mean_x'], out['top']['mean_x'], out['bottom']['mean_x']))
    x_max = int(max(out['right']['mean_x'], out['top']['mean_x'], out['bottom']['mean_x']))
    y_min = int(min(out['top']['mean_y'], out['left']['mean_y'], out['right']['mean_y']))
    y_max = int(max(out['bottom']['mean_y'], out['left']['mean_y'], out['right']['mean_y']))

    calib = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "raw_samples": out}
    cfg_path = os.path.expanduser("~/.camera_touch_calib.json")
    with open(cfg_path, "w") as f:
        json.dump(calib, f, indent=2)
    print("\nCalibration saved to", cfg_path)
    print("Calibration results:", calib)
    print("Now run your camera script as before. The camera will pick up this calibration automatically.")
except KeyboardInterrupt:
    print("\nAborted by user")
    sys.exit(1)
