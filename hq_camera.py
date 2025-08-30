#!/usr/bin/env python3
"""
Ultra‑simple HQ camera preview + fastest possible full‑res JPEG capture on Raspberry Pi 5.

Design goals (nothing extra):
• Keep the preview visible.
• Capture JPEGs as fast as possible.
• Always save at max quality.
• Write to RAM first, then move to SSD in the background.

How to use:
  chmod +x hq_preview.py
  ./hq_preview.py
  Press 'c' to capture, Ctrl+C (or 'q') to quit.

Where files go:
• RAM staging: /dev/shm/hq_shots
• SSD target:  /media/DCIM   (warning printed if not mounted)

Implementation highlights:
• Single, no‑switch configuration: full‑res MAIN + 640x480 LORES, with
  preview displayed from LORES (display="lores").
  → No reconfigure at shutter time; we just grab the next frame.
• At capture we freeze AE/AWB to current values to avoid 3A delays.
• JPEG quality is requested at 100 (best effort; some builds cap slightly lower).
"""

import os
import sys
import time
import signal
import datetime
import select
import termios
import tty
import threading
import queue
import shutil

from picamera2 import Picamera2, Preview
from libcamera import Transform

# ----------------------------- Constants ------------------------------------
RAM_DIR = "/dev/shm/hq_shots"
SSD_DIR = "/media/DCIM"
LORES_SIZE = (640, 480)  # your panel
JPEG_QUALITY = 100

# ----------------------------- Utilities ------------------------------------

def _ensure_dirs():
    os.makedirs(RAM_DIR, exist_ok=True)
    os.makedirs(SSD_DIR, exist_ok=True)
    if not os.path.ismount(SSD_DIR):
        print(f"[warn] {SSD_DIR} is not a mount point. Is the SSD mounted?")


def _ts_filename(prefix="IMG_", ext=".jpg"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return os.path.join(RAM_DIR, f"{prefix}{ts}{ext}")


class _RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass


class _Mover:
    """Move files from RAM to SSD in the background without blocking capture."""
    def __init__(self, dest_dir: str, max_retries: int = 3):
        self.dest_dir = dest_dir
        self.q: "queue.Queue[tuple[str, str, int]]" = queue.Queue()
        self.stop = threading.Event()
        self.max_retries = max_retries
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def enqueue(self, src_path: str):
        os.makedirs(self.dest_dir, exist_ok=True)
        dest_path = os.path.join(self.dest_dir, os.path.basename(src_path))
        self.q.put((src_path, dest_path, 0))
        print(f"[queue] -> {dest_path} (queued={self.q.qsize()})")

    def _work(self):
        while not self.stop.is_set() or not self.q.empty():
            try:
                src, dest, attempt = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(src, dest)
                print(f"[queue] moved -> {dest}")
            except Exception as e:
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    print(f"[queue] retry {attempt+1} for {src} in {backoff:.1f}s: {e}")
                    time.sleep(backoff)
                    self.q.put((src, dest, attempt + 1))
                else:
                    print(f"[queue] ERROR moving {src} -> {dest}: {e}")
            finally:
                self.q.task_done()

    def drain_and_stop(self, timeout: float = 10.0):
        self.stop.set()
        try:
            self.q.join()
        except Exception:
            pass
        self.t.join(timeout=timeout)


# ----------------------------- Main -----------------------------------------

def main():
    # Minimal sanity check
    if not Picamera2.global_camera_info():
        print("ERROR: No camera detected by libcamera. Try `libcamera-hello`.", file=sys.stderr)
        sys.exit(1)

    _ensure_dirs()
    mover = _Mover(SSD_DIR)

    cam = Picamera2()

    # One-time, no-switch configuration: full-res main + 640x480 lores; display lores.
    # We keep transform identity for speed; tweak here if you ever need flips/rotation.
    transform = Transform()
    sensor_w, sensor_h = (cam.camera_properties.get("PixelArraySize") or (4056, 3040))
    config = cam.create_still_configuration(
        main={"size": (int(sensor_w), int(sensor_h)), "format": "YUV420"},
        lores={"size": LORES_SIZE},
        transform=transform,
        display="lores",
        buffer_count=3,
        queue=False,  # ensure capture is the next frame, not an older queued one
    )
    cam.configure(config)

    # Fullscreen DRM preview; no desktop required.
    cam.start_preview(Preview.DRM)

    # Request max JPEG quality (best effort)
    try:
        cam.options["quality"] = JPEG_QUALITY
    except Exception:
        pass

    cam.start()
    print("[hq_preview] Ready. Press 'c' to capture, 'q' or Ctrl+C to quit.")

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def _freeze_controls():
        """Freeze AE/AWB to current values to avoid 3A latency."""
        try:
            md = cam.capture_metadata()
            controls = {}
            exp = md.get("ExposureTime")
            ag = md.get("AnalogueGain")
            if exp is not None:
                controls["ExposureTime"] = int(exp)
            if ag is not None:
                controls["AnalogueGain"] = float(ag)
            gains = md.get("ColourGains")
            if gains and len(gains) == 2:
                controls["ColourGains"] = (float(gains[0]), float(gains[1]))
            if controls:
                controls["AeEnable"] = False
                controls["AwbEnable"] = False
            if controls:
                cam.set_controls(controls)
        except Exception:
            pass

    def _capture_once():
        _freeze_controls()
        path = _ts_filename()
        t0 = time.time()
        try:
            cam.capture_file(path)
            dt_ms = (time.time() - t0) * 1000.0
            print(f"[capture] RAM saved {path} in {dt_ms:.0f} ms; moving in background…")
            mover.enqueue(path)
        except Exception as e:
            print(f"[capture] ERROR: {e}")

    with _RawTerminal():
        while not stop:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch.lower() == 'c':
                    _capture_once()
                elif ch.lower() == 'q':
                    stop = True

    try:
        cam.stop()
    except Exception:
        pass
    mover.drain_and_stop()
    print("[hq_preview] Stopped. Queue drained; exiting.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ultra‑simple HQ camera preview + fastest possible full‑res JPEG capture on Raspberry Pi 5.

Design goals (nothing extra):
• Keep the preview visible.
• Capture JPEGs as fast as possible.
• Always save at max quality.
• Write to RAM first, then move to SSD in the background.

How to use:
  chmod +x hq_preview.py
  ./hq_preview.py
  Press 'c' to capture, Ctrl+C (or 'q') to quit.

Where files go:
• RAM staging: /dev/shm/hq_shots
• SSD target:  /media/DCIM   (warning printed if not mounted)

Implementation highlights:
• Single, no‑switch configuration: full‑res MAIN + 640x480 LORES, with
  preview displayed from LORES (display="lores").
  → No reconfigure at shutter time; we just grab the next frame.
• At capture we freeze AE/AWB to current values to avoid 3A delays.
• JPEG quality is requested at 100 (best effort; some builds cap slightly lower).
"""

import os
import sys
import time
import signal
import datetime
import select
import termios
import tty
import threading
import queue
import shutil

from picamera2 import Picamera2, Preview
from libcamera import Transform

# ----------------------------- Constants ------------------------------------
RAM_DIR = "/dev/shm/hq_shots"
SSD_DIR = "/media/DCIM"
LORES_SIZE = (640, 480)  # your panel
JPEG_QUALITY = 100

# ----------------------------- Utilities ------------------------------------

def _ensure_dirs():
    os.makedirs(RAM_DIR, exist_ok=True)
    os.makedirs(SSD_DIR, exist_ok=True)
    if not os.path.ismount(SSD_DIR):
        print(f"[warn] {SSD_DIR} is not a mount point. Is the SSD mounted?")


def _ts_filename(prefix="IMG_", ext=".jpg"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return os.path.join(RAM_DIR, f"{prefix}{ts}{ext}")


class _RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass


class _Mover:
    """Move files from RAM to SSD in the background without blocking capture."""
    def __init__(self, dest_dir: str, max_retries: int = 3):
        self.dest_dir = dest_dir
        self.q: "queue.Queue[tuple[str, str, int]]" = queue.Queue()
        self.stop = threading.Event()
        self.max_retries = max_retries
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def enqueue(self, src_path: str):
        os.makedirs(self.dest_dir, exist_ok=True)
        dest_path = os.path.join(self.dest_dir, os.path.basename(src_path))
        self.q.put((src_path, dest_path, 0))
        print(f"[queue] -> {dest_path} (queued={self.q.qsize()})")

    def _work(self):
        while not self.stop.is_set() or not self.q.empty():
            try:
                src, dest, attempt = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(src, dest)
                print(f"[queue] moved -> {dest}")
            except Exception as e:
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    print(f"[queue] retry {attempt+1} for {src} in {backoff:.1f}s: {e}")
                    time.sleep(backoff)
                    self.q.put((src, dest, attempt + 1))
                else:
                    print(f"[queue] ERROR moving {src} -> {dest}: {e}")
            finally:
                self.q.task_done()

    def drain_and_stop(self, timeout: float = 10.0):
        self.stop.set()
        try:
            self.q.join()
        except Exception:
            pass
        self.t.join(timeout=timeout)


# ----------------------------- Main -----------------------------------------

def main():
    # Minimal sanity check
    if not Picamera2.global_camera_info():
        print("ERROR: No camera detected by libcamera. Try `libcamera-hello`.", file=sys.stderr)
        sys.exit(1)

    _ensure_dirs()
    mover = _Mover(SSD_DIR)

    cam = Picamera2()

    # One-time, no-switch configuration: full-res main + 640x480 lores; display lores.
    # We keep transform identity for speed; tweak here if you ever need flips/rotation.
    transform = Transform()
    sensor_w, sensor_h = (cam.camera_properties.get("PixelArraySize") or (4056, 3040))
    config = cam.create_still_configuration(
        main={"size": (int(sensor_w), int(sensor_h)), "format": "YUV420"},
        lores={"size": LORES_SIZE},
        transform=transform,
        display="lores",
        buffer_count=3,
        queue=False,  # ensure capture is the next frame, not an older queued one
    )
    cam.configure(config)

    # Fullscreen DRM preview; no desktop required.
    cam.start_preview(Preview.DRM)

    # Request max JPEG quality (best effort)
    try:
        cam.options["quality"] = JPEG_QUALITY
    except Exception:
        pass

    cam.start()
    print("[hq_preview] Ready. Press 'c' to capture, 'q' or Ctrl+C to quit.")

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def _freeze_controls():
        """Freeze AE/AWB to current values to avoid 3A latency."""
        try:
            md = cam.capture_metadata()
            controls = {}
            exp = md.get("ExposureTime")
            ag = md.get("AnalogueGain")
            if exp is not None:
                controls["ExposureTime"] = int(exp)
            if ag is not None:
                controls["AnalogueGain"] = float(ag)
            gains = md.get("ColourGains")
            if gains and len(gains) == 2:
                controls["ColourGains"] = (float(gains[0]), float(gains[1]))
            if controls:
                controls["AeEnable"] = False
                controls["AwbEnable"] = False
            if controls:
                cam.set_controls(controls)
        except Exception:
            pass

    def _capture_once():
        _freeze_controls()
        path = _ts_filename()
        t0 = time.time()
        try:
            cam.capture_file(path)
            dt_ms = (time.time() - t0) * 1000.0
            print(f"[capture] RAM saved {path} in {dt_ms:.0f} ms; moving in background…")
            mover.enqueue(path)
        except Exception as e:
            print(f"[capture] ERROR: {e}")

    with _RawTerminal():
        while not stop:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch.lower() == 'c':
                    _capture_once()
                elif ch.lower() == 'q':
                    stop = True

    try:
        cam.stop()
    except Exception:
        pass
    mover.drain_and_stop()
    print("[hq_preview] Stopped. Queue drained; exiting.")


if __name__ == "__main__":
    main()
