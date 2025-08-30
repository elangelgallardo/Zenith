#!/usr/bin/env python3
"""
hq_preview.py — Fullscreen preview for Raspberry Pi 5 + HQ Camera (Picamera2 + DRM),
with RAM‑backed, non-blocking full‑resolution JPEG capture queued to SSD.

Key features
• Preview stays up on the HDMI touchscreen; capture causes only a tiny hiccup.
• Full‑res JPEG is written to RAM (tmpfs) for speed, then moved in the background
  to /media/DCIM (your mounted SSD) without blocking further captures.
• Max JPEG quality by default (100). You can lower it if you want smaller files.
• Trigger capture via 'c' key or SIGUSR1 (kill -USR1 <pid>).

Quick start
  chmod +x hq_preview.py
  ./hq_preview.py --fps 60                           # fast preview hint
  ./hq_preview.py --ram-dir /dev/shm/hq_shots        # ensure tmpfs exists
  ./hq_preview.py --ssd-dir /media/DCIM              # default final dir
  # Programmatic capture: kill -USR1 $(pidof hq_preview.py)

Notes
- If /media/DCIM is not actually mounted, this program will still create it and
  save there on the root filesystem; it prints a warning so you can fix the mount.
- Moving from RAM to SSD is done by a background worker with retries.
- Use --fast-still to reuse current AE/AWB state for the still to reduce latency.

"""

import argparse
import os
import sys
import signal
import time
import datetime
import termios
import tty
import select
import threading
import queue
import shutil
from typing import Optional, Tuple

from picamera2 import Picamera2, Preview
from libcamera import Transform


# ------------------------------ CLI parsing ---------------------------------

def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse 'WxH' into (w, h)."""
    if not size_str:
        return None
    try:
        w_str, h_str = size_str.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid --size '{size_str}'. Use WIDTHxHEIGHT, e.g. 640x480"
        )


def build_args():
    p = argparse.ArgumentParser(
        description="Fullscreen camera preview with RAM‑staged full‑res JPEG capture queued to SSD."
    )
    p.add_argument(
        "--size",
        type=parse_size,
        default=(640, 480),  # your 640x480 panel
        help="Preview size WxH. Example: 640x480 (default).",
    )
    p.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=0,
        help="Rotate preview (degrees)."
    )
    p.add_argument("--hflip", action="store_true", help="Horizontal mirror.")
    p.add_argument("--vflip", action="store_true", help="Vertical mirror.")
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="Target frame-rate hint (not locked). Default: 30.0"
    )
    p.add_argument(
        "--ram-dir", default="/dev/shm/hq_shots",
        help="RAM tmpfs staging directory for fast captures. Default: /dev/shm/hq_shots",
    )
    p.add_argument(
        "--ssd-dir", default="/media/DCIM",
        help="Final destination on SSD (mounted at this path). Default: /media/DCIM",
    )
    # Back-compat: allow --out to act as --ssd-dir if provided
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument(
        "--prefix", default="IMG_",
        help="Filename prefix for captures. Default: IMG_",
    )
    p.add_argument(
        "--quality", type=int, default=100,
        help="JPEG quality (1-100). Max by default.",
    )
    p.add_argument(
        "--fast-still", action="store_true",
        help=(
            "Reuse current AE/AWB state for still to minimize latency "
            "(skips extra 3A settling)."
        ),
    )
    return p.parse_args()


# ------------------------------ Helpers -------------------------------------

def sanity_checks():
    """Basic environment check with actionable message."""
    if not Picamera2.global_camera_info():
        print(
            "ERROR: No camera detected by libcamera. "
            "Check ribbon cable/camera and try `libcamera-hello`.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_sensor_resolution(picam2: Picamera2) -> Tuple[int, int]:
    """Best-effort query of full sensor resolution."""
    try:
        return tuple(picam2.sensor_resolution)  # type: ignore[attr-defined]
    except Exception:
        try:
            size = picam2.camera_properties.get("PixelArraySize")
            if size:
                return int(size[0]), int(size[1])
        except Exception:
            pass
    # Fallback for Raspberry Pi HQ Cam (IMX477)
    return (4056, 3040)


def generate_filename(directory: str, prefix: str, ext: str = ".jpg") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    return os.path.join(directory, f"{prefix}{ts}{ext}")


class RawTerminal:
    """Context manager to put stdin in cbreak (single-char) mode and restore it."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        except Exception:
            pass


class FileMover:
    """Background mover that transfers files from RAM to SSD without blocking capture."""

    def __init__(self, dest_dir: str, max_retries: int = 3):
        self.dest_dir = dest_dir
        self.max_retries = max_retries
        self.q: "queue.Queue[tuple[str, str, int]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def enqueue_move(self, src_path: str) -> str:
        os.makedirs(self.dest_dir, exist_ok=True)
        dest_path = os.path.join(self.dest_dir, os.path.basename(src_path))
        self.q.put((src_path, dest_path, 0))
        print(f"[queue] enqueue -> {dest_path} (queued={self.q.qsize()})")
        return dest_path

    def _worker(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                src, dest, attempt = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # shutil.move works across filesystems (copy+unlink if needed)
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

    def flush_and_stop(self, timeout: float = 10.0):
        self.stop_event.set()
        try:
            self.q.join()
        except Exception:
            pass
        self.worker.join(timeout=timeout)


# ------------------------------ Main program --------------------------------

def main():
    args = build_args()
    if getattr(args, "out", None):  # back-compat alias
        args.ssd_dir = args.out

    args.quality = max(1, min(100, int(args.quality)))

    sanity_checks()

    os.makedirs(args.ram_dir, exist_ok=True)
    os.makedirs(args.ssd_dir, exist_ok=True)

    if not os.path.ismount(args.ssd_dir):
        print(f"[warn] {args.ssd_dir} is not a mount point. Is the SSD mounted?")

    mover = FileMover(args.ssd_dir)

    picam2 = Picamera2()
    transform = Transform(rotation=args.rotate, hflip=args.hflip, vflip=args.vflip)

    # Low-res preview pipeline (fast).
    preview_config = picam2.create_preview_configuration(
        main={"size": args.size}, transform=transform
    )
    picam2.configure(preview_config)

    # Fullscreen DRM preview (no X/Wayland required).
    picam2.start_preview(Preview.DRM)

    # FPS is a *hint*; AE may vary it depending on lighting/exposure.
    try:
        picam2.set_controls({"FrameRate": float(args.fps)})
    except Exception:
        pass  # harmless if not supported on this mode

    # Prepare a full-res still configuration.
    still_size = get_sensor_resolution(picam2)
    still_config = picam2.create_still_configuration(
        main={"size": still_size},
        transform=transform,
        buffer_count=1,
    )

    # Max JPEG quality (best-effort; some versions cap internally around ~93-95).
    try:
        picam2.options["quality"] = int(args.quality)
    except Exception:
        pass

    picam2.start()

    print(
        f"[hq_preview] DRM fullscreen | size={args.size[0]}x{args.size[1]} "
        f"| rotate={args.rotate} | hflip={args.hflip} | vflip={args.vflip} | fps~{args.fps}"
    )
    print(
        "[hq_preview] Press 'c' to capture full‑res JPEG (max quality), Ctrl+C to exit."
    )

    stop = False
    capture_flag = False

    def _stop(*_):
        nonlocal stop
        stop = True

    def _capture_signal(*_):
        nonlocal capture_flag
        capture_flag = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        signal.signal(signal.SIGUSR1, _capture_signal)
    except Exception:
        pass

    def perform_capture():
        nonlocal capture_flag
        capture_flag = False

        # Optionally freeze AE/AWB for lowest latency still.
        if args.fast_still:
            try:
                md = picam2.capture_metadata()
                controls = {}
                exp = md.get("ExposureTime")
                ag = md.get("AnalogueGain")
                if exp is not None:
                    controls["ExposureTime"] = int(exp)
                if ag is not None:
                    controls["AnalogueGain"] = float(ag)
                r = md.get("ColourGains")
                if r and len(r) == 2:
                    controls["ColourGains"] = (float(r[0]), float(r[1]))
                if controls:
                    controls["AeEnable"] = False
                    controls["AwbEnable"] = False
            except Exception:
                controls = {}
        else:
            controls = {}

        # Capture to RAM, then queue a move to SSD.
        ram_path = generate_filename(args.ram_dir, args.prefix)
        t0 = time.time()
        try:
            picam2.switch_mode_and_capture_file(
                still_config,
                ram_path,
                signal_function=(
                    (lambda cam: cam.set_controls(controls)) if controls else None
                ),
            )
            dt_ms = (time.time() - t0) * 1000.0
            print(f"[capture] RAM saved {ram_path} in {dt_ms:.0f} ms; queuing move…")
            mover.enqueue_move(ram_path)
        except Exception as e:
            print(f"[capture] ERROR: {e}")

    # Main loop
    with RawTerminal():
        try:
            while not stop:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "c":
                        capture_flag = True
                if capture_flag:
                    perform_capture()
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            mover.flush_and_stop()
            print("[hq_preview] Stopped. Queue drained; exiting.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
hq_preview.py — Fullscreen preview for Raspberry Pi 5 + HQ Camera (Picamera2 + DRM),
with RAM‑backed, non-blocking full‑resolution JPEG capture queued to SSD.

Key features
• Preview stays up on the HDMI touchscreen; capture causes only a tiny hiccup.
• Full‑res JPEG is written to RAM (tmpfs) for speed, then moved in the background
  to /media/DCIM (your mounted SSD) without blocking further captures.
• Max JPEG quality by default (100). You can lower it if you want smaller files.
• Trigger capture via 'c' key or SIGUSR1 (kill -USR1 <pid>).

Quick start
  chmod +x hq_preview.py
  ./hq_preview.py --fps 60                           # fast preview hint
  ./hq_preview.py --ram-dir /dev/shm/hq_shots        # ensure tmpfs exists
  ./hq_preview.py --ssd-dir /media/DCIM              # default final dir
  # Programmatic capture: kill -USR1 $(pidof hq_preview.py)

Notes
- If /media/DCIM is not actually mounted, this program will still create it and
  save there on the root filesystem; it prints a warning so you can fix the mount.
- Moving from RAM to SSD is done by a background worker with retries.
- Use --fast-still to reuse current AE/AWB state for the still to reduce latency.

"""

import argparse
import os
import sys
import signal
import time
import datetime
import termios
import tty
import select
import threading
import queue
import shutil
from typing import Optional, Tuple

from picamera2 import Picamera2, Preview
from libcamera import Transform


# ------------------------------ CLI parsing ---------------------------------

def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse 'WxH' into (w, h)."""
    if not size_str:
        return None
    try:
        w_str, h_str = size_str.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid --size '{size_str}'. Use WIDTHxHEIGHT, e.g. 640x480"
        )


def build_args():
    p = argparse.ArgumentParser(
        description="Fullscreen camera preview with RAM‑staged full‑res JPEG capture queued to SSD."
    )
    p.add_argument(
        "--size",
        type=parse_size,
        default=(640, 480),  # your 640x480 panel
        help="Preview size WxH. Example: 640x480 (default).",
    )
    p.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=0,
        help="Rotate preview (degrees)."
    )
    p.add_argument("--hflip", action="store_true", help="Horizontal mirror.")
    p.add_argument("--vflip", action="store_true", help="Vertical mirror.")
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="Target frame-rate hint (not locked). Default: 30.0"
    )
    p.add_argument(
        "--ram-dir", default="/dev/shm/hq_shots",
        help="RAM tmpfs staging directory for fast captures. Default: /dev/shm/hq_shots",
    )
    p.add_argument(
        "--ssd-dir", default="/media/DCIM",
        help="Final destination on SSD (mounted at this path). Default: /media/DCIM",
    )
    # Back-compat: allow --out to act as --ssd-dir if provided
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument(
        "--prefix", default="IMG_",
        help="Filename prefix for captures. Default: IMG_",
    )
    p.add_argument(
        "--quality", type=int, default=100,
        help="JPEG quality (1-100). Max by default.",
    )
    p.add_argument(
        "--fast-still", action="store_true",
        help=(
            "Reuse current AE/AWB state for still to minimize latency "
            "(skips extra 3A settling)."
        ),
    )
    return p.parse_args()


# ------------------------------ Helpers -------------------------------------

def sanity_checks():
    """Basic environment check with actionable message."""
    if not Picamera2.global_camera_info():
        print(
            "ERROR: No camera detected by libcamera. "
            "Check ribbon cable/camera and try `libcamera-hello`.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_sensor_resolution(picam2: Picamera2) -> Tuple[int, int]:
    """Best-effort query of full sensor resolution."""
    try:
        return tuple(picam2.sensor_resolution)  # type: ignore[attr-defined]
    except Exception:
        try:
            size = picam2.camera_properties.get("PixelArraySize")
            if size:
                return int(size[0]), int(size[1])
        except Exception:
            pass
    # Fallback for Raspberry Pi HQ Cam (IMX477)
    return (4056, 3040)


def generate_filename(directory: str, prefix: str, ext: str = ".jpg") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    return os.path.join(directory, f"{prefix}{ts}{ext}")


class RawTerminal:
    """Context manager to put stdin in cbreak (single-char) mode and restore it."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        except Exception:
            pass


class FileMover:
    """Background mover that transfers files from RAM to SSD without blocking capture."""

    def __init__(self, dest_dir: str, max_retries: int = 3):
        self.dest_dir = dest_dir
        self.max_retries = max_retries
        self.q: "queue.Queue[tuple[str, str, int]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def enqueue_move(self, src_path: str) -> str:
        os.makedirs(self.dest_dir, exist_ok=True)
        dest_path = os.path.join(self.dest_dir, os.path.basename(src_path))
        self.q.put((src_path, dest_path, 0))
        print(f"[queue] enqueue -> {dest_path} (queued={self.q.qsize()})")
        return dest_path

    def _worker(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                src, dest, attempt = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # shutil.move works across filesystems (copy+unlink if needed)
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

    def flush_and_stop(self, timeout: float = 10.0):
        self.stop_event.set()
        try:
            self.q.join()
        except Exception:
            pass
        self.worker.join(timeout=timeout)


# ------------------------------ Main program --------------------------------

def main():
    args = build_args()
    if getattr(args, "out", None):  # back-compat alias
        args.ssd_dir = args.out

    args.quality = max(1, min(100, int(args.quality)))

    sanity_checks()

    os.makedirs(args.ram_dir, exist_ok=True)
    os.makedirs(args.ssd_dir, exist_ok=True)

    if not os.path.ismount(args.ssd_dir):
        print(f"[warn] {args.ssd_dir} is not a mount point. Is the SSD mounted?")

    mover = FileMover(args.ssd_dir)

    picam2 = Picamera2()
    transform = Transform(rotation=args.rotate, hflip=args.hflip, vflip=args.vflip)

    # Low-res preview pipeline (fast).
    preview_config = picam2.create_preview_configuration(
        main={"size": args.size}, transform=transform
    )
    picam2.configure(preview_config)

    # Fullscreen DRM preview (no X/Wayland required).
    picam2.start_preview(Preview.DRM)

    # FPS is a *hint*; AE may vary it depending on lighting/exposure.
    try:
        picam2.set_controls({"FrameRate": float(args.fps)})
    except Exception:
        pass  # harmless if not supported on this mode

    # Prepare a full-res still configuration.
    still_size = get_sensor_resolution(picam2)
    still_config = picam2.create_still_configuration(
        main={"size": still_size},
        transform=transform,
        buffer_count=1,
    )

    # Max JPEG quality (best-effort; some versions cap internally around ~93-95).
    try:
        picam2.options["quality"] = int(args.quality)
    except Exception:
        pass

    picam2.start()

    print(
        f"[hq_preview] DRM fullscreen | size={args.size[0]}x{args.size[1]} "
        f"| rotate={args.rotate} | hflip={args.hflip} | vflip={args.vflip} | fps~{args.fps}"
    )
    print(
        "[hq_preview] Press 'c' to capture full‑res JPEG (max quality), Ctrl+C to exit."
    )

    stop = False
    capture_flag = False

    def _stop(*_):
        nonlocal stop
        stop = True

    def _capture_signal(*_):
        nonlocal capture_flag
        capture_flag = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        signal.signal(signal.SIGUSR1, _capture_signal)
    except Exception:
        pass

    def perform_capture():
        nonlocal capture_flag
        capture_flag = False

        # Optionally freeze AE/AWB for lowest latency still.
        if args.fast_still:
            try:
                md = picam2.capture_metadata()
                controls = {}
                exp = md.get("ExposureTime")
                ag = md.get("AnalogueGain")
                if exp is not None:
                    controls["ExposureTime"] = int(exp)
                if ag is not None:
                    controls["AnalogueGain"] = float(ag)
                r = md.get("ColourGains")
                if r and len(r) == 2:
                    controls["ColourGains"] = (float(r[0]), float(r[1]))
                if controls:
                    controls["AeEnable"] = False
                    controls["AwbEnable"] = False
            except Exception:
                controls = {}
        else:
            controls = {}

        # Capture to RAM, then queue a move to SSD.
        ram_path = generate_filename(args.ram_dir, args.prefix)
        t0 = time.time()
        try:
            picam2.switch_mode_and_capture_file(
                still_config,
                ram_path,
                signal_function=(
                    (lambda cam: cam.set_controls(controls)) if controls else None
                ),
            )
            dt_ms = (time.time() - t0) * 1000.0
            print(f"[capture] RAM saved {ram_path} in {dt_ms:.0f} ms; queuing move…")
            mover.enqueue_move(ram_path)
        except Exception as e:
            print(f"[capture] ERROR: {e}")

    # Main loop
    with RawTerminal():
        try:
            while not stop:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "c":
                        capture_flag = True
                if capture_flag:
                    perform_capture()
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            mover.flush_and_stop()
            print("[hq_preview] Stopped. Queue drained; exiting.")


if __name__ == "__main__":
    main()
