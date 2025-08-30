#!/usr/bin/env python3
"""
hq_camera.py

Preview + full-resolution capture button for Raspberry Pi Picamera2.

This patch adds a --jpeg-quality option and sets picam2.options['quality'] accordingly
so JPEG encoder uses higher quality (less compression -> larger file sizes).
"""

from pathlib import Path
from datetime import datetime
import argparse
import os
import sys
import threading
import time

try:
    from picamera2 import Picamera2, Preview
except Exception as e:
    print("ERROR: could not import Picamera2. Install python3-picamera2 on Raspberry Pi OS.")
    raise

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional evdev for touch
try:
    from evdev import list_devices, InputDevice, ecodes
    EVDEV_AVAILABLE = True
except Exception:
    EVDEV_AVAILABLE = False

# ---- Defaults / Layout ----
DEFAULT_PREVIEW_SIZE = (640, 480)
BUTTON_RADIUS_RATIO = 0.11
BUTTON_TOUCH_PADDING_PX = 8
BUTTON_MARGIN_RATIO = 0.03
BUTTON_COLOR = (255, 255, 255, 230)  # white mostly-opaque
BUTTON_PRESSED_COLOR = (255, 255, 255, 255)
BUTTON_STROKE = 2
FLASH_MS = 120

TEXT_COLOR = (255, 255, 255, 230)   # white, slightly translucent
TEXT_BG_COLOR = (0, 0, 0, 110)      # translucent black behind text for readability
TEXT_MARGIN = 8                      # px padding around text

def _load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w, h)
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (0, 0)

# Formatting helpers
def format_shutter(exposure_us):
    if not exposure_us:
        return "—"
    try:
        secs = float(exposure_us) / 1e6
        if secs <= 0:
            return "—"
        if secs < 1.0:
            denom = round(1.0 / secs)
            if denom == 0:
                return f"{secs:.3f} s"
            return f"1/{denom} s"
        else:
            return f"{secs:.1f} s"
    except Exception:
        return "—"

def format_iso(analogue_gain):
    if analogue_gain is None:
        return "—"
    try:
        iso = int(round(float(analogue_gain) * 100.0))
        return f"ISO {iso}"
    except Exception:
        return "—"

def format_wb(metadata):
    ct = metadata.get("ColourTemperature") or metadata.get("ColourTemp") or metadata.get("colour_temperature")
    if ct:
        try:
            return f"{int(round(ct))}K"
        except Exception:
            pass
    gains = metadata.get("ColourGains") or metadata.get("ColourGain") or metadata.get("colour_gains")
    if gains and isinstance(gains, (list, tuple)) and len(gains) >= 3:
        r, g, b = gains[:3]
        return f"{r:.2f}:{g:.2f}:{b:.2f}"
    awb = metadata.get("AwbMode") or metadata.get("AWBMode") or metadata.get("awb_mode")
    if awb:
        return str(awb)
    return "—"

# Touch helpers (same as before)
def find_touchscreen_device():
    if not EVDEV_AVAILABLE:
        return None
    devices = [InputDevice(path) for path in list_devices()]
    for dev in devices:
        caps = dev.capabilities(verbose=False)
        abs_ = caps.get(3, [])  # EV_ABS
        if ecodes.ABS_MT_POSITION_X in abs_ or ecodes.ABS_X in abs_:
            return dev
    return None

def get_abs_range(dev):
    caps = dev.capabilities(absinfo=True)
    absinfo = caps.get(ecodes.EV_ABS, {})
    if ecodes.ABS_MT_POSITION_X in absinfo and ecodes.ABS_MT_POSITION_Y in absinfo:
        ax = absinfo[ecodes.ABS_MT_POSITION_X]
        ay = absinfo[ecodes.ABS_MT_POSITION_Y]
        return {"x_min": ax.min, "x_max": ax.max, "y_min": ay.min, "y_max": ay.max}
    if ecodes.ABS_X in absinfo and ecodes.ABS_Y in absinfo:
        ax = absinfo[ecodes.ABS_X]
        ay = absinfo[ecodes.ABS_Y]
        return {"x_min": ax.min, "x_max": ax.max, "y_min": ay.min, "y_max": ay.max}
    return {"x_min": 0, "x_max": 32767, "y_min": 0, "y_max": 32767}

def touch_listener(dev, mapping, action_callback, stop_event):
    last_x = None
    last_y = None

    def looks_like_default_range(vmin, vmax):
        return vmin == 0 and vmax == 32767

    m_xmin = mapping.get("x_min", 0)
    m_xmax = mapping.get("x_max", 32767)
    m_ymin = mapping.get("y_min", 0)
    m_ymax = mapping.get("y_max", 32767)

    obs_x_min = None if looks_like_default_range(m_xmin, m_xmax) else m_xmin
    obs_x_max = None if looks_like_default_range(m_xmin, m_xmax) else m_xmax
    obs_y_min = None if looks_like_default_range(m_ymin, m_ymax) else m_ymin
    obs_y_max = None if looks_like_default_range(m_ymin, m_ymax) else m_ymax

    def update_observed(vx, vy):
        nonlocal obs_x_min, obs_x_max, obs_y_min, obs_y_max
        if vx is not None:
            if obs_x_min is None or vx < obs_x_min:
                obs_x_min = vx
            if obs_x_max is None or vx > obs_x_max:
                obs_x_max = vx
        if vy is not None:
            if obs_y_min is None or vy < obs_y_min:
                obs_y_min = vy
            if obs_y_max is None or vy > obs_y_max:
                obs_y_max = vy

    try:
        if mapping.get("debug"):
            print("DEBUG: touch_listener starting for", getattr(dev, "path", str(dev)),
                  "initial mapping:", {"x_min": m_xmin, "x_max": m_xmax, "y_min": m_ymin, "y_max": m_ymax, "preview_size": mapping.get("preview_size")})
            print("DEBUG: initial observed ranges:", obs_x_min, obs_x_max, obs_y_min, obs_y_max)

        for ev in dev.read_loop():
            if stop_event.is_set():
                break

            if ev.type == ecodes.EV_ABS:
                if ev.code in (ecodes.ABS_MT_POSITION_X, ecodes.ABS_X):
                    last_x = ev.value
                elif ev.code in (ecodes.ABS_MT_POSITION_Y, ecodes.ABS_Y):
                    last_y = ev.value

                update_observed(last_x, last_y)

                if mapping.get("debug"):
                    name = ecodes.ABS.get(ev.code, ev.code)
                    print(f"DEBUG: EV_ABS {name} ({ev.code}) = {ev.value}; observed_x=({obs_x_min},{obs_x_max}) observed_y=({obs_y_min},{obs_y_max})")

            elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_TOUCH:
                if mapping.get("debug"):
                    print(f"DEBUG: EV_KEY BTN_TOUCH = {ev.value} (last_x={last_x}, last_y={last_y})")

                if ev.value == 1 and last_x is not None and last_y is not None:
                    w, h = mapping['preview_size']

                    use_x_min, use_x_max = obs_x_min, obs_x_max
                    use_y_min, use_y_max = obs_y_min, obs_y_max

                    def sane_range(vmin, vmax):
                        try:
                            return vmin is not None and vmax is not None and (vmax - vmin) > 5
                        except Exception:
                            return False

                    if not sane_range(use_x_min, use_x_max):
                        use_x_min = m_xmin
                        use_x_max = m_xmax
                        if mapping.get("debug"):
                            print("DEBUG: observed X not sane; falling back to mapping absinfo:", use_x_min, use_x_max)
                    if not sane_range(use_y_min, use_y_max):
                        use_y_min = m_ymin
                        use_y_max = m_ymax
                        if mapping.get("debug"):
                            print("DEBUG: observed Y not sane; falling back to mapping absinfo:", use_y_min, use_y_max)

                    if use_x_max == use_x_min or use_y_max == use_y_min:
                        if mapping.get("debug"):
                            print("DEBUG: invalid axis ranges, skipping mapping")
                        continue

                    x_px = int((last_x - use_x_min) / float(use_x_max - use_x_min) * (w - 1))
                    y_px = int((last_y - use_y_min) / float(use_y_max - use_y_min) * (h - 1))

                    if mapping.get("swap_axes"):
                        x_px, y_px = y_px, x_px
                    if mapping.get("invert_x"):
                        x_px = (w - 1) - x_px
                    if mapping.get("invert_y"):
                        y_px = (h - 1) - y_px

                    x_px = max(0, min(w - 1, x_px))
                    y_px = max(0, min(h - 1, y_px))

                    if mapping.get("debug"):
                        print(f"DEBUG: mapped touch -> x_px={x_px}, y_px={y_px} (preview {w}x{h})")
                        print(f"DEBUG: using ranges X=({use_x_min},{use_x_max}) Y=({use_y_min},{use_y_max})")
                        print(f"DEBUG: button geom = {mapping.get('button_geom', '(not set)')}")

                    try:
                        action_callback(x_px, y_px)
                    except Exception as e:
                        print("DEBUG: action_callback exception:", e)

    except Exception as e:
        print("Touch listener exited:", e)

# Overlay composition (unchanged)
def compose_overlay_array(preview_size, text_lines, pressed=False):
    width, height = preview_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    font_size = max(10, int(min(width, height) * 0.045))
    font = _load_font(font_size)
    line_heights = []
    max_w = 0
    for line in text_lines:
        w, h = _text_size(draw, line, font)
        line_heights.append(h)
        if w > max_w:
            max_w = w
    block_w = max_w + 2 * TEXT_MARGIN
    block_h = sum(line_heights) + 2 * TEXT_MARGIN + (len(line_heights)-1) * 2

    draw.rectangle((TEXT_MARGIN, TEXT_MARGIN, TEXT_MARGIN + block_w, TEXT_MARGIN + block_h),
                   fill=TEXT_BG_COLOR)

    y = TEXT_MARGIN + TEXT_MARGIN // 2
    for i, line in enumerate(text_lines):
        draw.text((TEXT_MARGIN + 4, y), line, font=font, fill=TEXT_COLOR)
        y += line_heights[i] + 2

    r = int(min(width, height) * BUTTON_RADIUS_RATIO)
    margin = int(width * BUTTON_MARGIN_RATIO)
    cx = width - margin - r
    cy = height - margin - r

    shadow_offset = max(2, r // 8)
    shadow_bbox = (cx - r + shadow_offset, cy - r + shadow_offset, cx + r + shadow_offset, cy + r + shadow_offset)
    draw.ellipse(shadow_bbox, fill=(0, 0, 0, 90))

    fill = BUTTON_PRESSED_COLOR if pressed else BUTTON_COLOR
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse(bbox, fill=fill, outline=(0, 0, 0, 140), width=BUTTON_STROKE)

    inner_r = max(2, r // 3)
    draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r), fill=(0, 0, 0, 60))

    return np.array(img, dtype=np.uint8), (cx, cy, r)

# Main camera class
class HQCameraWithButton:
    def __init__(self, preview_size=(640, 480), outdir="captures", touchscreen_device_path=None,
                 debug_touch=False, invert_x=False, invert_y=False, swap_axes=False,
                 buffer_count=6, request_raw=True, still_size=None, jpeg_quality=95):
        self.preview_size = preview_size
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.picam2 = Picamera2()

        # --- NEW: set JPEG quality via picam2.options ---
        # picamera2 exposes a global 'options' mapping where 'quality' sets the JPEG encoder quality.
        # Higher -> less compression -> larger files. Typical acceptable range: 80..95. 95 is near-max.
        try:
            if isinstance(jpeg_quality, int) and 1 <= jpeg_quality <= 100:
                # clamp at 95 if underlying stack doesn't support >95; many stacks use 0..95 as max-quality.
                q = int(jpeg_quality)
                if q > 95:
                    q = 95
                self.picam2.options['quality'] = q
            else:
                # default fallback
                self.picam2.options['quality'] = 95
        except Exception:
            # Some picamera2 builds might not expose options; ignore failure and continue.
            pass

        # (rest of __init__ builds combined configuration like before)
        try:
            sensor_res = self.picam2.sensor_resolution
        except Exception:
            sensor_res = None

        self.preview_config = None
        self.still_config = None

        try:
            if still_size:
                target_main_size = still_size
            else:
                target_main_size = sensor_res

            cfg_kwargs = {
                "main": {"size": target_main_size, "format": "RGB888"},
                "lores": {"size": preview_size, "format": "YUV420"},
                "display": "lores",
                "buffer_count": max(2, int(buffer_count))
            }
            if request_raw and sensor_res:
                cfg_kwargs["raw"] = {"size": sensor_res}

            combined_cfg = self.picam2.create_still_configuration(**cfg_kwargs)
            self.still_config = combined_cfg
            self.preview_config = combined_cfg
            if debug_touch:
                print("DEBUG: Combined configuration created:", {"main": cfg_kwargs["main"]["size"], "lores": cfg_kwargs["lores"]["size"], "raw": cfg_kwargs.get("raw")})
        except Exception as e:
            if debug_touch:
                print("DEBUG: Combined configuration failed:", e)
                print("DEBUG: Trying fallback create_still_configuration() (no kwargs).")
            try:
                fallback_still = self.picam2.create_still_configuration()
                preview_cfg = self.picam2.create_preview_configuration({"size": preview_size})
                self.preview_config = preview_cfg
                self.still_config = fallback_still
                self.picam2.configure(preview_cfg)
                if debug_touch:
                    print("DEBUG: Fallback configured preview (will use mode-switching for still captures).")
            except Exception as e2:
                print("Failed to create camera configuration:", e2)
                raise

        if self.preview_config is not None:
            try:
                self.picam2.configure(self.preview_config)
            except Exception as e:
                print("Failed to configure Picamera2 with preview_config:", e)
                raise

        # overlay / UI state
        self.overlay_lock = threading.Lock()
        self.button_geom = (0, 0, 0)
        self.overlay_pressed = False

        self.debug_touch = debug_touch
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.swap_axes = swap_axes

        # touch initialization (unchanged)
        self.touch_dev = None
        self.touch_mapping = None
        if touchscreen_device_path:
            if not EVDEV_AVAILABLE:
                print(f"Touch device specified ({touchscreen_device_path}) but the Python 'evdev' package is not installed.")
                print("Install it with: sudo apt install python3-evdev   (or: pip3 install evdev)")
                print("Continuing without touch support.")
                self.touch_dev = None
            else:
                try:
                    self.touch_dev = InputDevice(touchscreen_device_path)
                except Exception as e:
                    print("Could not open specified touchscreen device:", e)
                    self.touch_dev = None
        else:
            if EVDEV_AVAILABLE:
                self.touch_dev = find_touchscreen_device()

        if self.touch_dev:
            abs_range = get_abs_range(self.touch_dev)
            import json, os
            calib_path = os.path.expanduser("~/.camera_touch_calib.json")
            if os.path.exists(calib_path):
                try:
                    with open(calib_path, "r") as f:
                        c = json.load(f)
                    if all(k in c for k in ("x_min","x_max","y_min","y_max")):
                        abs_range['x_min'] = int(c['x_min'])
                        abs_range['x_max'] = int(c['x_max'])
                        abs_range['y_min'] = int(c['y_min'])
                        abs_range['y_max'] = int(c['y_max'])
                        print("Loaded touch calibration from", calib_path, "->", {k: abs_range[k] for k in ('x_min','x_max','y_min','y_max')})
                except Exception as ex:
                    print("Failed to load calibration file:", ex)
            abs_range['preview_size'] = preview_size
            abs_range['debug'] = self.debug_touch
            abs_range['invert_x'] = self.invert_x
            abs_range['invert_y'] = self.invert_y
            abs_range['swap_axes'] = self.swap_axes
            abs_range['button_geom'] = self.button_geom
            self.touch_mapping = abs_range
            print(f"Using touchscreen device: {self.touch_dev.path} mapping: {abs_range}")
        else:
            if EVDEV_AVAILABLE:
                print("No touchscreen device found; will use keyboard 'c' fallback.")
            else:
                print("evdev not available; touch disabled. Use 'c' to capture.")

        self.stop_event = threading.Event()
        self.metadata_thread = None
        self.touch_thread = None
        self.key_thread = None
        self._last_metadata = {}

    def start(self):
        # print effective quality setting so you can verify it on startup
        try:
            q = self.picam2.options.get('quality', None)
            if q is not None:
                print(f"JPEG encoder quality set to: {q}")
        except Exception:
            pass

        try:
            self.picam2.start_preview(Preview.DRM, x=0, y=0,
                                      width=self.preview_size[0], height=self.preview_size[1])
            print("Preview started with Preview.DRM.")
        except Exception as e:
            print("Warning: Preview.DRM failed to start:", e)
            print("Continuing without rendered preview. Captures still available.")
        try:
            self.picam2.start()
        except Exception as e:
            print("Failed to start camera:", e)
            raise

        with self.overlay_lock:
            arr, geom = compose_overlay_array(self.preview_size, ["ISO —", "Shutter —", "WB —"], pressed=False)
            self.button_geom = geom
            try:
                self.picam2.set_overlay(arr)
            except Exception:
                pass

        if self.touch_mapping is not None:
            self.touch_mapping['button_geom'] = self.button_geom

        self.metadata_thread = threading.Thread(target=self._metadata_updater, daemon=True)
        self.metadata_thread.start()

        if self.touch_dev and EVDEV_AVAILABLE:
            self.touch_mapping['button_geom'] = self.button_geom
            self.touch_thread = threading.Thread(target=touch_listener,
                                                 args=(self.touch_dev, self.touch_mapping, self._on_touch, self.stop_event),
                                                 daemon=True)
            self.touch_thread.start()
            if self.debug_touch:
                print("DEBUG: touch thread started for", self.touch_dev.path)

        self.key_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.key_thread.start()

        print("Camera started. Press Ctrl-C to quit. Press 'c' + Enter to capture.")

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping...")
            self.stop()
        finally:
            self.stop()

    def stop(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.stop_preview()
            except Exception:
                pass
            print("Stopped camera and preview.")

    def _metadata_updater(self):
        while not self.stop_event.is_set():
            try:
                metadata = self.picam2.capture_metadata()
            except Exception:
                time.sleep(0.2)
                continue

            if not isinstance(metadata, dict):
                time.sleep(0.1)
                continue

            self._last_metadata = metadata

            exposure = metadata.get("ExposureTime") or metadata.get("exposure_time") or None
            analogue_gain = metadata.get("AnalogueGain") or metadata.get("analogue_gain") or None

            shutter_s = format_shutter(exposure)
            iso_s = format_iso(analogue_gain)
            wb_s = format_wb(metadata)

            lines = [iso_s, f"Shutter {shutter_s}", f"WB {wb_s}"]

            with self.overlay_lock:
                arr, geom = compose_overlay_array(self.preview_size, lines, pressed=self.overlay_pressed)
                self.button_geom = geom
                if self.touch_mapping is not None:
                    self.touch_mapping['button_geom'] = self.button_geom
                try:
                    self.picam2.set_overlay(arr)
                except Exception:
                    pass

            time.sleep(0.12)

    def _on_touch(self, x_px, y_px):
        cx, cy, r = self.button_geom
        pad = globals().get("BUTTON_TOUCH_PADDING_PX", 0)
        dx = x_px - cx
        dy = y_px - cy
        rp = r + pad
        inside = (dx * dx + dy * dy) <= (rp * rp)
        if self.debug_touch:
            print(f"DEBUG: _on_touch called with ({x_px},{y_px}). button (cx,cy,r)={self.button_geom}. pad={pad}. inside={inside}")
        if inside:
            if self.debug_touch:
                print("DEBUG: TOUCH INSIDE BUTTON -> triggering capture")
            self._capture_with_feedback()
        else:
            if self.debug_touch:
                print("DEBUG: TOUCH outside button")

    def _keyboard_listener(self):
        while not self.stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip().lower() == "c":
                    self._capture_with_feedback()
            except Exception:
                break

    def _capture_with_feedback(self):
        def do_capture():
            fname = self._timestamped_filename()
            try:
                if hasattr(self.picam2, "switch_mode_and_capture_file") and self.still_config is not None and self.preview_config is not None and self.still_config is not self.preview_config:
                    if self.debug_touch:
                        print("DEBUG: Using switch_mode_and_capture_file for capture:", fname)
                    self.picam2.switch_mode_and_capture_file(self.still_config, fname)
                else:
                    if self.debug_touch:
                        print("DEBUG: Using capture_file (combined config expected) ->", fname)
                    self.picam2.capture_file(fname)

                print("Captured:", fname)
            except Exception as e:
                print("Capture failed:", e)

            time.sleep(FLASH_MS / 1000.0)
            with self.overlay_lock:
                self.overlay_pressed = False
                meta = self._last_metadata or {}
                exposure = meta.get("ExposureTime")
                analogue_gain = meta.get("AnalogueGain")
                lines = [format_iso(analogue_gain), f"Shutter {format_shutter(exposure)}", f"WB {format_wb(meta)}"]
                arr, geom = compose_overlay_array(self.preview_size, lines, pressed=False)
                self.button_geom = geom
                if self.touch_mapping is not None:
                    self.touch_mapping['button_geom'] = self.button_geom
                try:
                    self.picam2.set_overlay(arr)
                except Exception:
                    pass

        with self.overlay_lock:
            self.overlay_pressed = True
            meta = self._last_metadata or {}
            exposure = meta.get("ExposureTime")
            analogue_gain = meta.get("AnalogueGain")
            lines = [format_iso(analogue_gain), f"Shutter {format_shutter(exposure)}", f"WB {format_wb(meta)}"]
            arr, geom = compose_overlay_array(self.preview_size, lines, pressed=True)
            self.button_geom = geom
            if self.touch_mapping is not None:
                self.touch_mapping['button_geom'] = self.button_geom
            try:
                self.picam2.set_overlay(arr)
            except Exception:
                pass

        t = threading.Thread(target=do_capture, daemon=True)
        t.start()

    def _timestamped_filename(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self.outdir / f"{ts}.jpg")

# CLI
def parse_preview_size(s):
    if "x" in s:
        w, h = s.split("x", 1)
        return (int(w), int(h))
    raise argparse.ArgumentTypeError("Preview size must be WIDTHxHEIGHT, e.g. 640x480")

def parse_still_size(s):
    if not s:
        return None
    if "x" in s:
        w, h = s.split("x", 1)
        return (int(w), int(h))
    raise argparse.ArgumentTypeError("Still size must be WIDTHxHEIGHT, e.g. 4056x3040")

def main():
    parser = argparse.ArgumentParser(description="HQ camera preview with capture button + metadata overlay (combined lores + main streams)")
    parser.add_argument("--preview-size", type=parse_preview_size, default=f"{DEFAULT_PREVIEW_SIZE[0]}x{DEFAULT_PREVIEW_SIZE[1]}",
                        help="Preview size WIDTHxHEIGHT (default: 640x480)")
    parser.add_argument("--outdir", type=str, default="/media/DCIM", help="Directory to save captures")
    parser.add_argument("--touch-device", type=str, default="/dev/input/event7", help="evdev touchscreen device path (optional)")
    parser.add_argument("--debug-touch", action="store_true", help="Enable touch debug logging")
    parser.add_argument("--invert-x", action="store_true", help="Invert mapped touch X coordinate")
    parser.add_argument("--invert-y", action="store_true", help="Invert mapped touch Y coordinate")
    parser.add_argument("--swap-axes", action="store_true", help="Swap X and Y axes (for rotated screens)")
    parser.add_argument("--buffer-count", type=int, default=6, help="Buffer count for camera pipeline (reduce if allocation fails)")
    parser.add_argument("--no-raw", action="store_true", help="Do not request full-res raw stream (saves memory)")
    parser.add_argument("--still-size", type=parse_still_size, default=None, help="Force still/main size WIDTHxHEIGHT (optional)")
    parser.add_argument("--jpeg-quality", type=int, default=100, help="JPEG encoder quality (1-100). Will be clamped to 95 max in practice.")
    args = parser.parse_args()

    preview_size = args.preview_size if isinstance(args.preview_size, tuple) else parse_preview_size(args.preview_size)
    cam = HQCameraWithButton(preview_size=preview_size, outdir=args.outdir, touchscreen_device_path=args.touch_device,
                             debug_touch=args.debug_touch, invert_x=args.invert_x, invert_y=args.invert_y, swap_axes=args.swap_axes,
                             buffer_count=args.buffer_count, request_raw=(not args.no_raw), still_size=args.still_size,
                             jpeg_quality=args.jpeg_quality)
    cam.start()

if __name__ == "__main__":
    main()
