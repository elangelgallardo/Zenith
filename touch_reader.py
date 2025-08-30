#!/usr/bin/env python3
# touch_reader.py — minimal evdev test
import sys
try:
    from evdev import InputDevice, ecodes
except Exception as e:
    print("evdev not installed:", e); sys.exit(2)

if len(sys.argv) < 2:
    print("usage: sudo python3 touch_reader.py /dev/input/event11")
    sys.exit(1)

devpath = sys.argv[1]
dev = InputDevice(devpath)
print("Opened device:", devpath, "name:", dev.name)
print("Capabilities (short):", dev.capabilities(verbose=True))
print("Listening... touch the screen (Ctrl-C to quit)\n")

last_x = None
last_y = None
try:
    for ev in dev.read_loop():
        if ev.type == ecodes.EV_ABS:
            name = ecodes.ABS.get(ev.code, ev.code)
            print(f"EV_ABS {name} ({ev.code}) = {ev.value}")
            if ev.code in (ecodes.ABS_MT_POSITION_X, ecodes.ABS_X):
                last_x = ev.value
            elif ev.code in (ecodes.ABS_MT_POSITION_Y, ecodes.ABS_Y):
                last_y = ev.value
        elif ev.type == ecodes.EV_KEY:
            name = ecodes.KEY.get(ev.code, ev.code)
            print(f"EV_KEY {name} ({ev.code}) = {ev.value}")
            if ev.code == ecodes.BTN_TOUCH and ev.value == 1:
                print("  BTN_TOUCH down; last_x,last_y:", last_x, last_y)
        elif ev.type == ecodes.EV_SYN:
            # often printed as a sync packet; ignore or print for timing
            pass
        else:
            print("OTHER ev", ev.type, ev.code, ev.value)
except KeyboardInterrupt:
    print("\nbye")
