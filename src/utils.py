import time
import os
import platform
import ctypes
from importlib.util import find_spec

def setup_windows_fix():
    """Windows 11 Temp Fix For pytorch 2.9.0 + cuda 12.9"""
    if platform.system() == "Windows":
        try:
            if (spec := find_spec("torch")) and spec.origin and os.path.exists(
                dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            ):
                ctypes.CDLL(os.path.normpath(dll_path))
        except Exception:
            pass

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)