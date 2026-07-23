"""Fast screen capture using mss."""

import numpy as np
import mss


class ScreenCapture:
    def __init__(self):
        self._sct = mss.mss()

    def capture(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        monitor = {
            "left": max(0, left),
            "top": max(0, top),
            "width": max(1, width),
            "height": max(1, height),
        }
        sct_img = self._sct.grab(monitor)
        arr = np.array(sct_img)
        return arr[:, :, :3]  # BGRA -> BGR
