"""Pick the real ball among YOLO's "sports ball" candidates.

At training sessions there are cones, spare balls and balls behind the goals,
and the stock detector reports several of them per frame. We drop candidates
that are off the pitch, cone-coloured (vivid orange/yellow), or have sat in the
same spot for a few seconds (a ball at the coach's feet), then follow the
candidate closest to where the ball was last seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def cone_colored(frame: np.ndarray, box, min_share: float = 0.12) -> bool:
    """True if the box is mostly vivid orange/yellow/red like a training cone.

    Measured on night footage: cones 20-43% vivid pixels, a white match ball 0%.
    Turn off (config ball.reject_cone_colours) if you play with an orange or yellow ball.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    vivid = (hsv[:, 1] >= 120) & (hsv[:, 2] >= 120) & ((hsv[:, 0] <= 40) | (hsv[:, 0] >= 165))
    return bool(vivid.mean() >= min_share)


@dataclass
class _StaticSpot:
    x: float
    y: float
    first: int
    last: int


class BallSelector:
    def __init__(self, static_frames: int = 90, radius_px: float = 12.0, gap_frames: int = 150,
                 memory_frames: int = 30):
        self.static_frames = static_frames      # ~3 s at 30 fps without moving = not the ball in play
        # Cones are detected on and off (confidence near the threshold), so a spot keeps its
        # history through dropouts of up to gap_frames (~5 s) instead of resetting.
        self.radius_px = radius_px
        self.gap_frames = gap_frames
        self.memory_frames = memory_frames
        self.spots: list[_StaticSpot] = []
        self.last: tuple[int, float, float] | None = None   # frame, x, y of last chosen ball

    def _is_static(self, frame_idx: int, x: float, y: float) -> bool:
        for s in self.spots:
            if np.hypot(s.x - x, s.y - y) <= self.radius_px:
                if frame_idx - s.last > self.gap_frames:
                    s.first = frame_idx
                s.last = frame_idx
                return s.last - s.first >= self.static_frames
        self.spots.append(_StaticSpot(x, y, frame_idx, frame_idx))
        return False

    def select(self, frame_idx: int, candidates):
        """candidates: list of (detection, on_pitch). Returns the chosen detection or None."""
        self.spots = [s for s in self.spots if frame_idx - s.last <= self.gap_frames]
        eligible = []
        for det, on_pitch in candidates:
            x1, y1, x2, y2 = det.box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            static = self._is_static(frame_idx, cx, cy)
            if on_pitch and not static:
                eligible.append((det, cx, cy))
        if not eligible:
            return None
        if self.last is not None and frame_idx - self.last[0] <= self.memory_frames:
            _, lx, ly = self.last
            det, cx, cy = min(eligible, key=lambda e: np.hypot(e[1] - lx, e[2] - ly))
        else:
            det, cx, cy = max(eligible, key=lambda e: e[0].conf)
        self.last = (frame_idx, cx, cy)
        return det
