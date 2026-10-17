"""Pitch model and camera -> pitch homography.

Pitch coordinates are in metres with the origin at the top-left corner flag
(as seen from the camera), x along the touchline (0..length) and y across the
pitch (0..width).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Pitch:
    length: float = 105.0
    width: float = 68.0

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named points that are easy to click on a frame."""
        L, W = self.length, self.width
        pa_d, pa_w = 16.5, 40.32      # penalty area depth / width
        ga_d, ga_w = 5.5, 18.32       # goal area depth / width
        cy = W / 2
        return {
            "corner_top_left": (0, 0),
            "corner_top_right": (L, 0),
            "corner_bottom_right": (L, W),
            "corner_bottom_left": (0, W),
            "halfway_top": (L / 2, 0),
            "halfway_bottom": (L / 2, W),
            "center_spot": (L / 2, cy),
            "center_circle_top": (L / 2, cy - 9.15),
            "center_circle_bottom": (L / 2, cy + 9.15),
            "left_box_top_line": (0, cy - pa_w / 2),
            "left_box_top_corner": (pa_d, cy - pa_w / 2),
            "left_box_bottom_corner": (pa_d, cy + pa_w / 2),
            "left_box_bottom_line": (0, cy + pa_w / 2),
            "left_6yd_top_corner": (ga_d, cy - ga_w / 2),
            "left_6yd_bottom_corner": (ga_d, cy + ga_w / 2),
            "left_penalty_spot": (11, cy),
            "right_box_top_line": (L, cy - pa_w / 2),
            "right_box_top_corner": (L - pa_d, cy - pa_w / 2),
            "right_box_bottom_corner": (L - pa_d, cy + pa_w / 2),
            "right_box_bottom_line": (L, cy + pa_w / 2),
            "right_6yd_top_corner": (L - ga_d, cy - ga_w / 2),
            "right_6yd_bottom_corner": (L - ga_d, cy + ga_w / 2),
            "right_penalty_spot": (L - 11, cy),
        }


@dataclass
class Homography:
    """Maps image pixels to pitch metres. Valid only while the camera is static."""
    H: np.ndarray
    pitch: Pitch = field(default_factory=Pitch)

    @classmethod
    def fit(cls, image_pts, pitch_pts, pitch: Pitch | None = None) -> "Homography":
        image_pts = np.asarray(image_pts, dtype=np.float32)
        pitch_pts = np.asarray(pitch_pts, dtype=np.float32)
        if len(image_pts) < 4:
            raise ValueError("Need at least 4 point correspondences")
        H, _ = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC if len(image_pts) > 4 else 0)
        if H is None:
            raise ValueError("Homography could not be estimated (points collinear?)")
        return cls(H=H, pitch=pitch or Pitch())

    def to_pitch(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "H": self.H.tolist(),
            "pitch": {"length": self.pitch.length, "width": self.pitch.width},
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        d = json.loads(Path(path).read_text())
        return cls(H=np.array(d["H"], dtype=np.float64), pitch=Pitch(**d["pitch"]))
