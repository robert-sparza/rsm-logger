"""Pitch model and camera -> pitch homography.

Pitch coordinates are in metres with the origin at the far-left corner flag
as seen from the camera, x along the touchline (0..length, left to right) and
y across the pitch (0 = far touchline, width = near touchline, closest to the camera).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Pitch:
    length: float = 105.0
    width: float = 68.0
    # Markings default to FIFA 11-a-side sizes; small-sided pitches differ (measure yours).
    circle_radius: float = 9.15
    box_depth: float = 16.5
    box_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    penalty_spot: float = 11.0

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named points that are easy to click on a frame.

        "far" = the touchline away from the camera, "near" = the one closest to it;
        left/right as seen in the video.
        """
        L, W = self.length, self.width
        pa_d, pa_w = self.box_depth, self.box_width
        ga_d, ga_w = self.goal_area_depth, self.goal_area_width
        r, ps = self.circle_radius, self.penalty_spot
        cy = W / 2
        return {
            "far_left_corner": (0, 0),
            "far_right_corner": (L, 0),
            "near_right_corner": (L, W),
            "near_left_corner": (0, W),
            "halfway_far_end": (L / 2, 0),
            "halfway_near_end": (L / 2, W),
            "center_spot": (L / 2, cy),
            "center_circle_far": (L / 2, cy - r),
            "center_circle_near": (L / 2, cy + r),
            "left_box_far_goalline": (0, cy - pa_w / 2),
            "left_box_far_corner": (pa_d, cy - pa_w / 2),
            "left_box_near_corner": (pa_d, cy + pa_w / 2),
            "left_box_near_goalline": (0, cy + pa_w / 2),
            "left_6yd_far_corner": (ga_d, cy - ga_w / 2),
            "left_6yd_near_corner": (ga_d, cy + ga_w / 2),
            "left_penalty_spot": (ps, cy),
            "right_box_far_goalline": (L, cy - pa_w / 2),
            "right_box_far_corner": (L - pa_d, cy - pa_w / 2),
            "right_box_near_corner": (L - pa_d, cy + pa_w / 2),
            "right_box_near_goalline": (L, cy + pa_w / 2),
            "right_6yd_far_corner": (L - ga_d, cy - ga_w / 2),
            "right_6yd_near_corner": (L - ga_d, cy + ga_w / 2),
            "right_penalty_spot": (L - ps, cy),
        }

    def marking_polylines(self, n: int = 40) -> list[np.ndarray]:
        """Every painted line as a polyline in pitch metres (for drawing/checking)."""
        L, W, cy = self.length, self.width, self.width / 2
        a = np.linspace(0, 2 * np.pi, n)
        lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]),
                 np.array([[L / 2, 0], [L / 2, W]]),
                 np.c_[L / 2 + self.circle_radius * np.cos(a), cy + self.circle_radius * np.sin(a)]]
        for x0, s in ((0, 1), (L, -1)):
            for d, w in ((self.box_depth, self.box_width), (self.goal_area_depth, self.goal_area_width)):
                lines.append(np.array([[x0, cy - w / 2], [x0 + s * d, cy - w / 2],
                                       [x0 + s * d, cy + w / 2], [x0, cy + w / 2]]))
        # Resample straight segments so they bend correctly under perspective.
        out = []
        for pl in lines:
            pts = [np.linspace(p, q, n // 4) for p, q in zip(pl[:-1], pl[1:])]
            out.append(np.concatenate(pts).astype(np.float32))
        return out


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

    def on_pitch(self, xy: np.ndarray, margin: float = 3.0) -> np.ndarray:
        """Boolean mask: which pitch positions lie inside the pitch plus `margin` metres."""
        xy = np.asarray(xy).reshape(-1, 2)
        return ((xy[:, 0] >= -margin) & (xy[:, 0] <= self.pitch.length + margin)
                & (xy[:, 1] >= -margin) & (xy[:, 1] <= self.pitch.width + margin))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "H": self.H.tolist(),
            "pitch": asdict(self.pitch),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        d = json.loads(Path(path).read_text())
        return cls(H=np.array(d["H"], dtype=np.float64), pitch=Pitch(**d["pitch"]))
