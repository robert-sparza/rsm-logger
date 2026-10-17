"""Detection + multi-object tracking on top of Ultralytics YOLO + ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

# COCO class ids used by the stock YOLO weights. A fine-tuned football model
# (see README, Phase 2) will have its own ids; set them in config.yaml.
COCO_PERSON, COCO_BALL = 0, 32


@dataclass
class Detection:
    track_id: int
    cls: str            # "player" | "ball"
    box: tuple[float, float, float, float]
    conf: float

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: the point that touches the pitch."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


class Tracker:
    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.25,
                 player_ids=(COCO_PERSON,), ball_ids=(COCO_BALL,), device: str | int | None = None):
        from ultralytics import YOLO  # imported lazily so tests don't need it
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.player_ids, self.ball_ids = set(player_ids), set(ball_ids)

    def step(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz,
                               conf=self.conf, device=self.device, verbose=False,
                               classes=sorted(self.player_ids | self.ball_ids))[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else -np.ones(len(boxes), int)
        best_ball: Detection | None = None
        for box, c, p, tid in zip(boxes, classes, confs, ids):
            if c in self.ball_ids:
                # There is only one ball: keep the most confident candidate.
                if best_ball is None or p > best_ball.conf:
                    best_ball = Detection(-1, "ball", tuple(box), float(p))
            elif c in self.player_ids and tid >= 0:
                out.append(Detection(int(tid), "player", tuple(box), float(p)))
        if best_ball:
            out.append(best_ball)
        return out


def read_frames(source: str, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, seconds, frame) from a file path or a stream URL (rtmp://, rtsp://)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()
