"""Detection + multi-object tracking on top of Ultralytics YOLO + ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Class names understood as players / balls: the stock COCO weights call them
# "person" / "sports ball", our own trained model (Phase 2) "player" / "ball".
# Ids are looked up from the model's own names, so switching weights needs no id edits.
PLAYER_NAMES = ("person", "player", "goalkeeper")
BALL_NAMES = ("sports ball", "ball")
DEFAULT_TRACKER = str(Path(__file__).resolve().parents[1] / "trackers" / "bytetrack_soccer.yaml")


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


def class_ids(model_names: dict[int, str], wanted: tuple[str, ...] | list[str]) -> set[int]:
    """Ids of the model classes whose name is in `wanted` (case-insensitive)."""
    wanted = {w.lower() for w in wanted}
    return {i for i, n in model_names.items() if n.lower() in wanted}


def make_mot_tracker(tracker_yaml: str):
    """Ultralytics' ByteTrack/BoT-SORT, built from a tracker yaml, for us to feed directly."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
    cfg.device = None
    cls = TRACKER_MAP[cfg.tracker_type]
    try:
        return cls(args=cfg)
    except TypeError:                       # older Ultralytics versions also take frame_rate
        return cls(args=cfg, frame_rate=30)


class Tracker:
    """Detects players and balls; tracks only the players.

    The ball is NOT sent through the multi-object tracker: in Ultralytics' track
    mode only detections the tracker has linked into a track are returned, and a
    small fast ball barely overlaps itself between frames, so most ball
    detections were silently dropped. We detect once, track the players
    ourselves, and hand every ball detection to BallSelector.
    """

    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.25,
                 player_names=PLAYER_NAMES, ball_names=BALL_NAMES, device: str | int | None = None,
                 tracker: str = DEFAULT_TRACKER, ball_conf: float = 0.15):
        from ultralytics import YOLO  # imported lazily so tests don't need it
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.ball_conf, self.device = imgsz, conf, ball_conf, device
        self.player_ids = class_ids(self.model.names, player_names)
        self.ball_ids = class_ids(self.model.names, ball_names)
        if not self.player_ids:
            raise ValueError(f"{weights} has no player class; its classes are {list(self.model.names.values())}")
        self.mot = make_mot_tracker(tracker)

    def step(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=min(self.conf, self.ball_conf), device=self.device,
                                 verbose=False, classes=sorted(self.player_ids | self.ball_ids))[0]
        out: list[Detection] = []
        boxes = res.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        players = boxes[np.isin(cls, list(self.player_ids)) & (boxes.conf >= self.conf)]
        tracks = self.mot.update(players, frame)            # rows: x1, y1, x2, y2, id, score, cls, idx
        for t in np.asarray(tracks).reshape(-1, 8) if len(tracks) else []:
            out.append(Detection(int(t[4]), "player", tuple(float(v) for v in t[:4]), float(t[5])))
        balls = np.isin(cls, list(self.ball_ids)) & (boxes.conf >= self.ball_conf)
        for box, p in zip(boxes.xyxy[balls], boxes.conf[balls]):
            out.append(Detection(-1, "ball", tuple(float(v) for v in box), float(p)))
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


class LiveStream:
    """Reads a live stream (rtmp://, rtsp://, webcam) on a background thread.

    Analysis is slower than the stream (~16 vs 30 fps), so iterating yields only
    the newest frame and skips the backlog; otherwise the live view drifts minutes
    behind the game. Every frame is still written to `record_path` if given.
    Waits up to `wait_s` for the stream to start, and reconnects if the phone's
    stream drops for less than `reconnect_s` seconds.
    """

    def __init__(self, source: str, record_path: str | None = None, reconnect_s: float = 30.0,
                 wait_s: float = 300.0):
        import threading
        self.source, self.record_path = source, record_path
        self.reconnect_s, self.wait_s = reconnect_s, wait_s
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._idx = -1
        self._running = True
        self._writer = None
        self.fps = 30.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        import time
        deadline = time.time() + self.wait_s
        while self._running:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                return cap
            cap.release()
            if time.time() >= deadline:
                return None
            print(f"Waiting for stream {self.source} (start streaming from the phone) ...")
            time.sleep(2)
        return None

    def _run(self):
        import time
        is_stream = "://" in self.source
        cap = self._open()
        dropped_at = None
        while self._running and cap is not None:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                if not is_stream:
                    break                      # end of a file: no reconnect
                dropped_at = dropped_at or time.time()
                if time.time() - dropped_at > self.reconnect_s:
                    print("Stream ended.")
                    break
                print("Stream dropped, reconnecting ...")
                time.sleep(1)
                cap = cv2.VideoCapture(self.source)
                continue
            dropped_at = None
            if self.record_path:
                if self._writer is None:
                    h, w = frame.shape[:2]
                    self._writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self._writer.write(frame)
            with self._lock:
                self._frame, self._idx = frame, self._idx + 1
        if cap is not None:
            cap.release()
        self._running = False

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import time
        t0, last = time.time(), -1
        while self._running or self._idx > last:
            with self._lock:
                frame, idx = self._frame, self._idx
            if idx == last or frame is None:
                time.sleep(0.003)
                continue
            last = idx
            yield idx, time.time() - t0, frame

    def close(self):
        self._running = False
        self._thread.join(timeout=5)
        if self._writer is not None:
            self._writer.release()
