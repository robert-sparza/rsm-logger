"""Pick frames from your videos for labelling in Roboflow (Phase 2).

Usage:
    python scripts/extract_frames.py data/videos/training.mp4 data/videos/match.mp4 --n 300 --out runs/label_batch1
    python scripts/extract_frames.py data/videos/training.mp4 --calib runs/calib_small.json

Writes runs/label_batch1/
    images/*.jpg    the chosen frames (max 1920 px wide)
    labels/*.txt    boxes pre-drawn by the current detector (YOLO format)
    data.yaml       class names: player, ball, cone
and runs/label_batch1.zip to drag into Roboflow (see PHASE2.md).

About half the frames are the "hard" moments (players bunched or overlapping, a
ball near someone's feet, cones around); the rest are spread at random.
With --calib, people outside the pitch (coach, bench) are not pre-labelled.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import ROOT, load_config, weights_path  # noqa: E402
from soccer_analyzer.ball import cone_colored  # noqa: E402
from soccer_analyzer.dataset import BALL, CONE, PLAYER, frame_score, select_frames, write_data_yaml, yolo_lines  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.tracking import BALL_NAMES, PLAYER_NAMES, class_ids  # noqa: E402


def detect(model, frame, imgsz, device, player_ids, ball_ids, hom, margin):
    """Boxes per class for one frame, using the current (stock or trained) model."""
    res = model.predict(frame, imgsz=imgsz, conf=0.25, device=device, verbose=False,
                        classes=sorted(player_ids | ball_ids | cone_ids(model)))[0]
    players, balls, cones = [], [], []
    for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
        box = tuple(float(v) for v in box)
        if c in player_ids:
            if hom is not None:
                foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
                if not hom.on_pitch(hom.to_pitch(foot), margin)[0]:
                    continue                         # coach / bench / spectators: not labelled
            players.append(box)
        elif c in cone_ids(model) or cone_colored(frame, box):
            cones.append(box)
        else:
            balls.append(box)
    return players, balls, cones


def cone_ids(model) -> set[int]:
    return class_ids(model.names, ("cone",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="runs/label_batch1")
    ap.add_argument("--n", type=int, default=300, help="total frames, shared between the videos by length")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between candidate frames")
    ap.add_argument("--max-width", type=int, default=1920, help="resize saved frames to at most this width")
    ap.add_argument("--calib", help="calibration json: skip pre-labelling people outside the pitch")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = load_config(Path(args.config)).get("model", {})
    model = YOLO(weights_path(cfg.get("weights", "yolo11m.pt")))
    imgsz, device = cfg.get("imgsz", 1280), cfg.get("device")
    player_ids, ball_ids = class_ids(model.names, PLAYER_NAMES), class_ids(model.names, BALL_NAMES)
    hom = Homography.load(args.calib) if args.calib else None

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()

    durations = []
    for v in args.videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {v}")
        durations.append(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30))
        cap.release()
    shares = np.maximum(1, np.round(args.n * np.array(durations) / sum(durations))).astype(int)

    total = 0
    for video, n_video in zip(args.videos, shares):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(args.every * fps)))
        stem = Path(video).stem
        print(f"{stem}: scoring a frame every {args.every}s ...")
        candidates, dets = [], {}
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    players, balls, cones = detect(model, frame, imgsz, device, player_ids, ball_ids,
                                                   hom, 1.5)
                    candidates.append((idx, idx / fps, frame_score(players, balls, cones)))
                    dets[idx] = (players, balls, cones)
            idx += 1
        cap.release()

        chosen = set(select_frames(candidates, n_video))
        cap = cv2.VideoCapture(video)
        idx = 0
        while chosen:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in chosen:
                chosen.discard(idx)
                h, w = frame.shape[:2]
                scale = min(1.0, args.max_width / w)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                players, balls, cones = dets[idx]
                boxes = ([(PLAYER, b) for b in players] + [(BALL, b) for b in balls] + [(CONE, b) for b in cones])
                boxes = [(c, tuple(v * scale for v in b)) for c, b in boxes]
                name = f"{stem}_{idx:06d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / f"{name}.txt").write_text("\n".join(yolo_lines(boxes, frame.shape[1], frame.shape[0])))
                total += 1
            idx += 1
        cap.release()

    write_data_yaml(out)
    zip_path = shutil.make_archive(str(out), "zip", out)
    print(f"\n{total} frames written to {out}")
    print(f"Upload this to Roboflow: {zip_path}")


if __name__ == "__main__":
    main()
