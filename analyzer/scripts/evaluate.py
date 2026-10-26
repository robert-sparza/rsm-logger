"""Compare detectors on a video: is the ball in play found, how many players, cones mistaken for balls.

Usage:
    python scripts/evaluate.py data/videos/training.mp4 --models yolo11m.pt runs/train/soccer-v1b/weights/best.pt \
        --calib runs/calib_small.json

Each model runs over the video (every --stride-th frame) with the same ball
filter as analyze.py, so spare balls, the coach's ball and cones don't count:

    ball in play     % of frames where a moving ball on the pitch was picked
    near a player    of those, % within 2 m of a player (needs --calib) - a check
                     that it really is the play ball
    players/frame    people on the pitch per frame (steadier +- is better)
    cones as ball    cone-coloured "balls" per frame (lower is better)

No hand-made ground truth is needed, so these are comparisons between models
on the same video, not absolute accuracy. A video the model was NOT trained on
gives the honest picture.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import ROOT, pitch_region, weights_path  # noqa: E402
from soccer_analyzer.ball import BallSelector, cone_colored  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.tracking import BALL_NAMES, PLAYER_NAMES, Detection, class_ids, merge_boxes, read_frames, tile_grid, tiled_boxes  # noqa: E402


def still_picks(picks, candidates, fps: float, still_s: float = 1.5, max_gap_s: float = 0.5) -> np.ndarray:
    """With hindsight: which picks are a ball sitting still.

    picks / candidates: rows of (frame, cx, cy, size); candidates are all ball
    detections. A pick is "still" if detections stay at its spot for >= still_s
    around it (gaps up to max_gap_s allowed). The live filter needs ~1.5 s to
    confirm a resting ball and then stops picking it, so its first picks of that
    ball would otherwise count as "in play".
    """
    if not len(picks):
        return np.zeros(0, bool)
    cf, cx, cy = (np.asarray(c, float) for c in list(zip(*candidates))[:3]) if len(candidates) else ([],) * 3
    still = np.zeros(len(picks), bool)
    for i, (f, x, y, s) in enumerate(picks):
        frames = np.unique(cf[np.hypot(cx - x, cy - y) <= max(12.0, 0.5 * s)]) if len(cf) else np.array([f])
        if f not in frames:
            frames = np.sort(np.append(frames, f))
        k = int(np.searchsorted(frames, f))
        lo = hi = k                                   # grow the run of detections around this pick
        while lo > 0 and frames[lo] - frames[lo - 1] <= max_gap_s * fps:
            lo -= 1
        while hi < len(frames) - 1 and frames[hi + 1] - frames[hi] <= max_gap_s * fps:
            hi += 1
        still[i] = frames[hi] - frames[lo] >= still_s * fps
    return still


def evaluate_model(name: str, video: str, imgsz: int, stride: int, hom, device, max_seconds=None,
                   conf: float = 0.25, ball_conf: float = 0.15, tile_size: int = 0) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights_path(name))
    pids, bids = class_ids(model.names, PLAYER_NAMES), class_ids(model.names, BALL_NAMES)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    selector = BallSelector(static_frames=int(1.5 * fps), gap_frames=int(5 * fps), memory_frames=int(fps),
                            confirmed_memory_frames=int(60 * fps))
    n, players, cones = 0, [], 0
    region = pitch_region(hom) if (hom is not None and tile_size) else None
    tile_cache: dict = {}

    def tracker_tiles(frame):
        key = frame.shape[:2]
        if key not in tile_cache:
            h, w = key
            roi = None
            if region is not None:
                pad = 0.03 * max(w, h)
                roi = (region[:, 0].min() - pad, region[:, 1].min() - pad, region[:, 0].max() + pad, region[:, 1].max() + pad)
            tile_cache[key] = tile_grid(w, h, tile_size, 0.2, roi)
        return tile_cache[key]

    picks, picks_near, candidates = [], [], []
    for idx, t, frame in read_frames(video, stride):
        if max_seconds and t > max_seconds:
            break
        res = model.predict(frame, imgsz=imgsz, conf=min(conf, ball_conf), device=device, verbose=False)[0]
        boxes = res.boxes.cpu().numpy()
        feet_xy = []
        ball_boxes = []
        for box, c, p in zip(boxes.xyxy, boxes.cls.astype(int), boxes.conf):
            foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
            xy = hom.to_pitch(foot)[0] if hom is not None else None
            on = hom.on_pitch(xy[None], 1.5)[0] if hom is not None else True
            if c in pids and p >= conf and on:
                feet_xy.append(xy)
            elif c in bids and p >= ball_conf:
                ball_boxes.append((tuple(float(v) for v in box), float(p)))
        if tile_size:
            tiles = tracker_tiles(frame)
            ball_boxes = merge_boxes(ball_boxes + tiled_boxes(model, frame, bids, ball_conf, tiles, device,
                                                              imgsz=min(tile_size, imgsz)))
        ball_cands = []
        for box, p in ball_boxes:
            foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
            xy = hom.to_pitch(foot)[0] if hom is not None else None
            on = hom.on_pitch(xy[None], 1.5)[0] if hom is not None else True
            if cone_colored(frame, box):
                cones += 1
                continue
            ball_cands.append((Detection(-1, "ball", box, p), on, xy))
        n += 1
        players.append(len(feet_xy))
        for d, _, _ in ball_cands:
            x1, y1, x2, y2 = d.box
            candidates.append((idx, (x1 + x2) / 2, (y1 + y2) / 2))
        pick = selector.select(idx, [(d, on) for d, on, _ in ball_cands])
        if pick is not None:
            x1, y1, x2, y2 = pick.box
            picks.append((idx, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)))
            is_near = False
            if hom is not None and feet_xy:
                bxy = next(xy for d, _, xy in ball_cands if d is pick)
                is_near = min(np.hypot(*(np.asarray(feet_xy) - bxy).T)) <= 2.0
            picks_near.append(is_near)
    moving = ~still_picks(picks, candidates, fps)
    in_play = int(moving.sum())
    near = int(np.asarray(picks_near, bool)[moving].sum()) if in_play else 0
    return dict(name=name + (f' +tile{tile_size}' if tile_size else ''), imgsz=imgsz, frames=n, players=np.mean(players), players_sd=np.std(players),
                in_play=in_play / max(n, 1), near=(near / in_play) if (hom is not None and in_play) else None,
                cones=cones / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280, 1600],
                    help="resolution per model (stock 1280, trained 1600); the last value repeats")
    ap.add_argument("--stride", type=int, default=2, help="check every Nth frame (2 = 15 checks per second at 30 fps)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--calib", help="count only players on the pitch and check the ball is near a player")
    ap.add_argument("--device", default="0")
    ap.add_argument("--tile", type=int, default=0, metavar="PX",
                    help="also search for the ball in PX-px crops of the pitch (e.g. 1280); compare with and without")
    args = ap.parse_args()

    hom = Homography.load(args.calib) if args.calib else None
    rows = []
    for k, name in enumerate(args.models):
        imgsz = args.imgsz[min(k, len(args.imgsz) - 1)]
        rows.append(evaluate_model(name, args.video, imgsz, args.stride, hom, args.device, args.max_seconds, tile_size=args.tile))
        print(f"done: {name}")

    lines = [f"{'model':42s} {'imgsz':>5s} {'frames':>6s} {'ball in play':>12s} {'near a player':>13s} "
             f"{'players/frame':>14s} {'cones as ball':>13s}"]
    for r in rows:
        near = f"{r['near']:.0%}" if r["near"] is not None else "-"
        lines.append(f"{r['name'][-42:]:42s} {r['imgsz']:5d} {r['frames']:6d} {r['in_play']:12.0%} {near:>13s} "
                     f"{r['players']:8.1f} +-{r['players_sd']:3.1f} {r['cones']:13.2f}")
    lines.append("ball in play / near a player: higher is better. players/frame: steadier is better. "
                 "cones as ball: lower is better.")
    print("\n" + "\n".join(lines))
    out = ROOT / "runs" / f"evaluate_{Path(args.video).stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
