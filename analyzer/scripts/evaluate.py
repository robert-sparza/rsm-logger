"""Compare detectors on a video: how often is the ball found, how many players, cones mistaken for balls.

Usage:
    python scripts/evaluate.py data/videos/training.mp4 --models yolo11m.pt runs/train/soccer-v1/weights/best.pt \
        --calib runs/calib_small.json

Checks a frame every --every seconds with each model, at that model's best
resolution (--imgsz, one value per model or one for all).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import weights_path  # noqa: E402
from soccer_analyzer.ball import cone_colored  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.tracking import BALL_NAMES, PLAYER_NAMES, class_ids, read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280, 1920],
                    help="resolution per model (stock 1280, trained 1920); the last value repeats")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--calib", help="count only players on the pitch")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    hom = Homography.load(args.calib) if args.calib else None
    rows = []
    for k, name in enumerate(args.models):
        imgsz = args.imgsz[min(k, len(args.imgsz) - 1)]
        model = YOLO(weights_path(name))
        pids, bids = class_ids(model.names, PLAYER_NAMES), class_ids(model.names, BALL_NAMES)
        n_frames, players, ball_found, cones_as_ball = 0, [], 0, 0
        next_t = 0.0
        for idx, t, frame in read_frames(args.video):
            if t < next_t:
                continue
            next_t = t + args.every
            res = model.predict(frame, imgsz=imgsz, conf=0.25, device=args.device, verbose=False)[0]
            n_frames += 1
            n_players, real_ball = 0, False
            for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
                if c in pids:
                    if hom is not None:
                        foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
                        if not hom.on_pitch(hom.to_pitch(foot), 1.5)[0]:
                            continue
                    n_players += 1
                elif c in bids:
                    if cone_colored(frame, box):
                        cones_as_ball += 1
                    else:
                        real_ball = True
            players.append(n_players)
            ball_found += real_ball
        rows.append((name, imgsz, n_frames, np.mean(players), np.std(players), ball_found / max(n_frames, 1),
                     cones_as_ball / max(n_frames, 1)))
        print(f"done: {name}")

    print(f"\n{'model':45s} {'imgsz':>5s} {'frames':>6s} {'players/frame':>14s} {'ball found':>10s} {'cones called ball/frame':>24s}")
    for name, imgsz, n, mp, sp, bf, cb in rows:
        print(f"{name[-45:]:45s} {imgsz:5d} {n:6d} {mp:8.1f} +-{sp:4.1f} {bf:9.0%} {cb:24.2f}")
    print("\nplayers/frame: steadier (small +-) is better. ball found: higher is better. cones called ball: lower is better.")


if __name__ == "__main__":
    main()
