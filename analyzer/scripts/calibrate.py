"""Click pitch landmarks on a frame to compute the camera -> pitch homography.

Usage:
    python scripts/calibrate.py data/videos/match.mp4 --out runs/calib.json [--at 30]

The first 7 landmarks (corners, halfway line ends, centre spot) depend only on
--length/--width. The others also need the real size of the markings on small
pitches (--circle-radius, --box-depth, --box-width, --penalty-spot, --goal-area-*); if you don't
know them, skip those landmarks with "s".

For each landmark name shown at the top of the window:
    left-click  = place the point
    s           = skip (landmark not visible)
    u           = undo last point
    q / Enter   = finish (needs >= 4 points, 6+ spread across the pitch is better)

Re-run this whenever the tripod or zoom changes.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.pitch import Homography, Pitch  # noqa: E402
from soccer_analyzer.viz import draw_pitch, to_canvas  # noqa: E402


def grab_frame(src: str, at_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {src}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video file or rtmp:// URL")
    ap.add_argument("--out", default="runs/calib.json")
    ap.add_argument("--at", type=float, default=5.0, help="seconds into the video to grab the frame")
    ap.add_argument("--length", type=float, default=105.0, help="pitch length in metres")
    ap.add_argument("--width", type=float, default=68.0, help="pitch width in metres")
    ap.add_argument("--circle-radius", type=float, default=9.15, help="centre circle radius (m)")
    ap.add_argument("--box-depth", type=float, default=16.5, help="penalty area depth from goal line (m)")
    ap.add_argument("--box-width", type=float, default=40.32, help="penalty area width (m)")
    ap.add_argument("--penalty-spot", type=float, default=11.0, help="penalty spot distance from goal line (m)")
    ap.add_argument("--goal-area-depth", type=float, default=5.5, help="goal area (6-yard box) depth (m)")
    ap.add_argument("--goal-area-width", type=float, default=18.32, help="goal area (6-yard box) width (m)")
    args = ap.parse_args()

    pitch = Pitch(args.length, args.width, circle_radius=args.circle_radius, box_depth=args.box_depth,
                  box_width=args.box_width, penalty_spot=args.penalty_spot,
                  goal_area_depth=args.goal_area_depth, goal_area_width=args.goal_area_width)
    frame = grab_frame(args.source, args.at)
    names = list(pitch.landmarks().items())
    img_pts, pitch_pts, i = [], [], 0
    click = {"pt": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    ref = draw_pitch(pitch, scale=4)

    while i < len(names):
        name, xy = names[i]
        view = frame.copy()
        for p in img_pts:
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
        cv2.putText(view, f"[{len(img_pts)} pts] click: {name}   (s=skip u=undo q=done)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        r = ref.copy()
        cv2.circle(r, to_canvas(xy, 4), 7, (0, 0, 255), -1)
        cv2.imshow(win, view)
        cv2.imshow("pitch reference", r)
        k = cv2.waitKey(30) & 0xFF
        if click["pt"] is not None:
            img_pts.append(click["pt"])
            pitch_pts.append(xy)
            click["pt"] = None
            i += 1
        elif k == ord("s"):
            i += 1
        elif k == ord("u") and img_pts:
            img_pts.pop()
            pitch_pts.pop()
            i = max(0, i - 1)
        elif k in (ord("q"), 13):
            break
    cv2.destroyAllWindows()

    hom = Homography.fit(img_pts, pitch_pts, pitch)
    err = np.linalg.norm(hom.to_pitch(np.array(img_pts)) - np.array(pitch_pts), axis=1)
    print(f"{len(img_pts)} points, reprojection error: mean {err.mean():.2f} m, max {err.max():.2f} m")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hom.save(args.out)

    # Visual check: project the pitch lines back onto the frame.
    check = frame.copy()
    L, W = pitch.length, pitch.width
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))]:
        line = hom.to_image(np.linspace(a, b, 50))
        cv2.polylines(check, [line.astype(np.int32)], False, (0, 0, 255), 2)
    cv2.imwrite(str(Path(args.out).with_suffix(".check.jpg")), check)
    print(f"Saved {args.out} and an overlay check image next to it")


if __name__ == "__main__":
    main()
