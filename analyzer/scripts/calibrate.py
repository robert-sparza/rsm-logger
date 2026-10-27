"""Click pitch landmarks on a frame to compute the camera -> pitch homography.

Usage:
    python scripts/calibrate.py data/videos/match.mp4 --out runs/calib.json [--at 30]

"far" = the touchline AWAY from the camera, "near" = the touchline closest to
the camera; left/right as you see them in the video. The red dot on the small
pitch diagram shows which point to click.

The first 7 landmarks (corners, halfway line ends, centre spot) depend only on
--length/--width. The others also need the real size of the markings on small
pitches (--circle-radius, --box-depth, --box-width, --penalty-spot, --goal-area-*); if you don't
know them, skip those landmarks with "s".

For each landmark name shown at the top of the window:
    left-click  = place the point for the highlighted landmark
    s / Space   = skip it (not visible / not sure); upper or lower case both work
    a, b, c ... = the letter next to a landmark on the small pitch diagram chooses it directly
                  (re-click to move a point)
    u / Backspace = undo
    q / Enter / Esc = finish (needs >= 4 points, 6+ spread across the pitch is better)
The diagram and the instructions are in the top-left corner of the frame; clicks there are ignored.
    --list      = print the landmark names and where they are on the pitch

Re-run this whenever the tripod or zoom changes.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.calib_ui import KEYS, CalibState  # noqa: E402
from soccer_analyzer.pitch import Homography, Pitch  # noqa: E402
from soccer_analyzer.viz import draw_pitch, to_canvas  # noqa: E402


MARGIN = 18                                               # room around the small pitch diagram for the letters


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
    ap.add_argument("--inset", type=float, default=0.26, help="width of the pitch diagram in the corner, as a share of the video width")
    ap.add_argument("--list", action="store_true", help="print the landmark names and positions, then stop")
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
    landmarks = pitch.landmarks()
    if args.list:
        for key, (name, xy) in zip(KEYS, landmarks.items()):
            print(f"{key}  {name:24s} at ({xy[0]:.1f} m, {xy[1]:.1f} m)")
        return
    frame = grab_frame(args.source, args.at)
    state = CalibState(list(landmarks))
    click = {"pt": None}
    H, Wf = frame.shape[:2]
    scale = max(0.6, Wf / 1920)                           # text size follows the video resolution
    inset_w = int(args.inset * Wf)
    ref_scale = (inset_w - 36) / pitch.length
    ref0 = draw_pitch(pitch, scale=ref_scale, margin=MARGIN)
    ih, iw = ref0.shape[:2]
    text_h = int(64 * scale)
    colors = {"done": (0, 255, 0), "skip": (140, 140, 140), "now": (0, 0, 255), "todo": (255, 255, 255)}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and not (x < iw + 8 and y < ih + text_h + 8):   # ignore clicks on the inset
            click["pt"] = (x, y)

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(Wf, 1600), int(min(Wf, 1600) * H / Wf))
    cv2.setMouseCallback(win, on_mouse)

    def reference() -> np.ndarray:
        """The pitch diagram with every landmark's letter: where each one is, and which are done / skipped / next."""
        r = ref0.copy()
        for name, xy in landmarks.items():
            st = ("now" if name == state.current else "done" if name in state.points
                  else "skip" if name in state.skipped else "todo")
            c = to_canvas(xy, ref_scale, MARGIN)
            cv2.circle(r, c, 7 if st == "now" else 4, colors[st], -1)
            if st == "now":
                cv2.circle(r, c, 11, (0, 255, 255), 2)
            cv2.putText(r, state.key_of(name), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale if st != "now" else 0.7 * scale, (0, 255, 255) if st == "now" else colors[st], 1)
        cv2.putText(r, "far side", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), 1)
        cv2.putText(r, "near side (camera)", (12, ih - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), 1)
        return r

    while True:
        view = frame.copy()
        for name, p in state.points.items():
            cv2.circle(view, (int(p[0]), int(p[1])), int(6 * scale), (0, 255, 255), -1)
            cv2.putText(view, state.key_of(name), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * scale, (0, 255, 255), 2)
        # The diagram and the instructions sit in the top-left corner (usually sky and trees, not pitch).
        view[4:4 + ih, 4:4 + iw] = reference()
        y0 = 4 + ih
        cv2.rectangle(view, (4, y0), (4 + iw, y0 + text_h), (0, 0, 0), -1)
        head = (f"click: {state.current}  [{state.key_of(state.current)}]" if state.current
                else "all done - press q")
        cv2.putText(view, head, (10, y0 + int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 255), 2)
        cv2.putText(view, "S/space=skip  U=undo  Q/Enter=finish  or press a point's letter", (10, y0 + int(52 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        ch = chr(k).lower() if 0 < k < 128 else ""
        if click["pt"] is not None:
            state.click(click["pt"])
            click["pt"] = None
        elif ch == "s" or k in (32, 9):                   # s / S, space or Tab: skip this point
            state.skip()
        elif ch == "u" or k == 8:                         # u / U or Backspace: undo
            state.undo()
        elif ch == "q" or k in (13, 27):                  # q / Q, Enter or Esc: finish
            break
        elif ch and ch in KEYS:
            state.jump(ch)
    cv2.destroyAllWindows()

    clicked_names = state.placed
    if len(clicked_names) < 4:
        raise SystemExit(f"Only {len(clicked_names)} points placed; need at least 4 (6 or more spread across the pitch is better). "
                         "Run it again.")
    img_pts = [state.points[n] for n in clicked_names]
    pitch_pts = [landmarks[n] for n in clicked_names]
    hom = Homography.fit(img_pts, pitch_pts, pitch)
    err = np.linalg.norm(hom.to_pitch(np.array(img_pts)) - np.array(pitch_pts), axis=1)
    print(f"{len(img_pts)} points, error: mean {err.mean():.2f} m, max {err.max():.2f} m")
    for n, e in zip(clicked_names, err):
        flag = "   <-- check this point / the pitch size" if e > 0.75 else ""
        print(f"   {n:24s} {e:5.2f} m{flag}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hom.save(args.out)

    # Visual check: draw every painted line back onto the frame, plus the clicks.
    check = frame.copy()
    for line in pitch.marking_polylines():
        cv2.polylines(check, [hom.to_image(line).astype(np.int32)], False, (0, 0, 255), 2)
    for p, q in zip(img_pts, hom.to_image(np.array(pitch_pts))):
        cv2.circle(check, (int(p[0]), int(p[1])), 7, (0, 255, 255), 2)      # where you clicked
        cv2.circle(check, (int(q[0]), int(q[1])), 3, (0, 0, 255), -1)      # where the model puts it
    cv2.imwrite(str(Path(args.out).with_suffix(".check.jpg")), check)
    print(f"Saved {args.out} and {Path(args.out).with_suffix('.check.jpg')}:")
    print("   red lines should sit on the real lines; yellow rings = your clicks.")


if __name__ == "__main__":
    main()
