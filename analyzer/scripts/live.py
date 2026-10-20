"""Live view from the camera's RTMP stream.

1. Run MediaMTX on the PC (see README) - it listens on rtmp://<PC-IP>:1935.
2. In the XbotGo app set the custom RTMP URL to rtmp://<PC-IP>:1935/live/falcon
3. python scripts/live.py rtmp://localhost:1935/live/falcon --calib runs/calib.json

Press q to stop; tracks and the report are saved to --out when you quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import ROOT, add_team_args, load_config, make_pipeline  # noqa: E402
from soccer_analyzer import viz  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.report import build_report  # noqa: E402
from soccer_analyzer.tracking import LiveStream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="rtmp://localhost:1935/live/falcon")
    ap.add_argument("--calib")
    ap.add_argument("--out", default=f"runs/live_{time.strftime('%Y%m%d_%H%M')}")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--record", action="store_true", help="also save the raw stream to out/raw.mp4")
    add_team_args(ap)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    hom = Homography.load(args.calib) if args.calib else None
    pipe = make_pipeline(cfg, hom, 30.0, args.bib)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = LiveStream(args.source, str(out / "raw.mp4") if args.record else None)
    t_last, fps = time.time(), 0.0

    # idx counts stream frames, so skipped frames leave gaps; t is seconds since start.
    for idx, t, frame in stream:
        dets = pipe.step(idx, t, frame)
        vis = viz.annotate_frame(frame, dets, pipe.team_map(), pipe.team_colors, pipe.jersey_colors)
        if hom is not None:
            pts = [(r["x"], r["y"], r["team"], r["cls"] == "ball") for r in pipe.rows if r["frame"] == idx]
            vis = viz.paste_minimap(vis, viz.minimap(hom.pitch, pts, pipe.team_colors))
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now
        cv2.putText(vis, f"{fps:4.1f} fps", (vis.shape[1] - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("soccer analyzer - live", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.close()
    cv2.destroyAllWindows()
    df = pipe.save(out)
    if not df.empty:
        print(build_report(df, out, hom.pitch if hom else None))


if __name__ == "__main__":
    main()
