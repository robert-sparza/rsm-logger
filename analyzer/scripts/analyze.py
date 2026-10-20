"""Analyze a recorded match.

Usage:
    python scripts/analyze.py data/videos/match.mp4 --calib runs/calib.json --out runs/match1

Writes to --out:
    tracks.csv            every detection per frame (pixels + pitch metres + team)
    annotated.mp4         video with team-coloured players, ball marker and minimap
    player_stats.csv      distance, top speed per track
    heatmap_team*.png     where each team spent time
    ball_path.png         ball trajectory
    passes.csv / passes_team*.png   inferred passes and turnovers
    summary.json          possession share, pass counts, team width/depth
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.pipeline import MatchPipeline  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.report import build_report  # noqa: E402
from soccer_analyzer.teams import BIB_COLORS, BibClassifier, TeamClassifier  # noqa: E402
from soccer_analyzer.tracking import Tracker, read_frames  # noqa: E402
from soccer_analyzer import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def make_tracker(cfg: dict) -> Tracker:
    m = cfg.get("model", {})
    return Tracker(weights=m.get("weights", "yolo11m.pt"), imgsz=m.get("imgsz", 1280),
                   conf=m.get("conf", 0.25), player_ids=m.get("player_class_ids", [0]),
                   ball_ids=m.get("ball_class_ids", [32]), device=m.get("device"),
                   tracker=str(ROOT / m.get("tracker", "trackers/bytetrack_soccer.yaml")))


def make_pipeline(cfg: dict, hom, video_fps: float, bib: str | None = None) -> MatchPipeline:
    """bib: bib colour name to split teams by bib (overrides config.yaml)."""
    t = cfg.get("teams", {})
    bib = bib or (t.get("bib_color") if t.get("mode") == "bib" else None)
    teams = BibClassifier(bib) if bib else TeamClassifier()
    print(f"Teams: {'bib colour ' + bib + ' = team 1' if bib else 'automatic (2 shirt colours)'}")
    return MatchPipeline(make_tracker(cfg), hom, teams=teams,
                         pitch_margin=cfg.get("analysis", {}).get("pitch_margin", 1.5), fps=video_fps,
                         reject_cone_colours=cfg.get("ball", {}).get("reject_cone_colours", True))


def add_team_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--bib", choices=sorted(BIB_COLORS),
                    help="training with bibs: players wearing this colour are team 1, everyone else team 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--calib", help="homography json from calibrate.py")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (2 = twice as fast)")
    ap.add_argument("--max-seconds", type=float, help="stop after this much video (for quick tests)")
    ap.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    add_team_args(ap)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    hom = Homography.load(args.calib) if args.calib else None
    if hom is None:
        print("No --calib given: tracking only, no pitch maps. Run scripts/calibrate.py first for full analysis.")
    cap = cv2.VideoCapture(args.source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = video_fps / args.stride
    cap.release()
    pipe = make_pipeline(cfg, hom, video_fps, args.bib)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for idx, t, frame in read_frames(args.source, args.stride):
        if args.max_seconds and t > args.max_seconds:
            break
        pipe.step(idx, t, frame)
        if idx % (100 * args.stride) == 0:
            print(f"  t={t:7.1f}s  frame={idx}  {idx / args.stride / max(time.time() - t0, 1e-6):.1f} fps")

    df = pipe.save(out)
    print(f"Tracked {len(df)} detections in {time.time() - t0:.0f}s -> {out / 'tracks.csv'}")
    if not args.no_video:
        # Drawn after the whole video is analyzed, so IDs and teams are the final, joined ones.
        print("Writing annotated.mp4 ...")
        viz.render_video(args.source, df, out / "annotated.mp4", fps, args.stride,
                         hom.pitch if hom else None, pipe.jersey_colors, args.max_seconds)
    summary = build_report(df, out, hom.pitch if hom else None)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
