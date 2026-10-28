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
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.pipeline import MatchPipeline  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402
from soccer_analyzer.quality import format_quality, run_quality  # noqa: E402
from soccer_analyzer.report import build_report  # noqa: E402
from soccer_analyzer.teams import BIB_COLORS, BibClassifier, TeamClassifier  # noqa: E402
from soccer_analyzer.tracking import Tracker, read_frames  # noqa: E402
from soccer_analyzer import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path) -> dict:
    """config.yaml, with anything in config.local.yaml (same folder) laid over it.

    config.local.yaml is yours: it is not part of the download, so copying a new version of the code over the
    folder never replaces your weights, bib count and so on. Put only the lines you want to change in it.
    """
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    local = path.with_name("config.local.yaml")
    if local.exists():
        cfg = _merge(cfg or {}, yaml.safe_load(local.read_text()) or {})
        print(f"Config: {path.name} + {local.name}")
    return cfg or {}


def weights_path(name: str) -> str:
    """A trained model path relative to the analyzer folder, or a stock name Ultralytics downloads."""
    local = ROOT / name
    return str(local) if local.exists() else name


def make_tracker(cfg: dict) -> Tracker:
    m = cfg.get("model", {})
    return Tracker(weights=weights_path(m.get("weights", "yolo11m.pt")), imgsz=m.get("imgsz", 1280),
                   conf=m.get("conf", 0.25), ball_conf=m.get("ball_conf", 0.15), device=m.get("device"),
                   tile_size=m.get("tile_size", 0), tile_overlap=m.get("tile_overlap", 0.2),
                   player_nms_iou=m.get("player_nms_iou", 0.45),
                   tracker=str(ROOT / m.get("tracker", "trackers/bytetrack_soccer.yaml")))


def pitch_region(hom) -> np.ndarray:
    """Pixel corners of the pitch (plus 2 m around it): the area worth searching for the ball."""
    L, W, m = hom.pitch.length, hom.pitch.width, 2.0
    return hom.to_image(np.array([[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]], float))


def make_pipeline(cfg: dict, hom, video_fps: float, bib: str | None = None) -> MatchPipeline:
    """bib: bib colour name to split teams by bib (overrides config.yaml)."""
    t = cfg.get("teams", {})
    bib = bib or (t.get("bib_color") if t.get("mode") == "bib" else None)
    teams = BibClassifier(bib) if bib else TeamClassifier()
    print(f"Teams: {'bib colour ' + bib + ' = team 1' if bib else 'automatic (2 shirt colours)'}")
    tracker = make_tracker(cfg)
    if hom is not None and tracker.tile_size:
        tracker.set_pitch_region(pitch_region(hom))
        print(f"Ball tiles: {tracker.tile_size}px, only over the pitch")
    return MatchPipeline(tracker, hom, teams=teams,
                         pitch_margin=cfg.get("analysis", {}).get("pitch_margin", 1.5), fps=video_fps,
                         reject_cone_colours=cfg.get("ball", {}).get("reject_cone_colours", True),
                         bib_count=(t.get("bib_count") if bib else None),
                         drop_standing_on_line=cfg.get("analysis", {}).get("drop_standing_on_line", True),
                         ignore_zones=cfg.get("analysis", {}).get("ignore_zones") or [],
                         drop_static_s=cfg.get("analysis", {}).get("drop_static_s", 30),
                         idle_edge_m_per_min=cfg.get("analysis", {}).get("drop_idle_at_edge", 20),
                         players=cfg.get("session", {}).get("players"))


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
    ap.add_argument("--tracker", help="tracker yaml for this run (overrides model.tracker in config.yaml), "
                                      "e.g. trackers/botsort_reid.yaml")
    ap.add_argument("--players", type=int, help="players on the pitch (8 for a 4 v 4; overrides session.players): "
                                                "scores the run, and with --bib joins broken tracks per team")
    add_team_args(ap)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.tracker:
        cfg = _merge(cfg, {"model": {"tracker": args.tracker}})
    if args.players:
        cfg = _merge(cfg, {"session": {"players": args.players}})
    print(f"Tracker: {cfg.get('model', {}).get('tracker', 'trackers/bytetrack_soccer.yaml')}")
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
    checked = False

    tr = pipe.tracker
    print(f"Model: {tr.model.ckpt_path if hasattr(tr.model, 'ckpt_path') else tr.model} "
          f"(classes: {', '.join(tr.model.names.values())}) at {tr.imgsz} px")
    for idx, t, frame in read_frames(args.source, args.stride):
        if args.max_seconds and t > args.max_seconds:
            break
        pipe.step(idx, t, frame)
        if t > 10 and not checked:
            checked = True
            if not any(r["cls"] == "player" for r in pipe.rows):
                raise SystemExit(
                    "\nNo players found in the first 10 seconds, so stopping. Likely causes:\n"
                    "  1. the wrong model: check `weights:` in config.yaml (the good one is runs/train/soccer-v1b/weights/best.pt)\n"
                    "  2. the wrong calibration for this camera position (--calib)\n"
                    "  3. a wrong video path\n")
        if idx % (100 * args.stride) == 0:
            print(f"  t={t:7.1f}s  frame={idx}  {idx / args.stride / max(time.time() - t0, 1e-6):.1f} fps")

    df = pipe.save(out)
    for reason, label in (("off_pitch", "outside the lines (off the pitch)"),
                          ("standing_on_line", "standing still on a touchline (taken for a coach)"),
                          ("ignore_zone", "inside an ignore_zones area"),
                          ("static", "that barely moved (coach, post, cone)"),
                          ("seated", "that were sitting or lying"),
                          ("idle_at_edge", "that hardly moved near a line (coach)")):
        ids = pipe.dropped.get(reason, set())
        if ids:
            print(f"Dropped {len(ids)} track(s) {label}. Wrong? Check the calibration"
                  + (" or set analysis.drop_standing_on_line: false in config.yaml" if reason == "standing_on_line" else "") + ".")
    print(f"Tracked {len(df)} detections in {time.time() - t0:.0f}s -> {out / 'tracks.csv'}")
    if not args.no_video:
        # Drawn after the whole video is analyzed, so IDs and teams are the final, joined ones.
        print("Writing annotated.mp4 ...")
        viz.render_video(args.source, df, out / "annotated.mp4", fps, args.stride,
                         hom.pitch if hom else None, pipe.jersey_colors, args.max_seconds, hom, hom is not None)
    summary = build_report(df, out, hom.pitch if hom else None)
    print(yaml.safe_dump(summary, sort_keys=False))
    bibs = cfg.get("teams", {}).get("bib_count") if (args.bib or cfg.get("teams", {}).get("mode") == "bib") else None
    text = format_quality(run_quality(df, cfg.get("session", {}).get("players"), bibs))
    (out / "quality.txt").write_text(text + "\n")
    print("Quality checks (set session.players in config.local.yaml to score the player count):\n" + text)


if __name__ == "__main__":
    main()
