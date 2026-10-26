"""Turn a tracks.csv into stats tables and plots."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import analytics as A
from . import viz
from .pitch import Pitch


def build_report(df: pd.DataFrame, out_dir: Path, pitch: Pitch | None = None) -> dict:
    pitch = pitch or Pitch()
    out_dir.mkdir(parents=True, exist_ok=True)
    on_pitch = df.dropna(subset=["x", "y"])
    players = df[df.cls == "player"]
    seen = players.groupby("track_id").t.agg(lambda s: s.max() - s.min())
    summary: dict = {
        "frames": int(df.frame.nunique()),
        "players_tracked": int((seen >= 3).sum()),            # seen for at least 3 s
        "short_fragments": int((seen < 3).sum()),             # bits the tracker couldn't join
        "players_per_team": {f"team{k}": int(v) for k, v in
                             players[players.track_id.isin(seen.index[seen >= 3])]
                             .groupby("track_id").team.first().value_counts().sort_index().items()},
    }

    if on_pitch.empty:
        summary["note"] = "No pitch coordinates - run calibrate.py first to get heatmaps and distances."
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    stats = A.player_stats(on_pitch)
    stats.to_csv(out_dir / "player_stats.csv")

    for team in sorted(t for t in on_pitch[on_pitch.cls == "player"].team.unique() if t >= 0):
        rows = on_pitch[(on_pitch.cls == "player") & (on_pitch.team == team)]
        viz.plot_heatmap(A.heatmap(rows, pitch.length, pitch.width), pitch, f"Team {team} heatmap",
                         out_dir / f"heatmap_team{team}.png")
        shape = A.team_shape(on_pitch, team)
        shape.to_csv(out_dir / f"shape_team{team}.csv", index=False)
        summary[f"team{team}_avg_width_m"] = round(float(shape.width_m.mean()), 1)
        summary[f"team{team}_avg_depth_m"] = round(float(shape.depth_m.mean()), 1)
        pace = stats[(stats.team == team)].m_per_min.dropna()
        if len(pace):
            summary[f"team{team}_avg_m_per_min"] = round(float(pace.mean()), 1)

    ball = A.ball_track(on_pitch)
    ball.to_csv(out_dir / "ball.csv", index=False)
    viz.plot_ball_path(ball, pitch, out_dir / "ball_path.png")
    summary["ball_frames"] = int(len(ball))

    poss = A.possession(on_pitch)
    passes = A.passes(poss)
    passes.to_csv(out_dir / "passes.csv", index=False)
    summary["possession_share"] = {f"team{k}": round(float(v), 3) for k, v in A.possession_share(poss).items()}
    summary["passes"] = passes[passes.kind == "pass"].groupby("team_from").size().rename(lambda k: f"team{k}").to_dict()
    summary["turnovers"] = int((passes.kind == "turnover").sum())
    for team in (0, 1):
        viz.plot_passes(passes, on_pitch, pitch, team, out_dir / f"passes_team{team}.png")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    return summary
