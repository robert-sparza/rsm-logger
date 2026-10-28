"""Quality checks on a finished run, from tracks.csv alone (no hand-made ground truth needed).

They use what we know about the session: how many players were on the pitch and how many wore bibs.
A perfect run shows the expected player count in every frame, about one track per player, no two boxes on
one player, and a ball that never jumps across the pitch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def box_overlap_frames(players: pd.DataFrame, iou: float = 0.45) -> float:
    """Share of frames in which two boxes overlap more than `iou` (one player boxed twice, or players on top of each other)."""
    hit, frames = 0, 0
    for _, g in players.groupby("frame"):
        frames += 1
        r = g[["bx1", "by1", "bx2", "by2"]].to_numpy()
        found = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                iw = min(r[i, 2], r[j, 2]) - max(r[i, 0], r[j, 0])
                ih = min(r[i, 3], r[j, 3]) - max(r[i, 1], r[j, 1])
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    union = (r[i, 2] - r[i, 0]) * (r[i, 3] - r[i, 1]) + (r[j, 2] - r[j, 0]) * (r[j, 3] - r[j, 1]) - inter
                    found |= inter / max(union, 1e-9) > iou
        hit += found
    return hit / max(frames, 1)


def run_quality(df: pd.DataFrame, expect_players: int | None = None, bib_count: int | None = None,
                min_track_s: float = 3.0) -> list[tuple[str, str, str]]:
    """Returns rows (name, value, verdict) where verdict is "ok", "check" or "" (information only)."""
    out: list[tuple[str, str, str]] = []
    p = df[df.cls == "player"]
    frames = df.frame.nunique()
    if p.empty:
        return [("players", "none found", "check")]

    n = p.groupby("frame").size().reindex(sorted(df.frame.unique()), fill_value=0)
    out.append(("players per frame", f"{n.mean():.1f} +- {n.std():.1f}", ""))
    if expect_players:
        exact = float((n == expect_players).mean())
        near = float((n.sub(expect_players).abs() <= 1).mean())
        out.append((f"frames with {expect_players} players (+-1)", f"{exact:.0%} ({near:.0%})",
                    "ok" if near >= 0.85 else "check"))

    dur = p.groupby("track_id").t.agg(lambda s: s.max() - s.min())
    long_tracks = int((dur >= min_track_s).sum())
    out.append((f"tracks over {min_track_s:.0f} s", str(long_tracks), ""))
    if expect_players:
        ratio = long_tracks / expect_players
        out.append(("tracks per expected player", f"{ratio:.1f} (1.0 = no broken IDs)", "ok" if ratio <= 1.5 else "check"))

    if bib_count:
        t1 = p[p.team == 1].groupby("frame").size().reindex(n.index, fill_value=0)
        share = float((t1 == bib_count).mean())
        out.append((f"frames with exactly {bib_count} bib players", f"{share:.0%}", "ok" if share >= 0.85 else "check"))
    unknown = float((p.team < 0).mean())
    out.append(("player rows with no team", f"{unknown:.1%}", "ok" if unknown < 0.02 else "check"))

    dup = box_overlap_frames(p)
    out.append(("frames with two boxes overlapping", f"{dup:.1%}", "ok" if dup < 0.15 else "check"))

    ball = df[df.cls == "ball"].sort_values("frame")
    out.append(("frames with a ball", f"{ball.frame.nunique() / max(frames, 1):.0%}", ""))
    if len(ball) > 2:
        step = np.hypot(ball.x.diff(), ball.y.diff())
        gap = ball.frame.diff()
        fps = frames / max(df.t.max(), 1e-9)
        jumps = int(((step > 4) & (gap <= 3)).sum())
        per_min = jumps / max(df.t.max() / 60, 1e-9)
        out.append(("ball jumps over 4 m per minute", f"{per_min:.1f}", "ok" if per_min < 5 else "check"))
    return out


def format_quality(rows: list[tuple[str, str, str]]) -> str:
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"  {name:<{w}}  {val}" + (f"   [{verdict}]" if verdict else "") for name, val, verdict in rows)
