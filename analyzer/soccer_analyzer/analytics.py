"""Match analytics computed from the tracks table.

Input is a DataFrame with one row per (frame, object):
    frame, t, track_id, cls, team, px, py, x, y
where (px, py) is the foot point in pixels and (x, y) the position on the
pitch in metres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPEED_MS = 12.0         # faster than this is a tracking glitch, not a sprint
POSSESSION_RADIUS_M = 2.0   # player this close to the ball "has" it


def smooth_positions(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling-median smoothing of pitch positions per track to remove jitter."""
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("x", "y"):
        df[col] = df.groupby("track_id")[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).median())
    return df


def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Distance covered, top speed and time on screen per tracked player."""
    p = smooth_positions(df[df.cls == "player"])
    p["dt"] = p.groupby("track_id")["t"].diff()
    p["d"] = np.hypot(p.groupby("track_id")["x"].diff(), p.groupby("track_id")["y"].diff())
    p["v"] = p["d"] / p["dt"]
    valid = p["v"].between(0, MAX_SPEED_MS)
    p.loc[~valid, ["d", "v"]] = np.nan
    g = p.groupby("track_id")
    stats = pd.DataFrame({
        "team": g["team"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else -1),
        "seconds_tracked": g["t"].agg(lambda s: s.max() - s.min()),
        "distance_m": g["d"].sum(),
        "top_speed_kmh": g["v"].quantile(0.95) * 3.6,
    })
    return stats.sort_values("distance_m", ascending=False)


def heatmap(df: pd.DataFrame, length: float = 105, width: float = 68, bins=None) -> np.ndarray:
    """2D occupancy histogram (shape bins[1] x bins[0]) of the given rows, ~1 m cells by default."""
    bins = bins or (max(int(round(length)), 10), max(int(round(width)), 6))
    h, _, _ = np.histogram2d(df["y"], df["x"], bins=[bins[1], bins[0]], range=[[0, width], [0, length]])
    return h


def ball_track(df: pd.DataFrame, max_gap: int = 15) -> pd.DataFrame:
    """Ball positions per frame with short gaps linearly interpolated."""
    b = df[df.cls == "ball"].sort_values("frame").drop_duplicates("frame")
    if b.empty:
        return b[["frame", "t", "x", "y"]]
    frames = pd.RangeIndex(b.frame.min(), b.frame.max() + 1)
    b = b.set_index("frame")[["t", "x", "y"]].reindex(frames)
    b = b.interpolate(limit=max_gap, limit_area="inside")
    return b.dropna().rename_axis("frame").reset_index()


def possession(df: pd.DataFrame, radius: float = POSSESSION_RADIUS_M) -> pd.DataFrame:
    """Per frame: the player closest to the ball if within `radius` metres."""
    ball = ball_track(df).set_index("frame")
    players = df[df.cls == "player"]
    rows = []
    for frame, grp in players[players.frame.isin(ball.index)].groupby("frame"):
        bx, by = ball.loc[frame, ["x", "y"]]
        d = np.hypot(grp.x - bx, grp.y - by)
        i = d.idxmin()
        if d[i] <= radius:
            rows.append((frame, ball.loc[frame, "t"], int(grp.at[i, "track_id"]), int(grp.at[i, "team"])))
    return pd.DataFrame(rows, columns=["frame", "t", "track_id", "team"])


def possession_share(poss: pd.DataFrame) -> pd.Series:
    if poss.empty:
        return pd.Series(dtype=float)
    return poss[poss.team >= 0].team.value_counts(normalize=True).sort_index()


def passes(poss: pd.DataFrame, min_hold: int = 3) -> pd.DataFrame:
    """Infer passes/turnovers from changes of the player in possession.

    A holder must keep the ball for `min_hold` frames to count, which filters
    out the ball rolling past a player.
    """
    if poss.empty:
        return pd.DataFrame(columns=["t_start", "t_end", "from_id", "to_id", "team_from", "team_to", "kind"])
    seg = (poss.track_id != poss.track_id.shift()).cumsum()
    spells = poss.groupby(seg).agg(track_id=("track_id", "first"), team=("team", "first"),
                                   t0=("t", "first"), t1=("t", "last"), n=("frame", "size"))
    spells = spells[spells.n >= min_hold].reset_index(drop=True)
    spells = spells[spells.track_id != spells.track_id.shift()].reset_index(drop=True)
    ev = []
    for a, b in zip(spells.itertuples(), spells.iloc[1:].itertuples()):
        kind = "pass" if a.team == b.team else "turnover"
        ev.append((a.t1, b.t0, a.track_id, b.track_id, a.team, b.team, kind))
    return pd.DataFrame(ev, columns=["t_start", "t_end", "from_id", "to_id", "team_from", "team_to", "kind"])


def team_shape(df: pd.DataFrame, team: int) -> pd.DataFrame:
    """Per-frame team centroid, width and depth: a quick read of how compact a team is."""
    p = df[(df.cls == "player") & (df.team == team)]
    g = p.groupby("frame")
    return pd.DataFrame({
        "t": g.t.first(),
        "cx": g.x.mean(), "cy": g.y.mean(),
        "depth_m": g.x.max() - g.x.min(),
        "width_m": g.y.max() - g.y.min(),
        "n": g.size(),
    }).reset_index()
