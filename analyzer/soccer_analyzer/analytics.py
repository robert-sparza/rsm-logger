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
MEDIAN_S = 0.5             # jitter filter: median of each position over this many seconds
STEP_S = 0.33              # distance is measured between positions this far apart in time
MIN_ACTIVE_S = 10.0         # distance per minute is only reported for players followed at least this long
POSSESSION_RADIUS_M = 2.0   # player this close to the ball "has" it


def smooth_positions(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling-median smoothing of pitch positions per track to remove jitter."""
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("x", "y"):
        df[col] = df.groupby("track_id")[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).median())
    return df


def _frame_rate(df: pd.DataFrame) -> float:
    dt = df.drop_duplicates("frame").sort_values("frame").t.diff().dropna()
    dt = dt[dt > 0]
    return float(1 / dt.median()) if len(dt) else 30.0


def smooth_track(q: pd.DataFrame, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One track's positions with detection jitter taken out, sampled every STEP_S seconds.

    Foot points of a box jitter by a few tens of centimetres from frame to frame; summed up, that
    makes a player standing still "run" 30-50 m a minute. So: a median over MEDIAN_S seconds, then
    positions every STEP_S seconds, and distance is measured between those.
    """
    win = max(3, int(round(MEDIAN_S * fps)) | 1)
    x = q.x.rolling(win, center=True, min_periods=1).median().to_numpy()
    y = q.y.rolling(win, center=True, min_periods=1).median().to_numpy()
    t = q.t.to_numpy()
    keep = np.r_[True, np.diff(np.floor(t / STEP_S)) > 0]          # first sample of each STEP_S slot
    return t[keep], x[keep], y[keep]


def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Distance covered, speed and time followed per tracked player (jitter-filtered, see smooth_track)."""
    p = df[df.cls == "player"].sort_values(["track_id", "frame"])
    fps = _frame_rate(p) if len(p) else 30.0
    rows = {}
    for tid, q in p.groupby("track_id"):
        t, x, y = smooth_track(q, fps)
        dt = np.diff(t)
        d = np.hypot(np.diff(x), np.diff(y))
        ok = (dt > 0) & (dt <= 3 * STEP_S) & (d / np.where(dt > 0, dt, 1) <= MAX_SPEED_MS)   # gaps / glitches don't count
        v = d[ok] / dt[ok]
        rows[tid] = dict(
            team=q.team.mode().iat[0] if len(q.team.mode()) else -1,
            seconds_tracked=float(q.t.max() - q.t.min()),
            active_s=float(dt[ok].sum()),
            distance_m=float(d[ok].sum()),
            top_speed_kmh=float(np.percentile(v, 95) * 3.6) if len(v) else np.nan)
    stats = pd.DataFrame.from_dict(rows, orient="index").rename_axis("track_id")
    if stats.empty:
        return stats.assign(m_per_min=pd.Series(dtype=float))
    # Distance per minute counts only the time the player was actually followed (gaps where the track
    # was lost don't dilute it), and is blank for players followed under MIN_ACTIVE_S seconds.
    stats["m_per_min"] = (stats.distance_m / stats.active_s * 60).where(stats.active_s >= MIN_ACTIVE_S)
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
