"""Team polygons and space control from player positions (pitch metres).

Two views of "how much space does a team use / own", both needing only where each player is and which team
he is on. Track ids don't matter, so they are robust to the ID swaps that happen when players crowd together.

hull          smallest convex polygon around a team's players (its "effective play space"). Its area divided
              by the number of players says how compact the team is; a 4 v 4 in a tight drill is small.
control       Voronoi space control: every point of the pitch belongs to the nearest player, so a team owns
              the cells of its players. The share of the pitch each team owns says who has the space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hull(points: np.ndarray) -> np.ndarray | None:
    """Convex hull vertices in order (n x 2), or None with fewer than 3 points or all on a line."""
    pts = np.unique(np.round(np.asarray(points, float), 3), axis=0)
    if len(pts) < 3:
        return None
    from scipy.spatial import ConvexHull, QhullError
    try:
        h = ConvexHull(pts)
    except QhullError:
        return None
    return pts[h.vertices]


def polygon_area(poly: np.ndarray | None) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def control_shares(points: np.ndarray, teams: np.ndarray, length: float, width: float, cell: float = 1.0) -> dict[int, float]:
    """Share of the pitch (0-1) whose nearest player is on each team. Grid of `cell` metre squares."""
    points, teams = np.asarray(points, float), np.asarray(teams)
    if len(points) == 0:
        return {}
    gx = np.arange(cell / 2, length, cell)
    gy = np.arange(cell / 2, width, cell)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    nearest = np.argmin(((grid[:, None, :] - points[None, :, :]) ** 2).sum(-1), axis=1)
    owner = teams[nearest]
    return {int(t): float((owner == t).mean()) for t in np.unique(teams)}


def space_table(df: pd.DataFrame, length: float, width: float, cell: float = 1.0, min_players: int = 3) -> pd.DataFrame:
    """Per frame: players, hull area and Voronoi share of each team (0 and 1).

    Columns: frame, t, n0, n1, hull0_m2, hull1_m2, control0, control1. Hull area is NaN with under min_players.
    """
    p = df[(df.cls == "player") & (df.team >= 0)].dropna(subset=["x", "y"])
    p = p[p.x.between(-1, length + 1) & p.y.between(-1, width + 1)]
    rows = []
    for frame, g in p.groupby("frame"):
        xy = g[["x", "y"]].to_numpy()
        team = g.team.to_numpy()
        both = len(set(team)) == 2
        shares = control_shares(xy, team, length, width, cell) if both else {}
        row = dict(frame=int(frame), t=float(g.t.iloc[0]))
        for k in (0, 1):
            pts = xy[team == k]
            row[f"n{k}"] = len(pts)
            row[f"hull{k}_m2"] = polygon_area(hull(pts)) if len(pts) >= min_players else np.nan
            row[f"control{k}"] = shares.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def frame_polygons(g: pd.DataFrame, min_players: int = 3) -> dict[int, np.ndarray]:
    """{team: hull vertices} for one frame's player rows (x, y, team), for drawing."""
    out = {}
    for k, gt in g[g.team >= 0].groupby("team"):
        if len(gt) >= min_players:
            h = hull(gt[["x", "y"]].to_numpy())
            if h is not None:
                out[int(k)] = h
    return out
