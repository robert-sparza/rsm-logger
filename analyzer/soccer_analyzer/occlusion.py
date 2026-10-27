"""Fix pitch positions of players whose legs are hidden.

The pitch position comes from the bottom of the player's box (the feet). When
the legs are hidden (a tree or the coach in front, the bottom edge of the
frame), the box stops at the hidden part and the player seems to jump metres
away from the camera. The top of the box (the head) is still right, so we learn
how tall a fully visible player looks at each height in the image and, for
boxes that are clearly too short, put the feet where they should be.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_height_model(players: pd.DataFrame, bins: int = 20, quantile: float = 0.8,
                          min_rows: int = 20) -> np.ndarray | None:
    """Fit box height ~ a + b * box_top on full-looking boxes; returns (a, b) or None.

    The 80th percentile height per band of box_top approximates a fully visible,
    upright player at that distance from the camera.
    """
    top = players.by1.to_numpy()
    h = (players.by2 - players.by1).to_numpy()
    if len(top) < min_rows * 3:
        return None
    bins = min(bins, len(top) // min_rows)          # short clips: fewer, fuller bands
    edges = np.quantile(top, np.linspace(0, 1, bins + 1))
    xs, ys, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top >= lo) & (top <= hi)
        if m.sum() >= min_rows:
            xs.append(np.median(top[m]))
            ys.append(np.quantile(h[m], quantile))
            ws.append(m.sum())
    if len(xs) < 3:
        return None
    b, a = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    return np.array([a, b])


def player_scale(players: pd.DataFrame, expected: pd.Series, min_rows: int = 30, quantile: float = 0.8,
                 low: float = 0.7, high: float = 1.15) -> pd.Series:
    """How tall each player is compared with the typical player at that spot (1.0 = typical).

    A small kid is not an occluded kid: his box is always shorter than the average one. The scale is
    the 80th percentile of his own box heights relative to the expected height (so the moments with hidden
    legs don't drag it down). Players seen in fewer than `min_rows` frames get 1.0.
    """
    if "track_id" not in players.columns:
        return pd.Series(1.0, index=players.index)
    ratio = (players.by2 - players.by1) / expected
    scale = ratio.groupby(players.track_id).transform(lambda r: r.quantile(quantile) if len(r) >= min_rows else 1.0)
    return scale.clip(low, high)


def fix_occluded_feet(players: pd.DataFrame, hom, min_ratio: float = 0.7) -> pd.DataFrame:
    """Return a copy where boxes shorter than `min_ratio` x the player's expected height
    get their foot point (px, py) and pitch position (x, y) re-estimated from the
    head. Adds a boolean `occluded` column."""
    out = players.copy()
    out["occluded"] = False
    model = expected_height_model(players)
    if model is None or hom is None:
        return out
    a, b = model
    expected = (a + b * out.by1) * player_scale(out, a + b * out.by1)      # this player's own full height here
    h, w = out.by2 - out.by1, out.bx2 - out.bx1
    # Short but wide = lying on the ground: the feet really are at the bottom of the box.
    occ = (h < min_ratio * expected) & (w <= 1.3 * h)
    if not occ.any():
        return out
    py = out.loc[occ, "by1"] + expected[occ]
    xy = hom.to_pitch(np.c_[out.loc[occ, "px"], py])
    out.loc[occ, "py"] = py
    out.loc[occ, "x"] = xy[:, 0]
    out.loc[occ, "y"] = xy[:, 1]
    out.loc[occ, "occluded"] = True
    return out
