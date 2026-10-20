"""Offline clean-up of player tracks.

The tracker hands out a new ID whenever it loses a player for too long (two
players crossing, a player behind the coach). Afterwards we know the whole
match, so we can join a track that ends with one that starts shortly after at
a place the player could have run to. We also drop people whose typical
position is outside the lines (coach on the touchline, bench, spectators).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def _track_table(players: pd.DataFrame, edge_rows: int = 5) -> pd.DataFrame:
    g = players.sort_values("t").groupby("track_id")
    return pd.DataFrame({
        "t0": g.t.first(), "t1": g.t.last(),
        "x0": g.x.apply(lambda s: s.head(edge_rows).median()), "y0": g.y.apply(lambda s: s.head(edge_rows).median()),
        "x1": g.x.apply(lambda s: s.tail(edge_rows).median()), "y1": g.y.apply(lambda s: s.tail(edge_rows).median()),
    })


def stitch_tracks(players: pd.DataFrame, max_gap_s: float = 2.0, max_speed: float = 7.0,
                  slack_m: float = 1.5, teams: dict[int, int] | None = None) -> dict[int, int]:
    """Map every track id to the id of the first track in its chain.

    players: rows of one class with t, track_id, x, y (pitch metres).
    Track B continues track A if B starts at most `max_gap_s` after A ends and
    no further than a player could run in that gap (+ `slack_m` for jitter).
    When several ended tracks could continue into B, the closest one wins.
    teams: confident team per track id (leave out unsure ones); tracks of
    different teams are never joined.
    """
    teams = teams or {}
    chain_team: dict[int, int] = {}
    tracks = _track_table(players.dropna(subset=["x", "y"]))
    root: dict[int, int] = {}
    ends: dict[int, tuple[float, float, float]] = {}   # root -> (t1, x1, y1) of its last track
    for tid, tr in tracks.sort_values("t0").iterrows():
        best, best_score = None, np.inf
        for r, (t1, x1, y1) in ends.items():
            gap = tr.t0 - t1
            if not (0 < gap <= max_gap_s):
                continue
            if tid in teams and r in chain_team and teams[tid] != chain_team[r]:
                continue
            d = np.hypot(tr.x0 - x1, tr.y0 - y1)
            reach = max_speed * gap + slack_m
            if d <= reach and d / reach < best_score:
                best, best_score = r, d / reach
        r = best if best is not None else tid
        root[tid] = r
        if tid in teams:
            chain_team.setdefault(r, teams[tid])
        ends[r] = (tr.t1, tr.x1, tr.y1)
    return root


def off_pitch_tracks(players: pd.DataFrame, length: float, width: float, tolerance_m: float = 0.3,
                     touchline_m: float = 1.0, still_m: float = 2.0, min_s: float = 5.0) -> set[int]:
    """Not players: tracks whose median position is outside the lines (bench,
    spectators), or who stand almost still on a touchline for a while (a coach
    on the line). Goal lines are left alone so goalkeepers are kept."""
    p = players.dropna(subset=["x", "y"])
    g = p.groupby("track_id")
    med = g[["x", "y"]].median()
    out = ((med.x < -tolerance_m) | (med.x > length + tolerance_m)
           | (med.y < -tolerance_m) | (med.y > width + tolerance_m))
    on_touchline = (med.y < touchline_m) | (med.y > width - touchline_m)
    # How far the person wanders: 90th percentile distance from their median spot.
    spread = p.join(med, on="track_id", rsuffix="_m").assign(
        d=lambda d: np.hypot(d.x - d.x_m, d.y - d.y_m)).groupby("track_id").d.quantile(0.9)
    duration = g.t.max() - g.t.min()
    coach = on_touchline & (spread < still_m) & (duration >= min_s)
    return set(med.index[out | coach])


def split_on_team_change(players: pd.DataFrame, decide, window_s: float = 3.0, min_segment_s: float = 4.0,
                         first_new_id: int | None = None) -> pd.Series:
    """New track ids where the tracker swapped two players of different teams.

    When two players cross, the tracker sometimes swaps their IDs. If one wears
    a bib and the other does not, the vote column of that track flips. Each
    sample gets the team decided over a `window_s` window around it; runs
    shorter than `min_segment_s` are absorbed by their neighbours; the track is
    cut where the remaining runs change team. Returns new ids aligned with
    `players` (unchanged ids where nothing was cut).
    """
    new_ids = players.track_id.copy()
    next_id = first_new_id if first_new_id is not None else int(players.track_id.max()) + 1
    for tid, tr in players.groupby("track_id"):
        s = tr[tr.vote.notna()].sort_values("t")
        if len(s) < 10 or s.t.iloc[-1] - s.t.iloc[0] < 2 * min_segment_s:
            continue
        t = s.t.to_numpy()
        v = s.vote.to_numpy().astype(int)
        ones = np.concatenate([[0], np.cumsum(v)])
        lo = np.searchsorted(t, t - window_s / 2)
        hi = np.searchsorted(t, t + window_s / 2, side="right")
        labels = []
        for a, b in zip(lo, hi):
            n1 = ones[b] - ones[a]
            labels.append(decide(Counter({0: (b - a) - n1, 1: n1})))
        # Runs of equal label: [start_index, end_index, label]
        runs = []
        for i, lab in enumerate(labels):
            if runs and runs[-1][2] == lab:
                runs[-1][1] = i
            else:
                runs.append([i, i, lab])
        # Absorb short runs (noise) into the previous run, or the next one at the start.
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for k, (a, b, lab) in enumerate(runs):
                if t[b] - t[a] < min_segment_s:
                    if k > 0:
                        runs[k - 1][1] = b
                    else:
                        runs[k + 1][0] = a
                    runs.pop(k)
                    changed = True
                    break
            merged = []
            for r in runs:
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged
        if len(runs) < 2:
            continue
        cuts = [t[r[0]] for r in runs[1:]]          # start time of every run after the first
        seg = np.searchsorted(cuts, tr.t.to_numpy(), side="right")
        for k in range(1, len(runs)):
            new_ids.loc[tr.index[seg == k]] = next_id
            next_id += 1
    return new_ids
