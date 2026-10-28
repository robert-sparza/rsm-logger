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


def stitch_tracks(players: pd.DataFrame, max_gap_s: float = 8.0, max_speed: float = 7.0,
                  slack_m: float = 1.5, teams: dict[int, int] | None = None, overlap_s: float = 0.2) -> dict[int, int]:
    """Map every track id to the id of the first track in its chain.

    players: rows of one class with t, track_id, x, y (pitch metres).
    Track B continues track A if B starts at most `max_gap_s` after A ends and
    no further than a player could have moved in that gap (+ `slack_m` for jitter).
    Short gaps allow sprinting speed; over long gaps (a player hidden behind a
    tree for several seconds) the allowed distance grows only ~2 m/s, since
    nobody sprints flat out the whole time and the pitch is small.
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
            if not (-overlap_s < gap <= max_gap_s):          # a new id often starts a frame or two before the old one ends
                continue
            if tid in teams and r in chain_team and teams[tid] != chain_team[r]:
                continue
            d = np.hypot(tr.x0 - x1, tr.y0 - y1)
            gap = max(gap, 0.0)
            reach = slack_m + min(max_speed * gap, 4.0 + 2.0 * gap)
            if d <= reach and d / reach < best_score:
                best, best_score = r, d / reach
        r = best if best is not None else tid
        root[tid] = r
        if tid in teams:
            chain_team.setdefault(r, teams[tid])
        ends[r] = (tr.t1, tr.x1, tr.y1)
    return root


def _reach(gap: float, max_speed: float = 7.0, slack_m: float = 1.5) -> float:
    """How far a player can plausibly get in `gap` seconds (same rule as stitch_tracks)."""
    gap = max(gap, 0.0)
    return slack_m + min(max_speed * gap, 4.0 + 2.0 * gap)


def _join_cost(rows: np.ndarray, edge_rows: int = 5, short_of_player=None, long_gap_s: float = 1.0,
               min_short: float = 0.8) -> float:
    """rows: (t, x, y, group) sorted by t, of two groups that are never seen in the same frame.

    Wherever the timeline switches from one group to the other, the jump from where the first was last seen to
    where the second shows up is compared with how far a player could run in that time. Returns the worst
    switch (distance / reach): <= 1 is a plausible single player, higher is less and less likely.
    short_of_player(t0, t1): share of that time in which the team was seen with fewer players than it has.
    Across a gap over `long_gap_s` the join only counts (otherwise inf) if the team was short most of the gap
    (`min_short`): if everyone was seen, the vanished player was on the pitch under another id, and the
    track that appears later is no more likely to be him than anyone else.
    """
    g = rows[:, 3]
    cuts = np.flatnonzero(g[1:] != g[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(rows)]])
    worst = 0.0
    for a, b, c in zip(bounds[:-2], bounds[1:-1], bounds[2:]):
        end = np.median(rows[max(a, b - edge_rows):b, 1:3], axis=0)
        start = np.median(rows[b:min(c, b + edge_rows), 1:3], axis=0)
        gap = rows[b, 0] - rows[b - 1, 0]
        if short_of_player is not None and gap > long_gap_s and short_of_player(rows[b - 1, 0], rows[b, 0]) < min_short:
            return np.inf
        worst = max(worst, float(np.hypot(*(start - end))) / _reach(gap))
    return worst


def link_by_count(players: pd.DataFrame, teams: dict[int, int], team_size: dict[int, int],
                  max_cost: float = 1.5, shared_frames: int = 3, same_spot_m: float = 1.5) -> dict[int, int]:
    """Join a team's tracks until it has no more tracks than players (e.g. 4 in bibs, 4 without).

    stitch_tracks only joins a track to one that ended shortly before, nearby. Knowing how many players a team
    has adds a stronger clue: two tracks of the same team that are never on the pitch at the same moment can be
    one player, even after a long gap or when the pieces interleave. Pairs are joined cheapest first (see
    _join_cost) while the team has more tracks than `team_size[team]`. Pairs seen more than `same_spot_m`
    apart in more than `shared_frames` frames are two people (both boxes on one spot is one player boxed
    twice, which the pipeline cleans up after). Joins costing more than `max_cost` are never made, so a stray
    extra (a coach, a wrong team) stays apart rather than being glued onto a player.
    Returns {track id: id of the group it joined} for every track that was joined.
    """
    root: dict[int, int] = {}
    p = players.dropna(subset=["x", "y"])
    for team, size in team_size.items():
        tids = [t for t, tm in teams.items() if tm == team and t in set(p.track_id)]
        if len(tids) <= size:
            continue
        sub = p[p.track_id.isin(tids)].sort_values("t").drop_duplicates(["track_id", "frame"])
        frame_t = players.drop_duplicates("frame").sort_values("t").t.to_numpy()     # every analysed frame
        seen = sub.groupby("t").size().reindex(frame_t, fill_value=0).to_numpy()   # team players seen per frame

        def short_of_player(t0: float, t1: float) -> float:
            inside = (frame_t > t0) & (frame_t < t1)
            return float((seen[inside] < size).mean()) if inside.any() else 1.0

        pos = {t: g.set_index("frame")[["x", "y"]] for t, g in sub.groupby("track_id")}
        rows = {t: g[["t", "x", "y"]].to_numpy() for t, g in sub.groupby("track_id")}
        members = {t: [t] for t in tids}
        cache: dict[tuple[int, int], float] = {}

        def apart(a: int, b: int) -> int:
            both = pos[a].join(pos[b], how="inner", rsuffix="_b")
            return int((np.hypot(both.x - both.x_b, both.y - both.y_b) > same_spot_m).sum())

        def cost(a: int, b: int) -> float:
            key = (min(a, b), max(a, b))
            if key not in cache:
                if apart(a, b) > shared_frames:
                    cache[key] = np.inf
                else:
                    r = np.vstack([np.column_stack([rows[a], np.zeros(len(rows[a]))]),
                                   np.column_stack([rows[b], np.ones(len(rows[b]))])])
                    r = r[np.argsort(r[:, 0], kind="stable")]
                    _, first = np.unique(r[:, 0], return_index=True)   # a shared frame: one sample is enough
                    cache[key] = _join_cost(r[np.sort(first)], short_of_player=short_of_player)
            return cache[key]

        while len(members) > size:
            pairs = [(cost(a, b), a, b) for i, a in enumerate(members) for b in list(members)[i + 1:]]
            c, a, b = min(pairs)
            if c > max_cost:
                break
            keep, gone = (a, b) if rows[a][0, 0] <= rows[b][0, 0] else (b, a)   # the group seen first keeps its id
            members[keep] += members.pop(gone)
            extra = pos.pop(gone)
            pos[keep] = pd.concat([pos[keep], extra[~extra.index.isin(pos[keep].index)]]).sort_index()
            rows[keep] = np.vstack([rows[keep], rows.pop(gone)])
            rows[keep] = rows[keep][np.argsort(rows[keep][:, 0], kind="stable")]
            cache = {k: v for k, v in cache.items() if keep not in k and gone not in k}
        for keep, ms in members.items():
            for t in ms:
                if t != keep:
                    root[t] = keep
    return root


def off_pitch_tracks(players: pd.DataFrame, length: float, width: float, tolerance_m: float = 0.3,
                     touchline_m: float = 1.0, still_m: float = 2.0, min_s: float = 5.0,
                     drop_standing_on_line: bool = True, reasons: dict | None = None,
                     ignore_zones=None, static_s: float = 0.0, static_spread_m: float = 0.6,
                     seated_aspect: float = 0.75, idle_edge_m_per_min: float = 0.0, edge_m: float = 1.5,
                     idle_s: float = 20.0) -> set[int]:
    """Tracks that are not players, by reason:

    off_pitch         median position outside the lines (bench, spectators)
    standing_on_line  stands almost still on a touchline for a while (a coach on the line)
                      (drop_standing_on_line=False keeps them: players waiting their turn look like that)
    ignore_zone       median position inside a zone you listed (x_min, y_min, x_max, y_max in metres)
    static            barely moves for `static_s` seconds (static_s > 0): a coach, a post, a cone. Anything
                      moving less than ~15 cm over 10 s is always dropped: no person stands that still
    seated            the box is as wide as tall (sitting or lying), for 5+ seconds
    idle_at_edge      (idle_edge_m_per_min > 0) stays within `edge_m` of a boundary line for `idle_s`+ seconds and covers
                      less than that many metres per minute: a coach pacing the line. Players in a drill do 40+
    Goal lines are left alone so goalkeepers are kept. `reasons`, if given, is filled with {reason: ids}.
    """
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
    line = on_touchline & (spread < still_m) & (duration >= min_s) & ~out
    zone = pd.Series(False, index=med.index)
    for x1, y1, x2, y2 in (ignore_zones or []):
        zone |= med.x.between(x1, x2) & med.y.between(y1, y2)
    static = (spread < 0.15) & (duration >= 10.0)
    if static_s > 0:
        static |= (spread < static_spread_m) & (duration >= static_s)
    seated = pd.Series(False, index=med.index)
    if {"bx1", "bx2", "by1", "by2"} <= set(p.columns):
        aspect = ((p.bx2 - p.bx1) / (p.by2 - p.by1).clip(lower=1)).groupby(p.track_id).median()
        seated = (aspect > seated_aspect) & (duration >= 5.0)
    idle = pd.Series(False, index=med.index)
    if idle_edge_m_per_min > 0:
        from .analytics import smooth_track
        near_edge = (np.minimum.reduce([med.x.abs(), (length - med.x).abs(), med.y.abs(), (width - med.y).abs()]) <= edge_m)
        for tid in med.index[near_edge & (duration[med.index] >= idle_s)]:
            q_ = p[p.track_id == tid].sort_values("t")
            dts = q_.t.diff().dropna()
            fps = float(1 / dts[dts > 0].median()) if (dts > 0).any() else 30.0
            ts, xs, ys = smooth_track(q_, fps)
            d, dt = np.hypot(np.diff(xs), np.diff(ys)), np.diff(ts)
            ok = (dt > 0) & (dt <= 1.0)
            if ok.sum() and dt[ok].sum() >= idle_s / 2 and d[ok].sum() / dt[ok].sum() * 60 < idle_edge_m_per_min:
                idle[tid] = True
    if reasons is not None:
        reasons["idle_at_edge"] = set(med.index[idle & ~out])
        reasons["off_pitch"] = set(med.index[out])
        reasons["standing_on_line"] = set(med.index[line])
        reasons["ignore_zone"] = set(med.index[zone])
        reasons["static"] = set(med.index[static & ~out])
        reasons["seated"] = set(med.index[seated])
    drop = out | zone | static | seated | idle | (line if drop_standing_on_line else False)
    return set(med.index[drop])


def split_on_team_change(players: pd.DataFrame, decide, window_s: float = 1.5, min_segment_s: float = 1.5,
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


def cap_team_size(players: pd.DataFrame, teams: dict[int, int], max_players: int, team: int = 1,
                  segment_gap_s: float = 1.0, min_run_s: float = 1.0) -> tuple[pd.Series, dict[int, int]]:
    """Never more than `max_players` players of `team` at once (e.g. 2 players wear the bib).

    A non-bib player standing next to bib wearers picks up some bib colour, so his track can be
    labelled bib team and for a while there are three bibs on the pitch. Each stretch of time that has too
    many ("crowded") is looked at on its own: every track in it is scored by the share of bib votes it
    got during that stretch, and in each frame the weakest ones above the limit are put in the other team.
    Their rows get a new track id, so a track never changes team in the middle. Stretches shorter than
    `min_run_s` are left alone: they are borderline flickers, and a one-frame track is worse than an
    extra bib for a moment. Returns (new ids aligned with `players`, team per id).
    """
    ids = players.track_id.copy()
    new_teams = dict(teams)
    in_team = players[players.track_id.map(teams).eq(team)]
    n_per_frame = in_team.groupby("frame").size()
    crowded = n_per_frame[n_per_frame > max_players].index
    if len(crowded) == 0:
        return ids, new_teams

    t_of = in_team.drop_duplicates("frame").set_index("frame").t
    ct = t_of.loc[crowded].sort_values()
    segment = (ct.diff() > segment_gap_s).cumsum()                  # crowded stretches
    overall = in_team.groupby("track_id").vote.mean().fillna(0.0)
    rows_in_crowd = in_team[in_team.frame.isin(crowded)].assign(seg=lambda d: d.frame.map(segment))
    seg_score = rows_in_crowd.groupby(["seg", "track_id"]).vote.mean()

    demote = []
    for (seg, frame), rows in rows_in_crowd.groupby(["seg", "frame"]):
        scores = [seg_score.get((seg, tid), np.nan) for tid in rows.track_id]
        scores = [overall[tid] if np.isnan(sc) else sc for sc, tid in zip(scores, rows.track_id)]
        order = np.argsort(scores, kind="stable")
        demote.extend(rows.index[order[: len(rows) - max_players]])
    other = 1 - team
    next_id = int(max(players.track_id.max(), max(teams, default=0))) + 1
    d = players.loc[pd.Index(demote)].sort_values(["track_id", "frame"])
    # Runs of consecutive demoted frames of one track become one new track each.
    run = ((d.track_id != d.track_id.shift()) | (d.frame - d.frame.shift() > 3)).cumsum()
    for _, grp in d.groupby(run):
        if grp.t.max() - grp.t.min() < min_run_s:
            continue
        ids[grp.index] = next_id
        new_teams[next_id] = other
        next_id += 1
    return ids, new_teams
