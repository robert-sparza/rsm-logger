"""Frame-by-frame pipeline shared by offline analysis and the live view."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .ball import BallSelector, cone_colored
from .pitch import Homography, Pitch
from .occlusion import fix_occluded_feet
from .stitch import cap_team_size, off_pitch_tracks, split_on_team_change, stitch_tracks
from .teams import BibClassifier, TeamClassifier
from .tracking import Detection, Tracker
from .viz import TEAM_FALLBACK


def reliable_colour_box(box, all_boxes, max_overlap: float = 0.3) -> bool:
    """Shirt colour is only trusted for an upright player not mixed up with another:
    a box wider than tall is someone lying down (the "torso" crop misses the shirt),
    and a box overlapping another player's mixes the two shirts."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or w > 1.2 * h:
        return False
    for o in all_boxes:
        if o is box:
            continue
        iw = min(x2, o[2]) - max(x1, o[0])
        ih = min(y2, o[3]) - max(y1, o[1])
        if iw > 0 and ih > 0 and iw * ih > max_overlap * w * h:
            return False
    return True


class MatchPipeline:
    def __init__(self, tracker: Tracker, homography: Homography | None,
                 teams: TeamClassifier | BibClassifier | None = None,
                 team_warmup_frames: int = 50, color_every: int = 3, pitch_margin: float = 1.5,
                 fps: float = 30.0, reject_cone_colours: bool = True, bib_count: int | None = None,
                 drop_standing_on_line: bool = True, ignore_zones=None, drop_static_s: float = 0.0,
                 ball_max_speed: float = 18.0, idle_edge_m_per_min: float = 0.0):
        self.idle_edge_m_per_min = idle_edge_m_per_min     # slower than this near a line = not a player (0 = off)
        self.ignore_zones = ignore_zones or []      # [[x_min, y_min, x_max, y_max], ...] in metres: nobody in there is a player
        self.drop_static_s = drop_static_s          # people who barely move for this many seconds are not players (0 = off)
        self.fps = fps
        self.ball_max_speed = ball_max_speed        # m/s: a ball can't jump faster than this between detections
        self._ball_last: tuple[int, float, float] | None = None       # frame, x, y of the last accepted ball
        self._challenger: tuple[float, float, int] | None = None     # a ball somewhere else that keeps showing up
        self.drop_standing_on_line = drop_standing_on_line
        self.dropped: dict[str, set] = {}          # tracks removed as not-players, by reason (filled by dataframe())
        self.bib_count = bib_count        # how many players wear the bib (team 1): never label more than this at once
        self.tracker = tracker
        self.hom = homography
        self.pitch = homography.pitch if homography else Pitch()
        self.teams = teams or TeamClassifier()
        self.team_warmup_frames = team_warmup_frames
        self.color_every = color_every
        self.pitch_margin = pitch_margin
        self.ball = BallSelector(static_frames=int(1.5 * fps), gap_frames=int(5 * fps), memory_frames=int(fps),
                                 confirmed_memory_frames=int(60 * fps))
        self.reject_cone_colours = reject_cone_colours
        self.rows: list[dict] = []
        self._n = 0

    @property
    def team_colors(self):
        return TEAM_FALLBACK[:2]

    @property
    def jersey_colors(self):
        return self.teams.team_colors_bgr() if self.teams.fitted else None

    def _ball_plausible(self, frame: int, x: float, y: float) -> bool:
        """A ball can't teleport. A detection far from where the ball just was (a boot, a bottle) is held back
        until it keeps showing up at the same place for ~0.4 s; then the ball really is there."""
        last = self._ball_last
        dt = (frame - last[0]) / self.fps if last else None
        if last is None or dt > 2.0 or np.hypot(x - last[1], y - last[2]) <= 3.0 + self.ball_max_speed * dt:
            self._ball_last, self._challenger = (frame, x, y), None
            return True
        c = self._challenger
        n = c[2] + 1 if c is not None and np.hypot(x - c[0], y - c[1]) <= 2.0 else 1
        self._challenger = (x, y, n)
        if n >= max(3, int(0.4 * self.fps)):
            self._ball_last, self._challenger = (frame, x, y), None
            return True
        return False

    def step(self, frame_idx: int, t: float, frame: np.ndarray) -> list[Detection]:
        dets = self.tracker.step(frame)
        self._n += 1
        feet = np.array([d.foot for d in dets], dtype=np.float32).reshape(-1, 2)
        if self.hom is not None and len(dets):
            # Ball "foot" is where it would touch the ground; fine for ground balls,
            # wrong for balls in the air (needs height estimation - later phase).
            xy = self.hom.to_pitch(feet)
            on_pitch = self.hom.on_pitch(xy, self.pitch_margin)
        else:
            xy = np.full((len(dets), 2), np.nan, np.float32)
            on_pitch = np.ones(len(dets), bool)

        # Spectators, subs, coaches and people on neighbouring fields are not players.
        ball = self.ball.select(frame_idx, [(d, k and not (self.reject_cone_colours and cone_colored(frame, d.box)))
                                            for d, k in zip(dets, on_pitch) if d.cls == "ball"])
        if ball is not None and self.hom is not None:
            i = next(j for j, d in enumerate(dets) if d is ball)
            if not self._ball_plausible(frame_idx, float(xy[i][0]), float(xy[i][1])):
                ball = None
        keep = [(d.cls == "player" and k) or d is ball for d, k in zip(dets, on_pitch)]
        dets = [d for d, k in zip(dets, keep) if k]
        feet, xy = feet[keep], xy[keep]

        player_boxes = [d.box for d in dets if d.cls == "player"]
        for d, (px, py), (x, y) in zip(dets, feet, xy):
            team, bib, vote = -1, np.nan, np.nan
            if d.cls == "player" and frame_idx % self.color_every == 0 and reliable_colour_box(d.box, player_boxes):
                feat = self.teams.feature(frame, d.box)
                if isinstance(feat, float):
                    bib = feat          # saved for checking/tuning the bib rule
                if feat is not None:
                    if self.teams.fitted:
                        vote = self.teams.sample_vote(feat)
                        team = self.teams.vote(d.track_id, feat)
                    else:
                        self.teams.add_sample(feat)
            if d.cls == "player" and team < 0 and self.teams.fitted:
                team = self.teams.team_of(d.track_id)
            bx1, by1, bx2, by2 = (float(v) for v in d.box)
            self.rows.append(dict(frame=frame_idx, t=t, track_id=d.track_id, cls=d.cls, team=team,
                                  conf=d.conf, px=float(px), py=float(py), x=float(x), y=float(y),
                                  bx1=bx1, by1=by1, bx2=bx2, by2=by2, bib=bib, vote=vote))

        if not self.teams.fitted and self._n >= self.team_warmup_frames and len(self.teams.samples) >= 20:
            self.teams.fit()
        return dets

    def team_map(self) -> dict[int, int]:
        return {tid: self.teams.team_of(tid) for tid in self.teams.votes}

    def dataframe(self, clean: bool = True) -> pd.DataFrame:
        """All detections. With clean=True and a calibration:
        1. tracks are cut where the tracker swapped players of different teams,
        2. people whose usual position is off the pitch (coach, bench) are dropped,
        3. broken tracks are joined (never across teams),
        4. every player's team is decided over all of its samples,
        5. players get short IDs (1, 2, 3...) by first appearance.
        The tracker's own id is kept in raw_track_id."""
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        df["raw_track_id"] = df.track_id
        players = df.cls == "player"

        def votes_by_track(frame: pd.DataFrame) -> dict[int, Counter]:
            v = frame[frame.vote.notna()]
            return {tid: Counter(g.vote.astype(int)) for tid, g in v.groupby("track_id")}

        if clean and self.hom is not None:
            fixed = fix_occluded_feet(df[players], self.hom)
            df.loc[players, ["py", "x", "y"]] = fixed[["py", "x", "y"]]
            df["occluded"] = False
            df.loc[players, "occluded"] = fixed["occluded"]
            df.loc[players, "track_id"] = split_on_team_change(df[players], self.teams.decide)
            off = off_pitch_tracks(df[players], self.pitch.length, self.pitch.width,
                                   drop_standing_on_line=self.drop_standing_on_line, reasons=self.dropped,
                                   ignore_zones=self.ignore_zones, static_s=self.drop_static_s,
                                   idle_edge_m_per_min=self.idle_edge_m_per_min)
            df = df[~(players & df.track_id.isin(off))].copy()
            players = df.cls == "player"
            # Only pieces with >= 10 colour samples count as sure of their team.
            sure = {tid: self.teams.decide(c) for tid, c in votes_by_track(df[players]).items()
                    if sum(c.values()) >= 10}
            root = stitch_tracks(df[players], teams={k: t for k, t in sure.items() if t >= 0})
            df.loc[players, "track_id"] = df.loc[players, "track_id"].map(root).fillna(df.track_id).astype(int)
        teams = {tid: self.teams.decide(c) for tid, c in votes_by_track(df[players]).items()}
        if clean and self.bib_count and self.hom is not None:
            ids, teams = cap_team_size(df[players], teams, self.bib_count)
            df.loc[players, "track_id"] = ids
            # The pieces just moved out of the bib team can join the other team's tracks.
            root = stitch_tracks(df[players], teams={k: t for k, t in teams.items() if t >= 0})
            df.loc[players, "track_id"] = df.loc[players, "track_id"].map(root).fillna(df.track_id).astype(int)
            teams = {root.get(k, k): t for k, t in teams.items()}
        if clean:
            # Joining tracks that overlap by a frame can leave two boxes for one player in that frame: keep the surer one.
            dup = df[players].sort_values("conf", ascending=False).duplicated(["frame", "track_id"])
            df = df.drop(dup[dup].index)
            players = df.cls == "player"
        df.loc[players, "team"] = df.loc[players, "track_id"].map(teams).fillna(-1).astype(int)
        if clean:
            order = df[players].groupby("track_id").frame.min().sort_values().index
            df.loc[players, "track_id"] = df.loc[players, "track_id"].map({t: i + 1 for i, t in enumerate(order)})
        return df

    def save(self, out_dir: Path) -> pd.DataFrame:
        df = self.dataframe()
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tracks.csv", index=False)
        return df
