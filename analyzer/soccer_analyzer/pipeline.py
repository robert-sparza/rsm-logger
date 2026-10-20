"""Frame-by-frame pipeline shared by offline analysis and the live view."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .ball import BallSelector
from .pitch import Homography, Pitch
from .stitch import off_pitch_tracks, stitch_tracks
from .teams import BibClassifier, TeamClassifier
from .tracking import Detection, Tracker
from .viz import TEAM_FALLBACK


class MatchPipeline:
    def __init__(self, tracker: Tracker, homography: Homography | None,
                 teams: TeamClassifier | BibClassifier | None = None,
                 team_warmup_frames: int = 50, color_every: int = 3, pitch_margin: float = 1.5,
                 fps: float = 30.0):
        self.tracker = tracker
        self.hom = homography
        self.pitch = homography.pitch if homography else Pitch()
        self.teams = teams or TeamClassifier()
        self.team_warmup_frames = team_warmup_frames
        self.color_every = color_every
        self.pitch_margin = pitch_margin
        self.ball = BallSelector(static_frames=int(3 * fps), gap_frames=int(0.5 * fps), memory_frames=int(fps))
        self.rows: list[dict] = []
        self._n = 0

    @property
    def team_colors(self):
        return TEAM_FALLBACK[:2]

    @property
    def jersey_colors(self):
        return self.teams.team_colors_bgr() if self.teams.fitted else None

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
        ball = self.ball.select(frame_idx, [(d, k) for d, k in zip(dets, on_pitch) if d.cls == "ball"])
        keep = [(d.cls == "player" and k) or d is ball for d, k in zip(dets, on_pitch)]
        dets = [d for d, k in zip(dets, keep) if k]
        feet, xy = feet[keep], xy[keep]

        for d, (px, py), (x, y) in zip(dets, feet, xy):
            team, bib = -1, np.nan
            if d.cls == "player" and frame_idx % self.color_every == 0:
                feat = self.teams.feature(frame, d.box)
                if isinstance(feat, float):
                    bib = feat          # saved for checking/tuning the bib rule
                if feat is not None:
                    if self.teams.fitted:
                        team = self.teams.vote(d.track_id, feat)
                    else:
                        self.teams.add_sample(feat)
            if d.cls == "player" and team < 0 and self.teams.fitted:
                team = self.teams.team_of(d.track_id)
            bx1, by1, bx2, by2 = (float(v) for v in d.box)
            self.rows.append(dict(frame=frame_idx, t=t, track_id=d.track_id, cls=d.cls, team=team,
                                  conf=d.conf, px=float(px), py=float(py), x=float(x), y=float(y),
                                  bx1=bx1, by1=by1, bx2=bx2, by2=by2, bib=bib))

        if not self.teams.fitted and self._n >= self.team_warmup_frames and len(self.teams.samples) >= 20:
            self.teams.fit()
        return dets

    def team_map(self) -> dict[int, int]:
        return {tid: self.teams.team_of(tid) for tid in self.teams.votes}

    def dataframe(self, clean: bool = True) -> pd.DataFrame:
        """All detections. With clean=True (and a calibration): people whose usual
        position is off the pitch are dropped, broken tracks are joined, and the
        joined players get short IDs (1, 2, 3...) by first appearance. The
        tracker's own id is kept in raw_track_id."""
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        df["raw_track_id"] = df.track_id
        players = df.cls == "player"
        if clean and self.hom is not None:
            off = off_pitch_tracks(df[players], self.pitch.length, self.pitch.width)
            df = df[~(players & df.track_id.isin(off))].copy()
            players = df.cls == "player"
            # Only pieces seen long enough (>= 10 colour samples) count as sure of their team.
            sure = {raw: self.teams.decide(v) for raw, v in self.teams.votes.items() if sum(v.values()) >= 10}
            root = stitch_tracks(df[players], teams={k: t for k, t in sure.items() if t >= 0})
            df.loc[players, "track_id"] = df.loc[players, "track_id"].map(root).fillna(df.track_id).astype(int)
        # Team = decision over all votes of every piece of the (joined) track.
        votes: dict[int, Counter] = {}
        for raw, tid in df.loc[players, ["raw_track_id", "track_id"]].drop_duplicates().itertuples(index=False):
            votes.setdefault(tid, Counter()).update(self.teams.votes.get(raw, Counter()))
        df.loc[players, "team"] = df.loc[players, "track_id"].map(
            {tid: self.teams.decide(v) for tid, v in votes.items()}).fillna(-1).astype(int)
        if clean:
            order = df[players].groupby("track_id").frame.min().sort_values().index
            df.loc[players, "track_id"] = df.loc[players, "track_id"].map({t: i + 1 for i, t in enumerate(order)})
        return df

    def save(self, out_dir: Path) -> pd.DataFrame:
        df = self.dataframe()
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tracks.csv", index=False)
        return df
