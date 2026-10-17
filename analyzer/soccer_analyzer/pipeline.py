"""Frame-by-frame pipeline shared by offline analysis and the live view."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pitch import Homography, Pitch
from .teams import TeamClassifier, jersey_color
from .tracking import Detection, Tracker
from .viz import TEAM_FALLBACK


class MatchPipeline:
    def __init__(self, tracker: Tracker, homography: Homography | None,
                 team_warmup_frames: int = 50, color_every: int = 3):
        self.tracker = tracker
        self.hom = homography
        self.pitch = homography.pitch if homography else Pitch()
        self.teams = TeamClassifier()
        self.team_warmup_frames = team_warmup_frames
        self.color_every = color_every
        self.rows: list[dict] = []
        self._n = 0

    @property
    def team_colors(self):
        return self.teams.team_colors_bgr() if self.teams.fitted else TEAM_FALLBACK[:2]

    def step(self, frame_idx: int, t: float, frame: np.ndarray) -> list[Detection]:
        dets = self.tracker.step(frame)
        self._n += 1
        feet = np.array([d.foot for d in dets], dtype=np.float32).reshape(-1, 2)
        if self.hom is not None and len(dets):
            # Ball "foot" is where it would touch the ground; fine for ground balls,
            # wrong for balls in the air (needs height estimation - later phase).
            xy = self.hom.to_pitch(feet)
        else:
            xy = np.full((len(dets), 2), np.nan, np.float32)

        for d, (px, py), (x, y) in zip(dets, feet, xy):
            team = -1
            if d.cls == "player" and frame_idx % self.color_every == 0:
                col = jersey_color(frame, d.box)
                if col is not None:
                    if self.teams.fitted:
                        team = self.teams.vote(d.track_id, col)
                    else:
                        self.teams.add_sample(col)
            if d.cls == "player" and team < 0 and self.teams.fitted:
                team = self.teams.team_of(d.track_id)
            self.rows.append(dict(frame=frame_idx, t=t, track_id=d.track_id, cls=d.cls, team=team,
                                  conf=d.conf, px=float(px), py=float(py), x=float(x), y=float(y)))

        if not self.teams.fitted and self._n >= self.team_warmup_frames and len(self.teams.samples) >= 20:
            self.teams.fit()
        return dets

    def team_map(self) -> dict[int, int]:
        return {tid: self.teams.team_of(tid) for tid in self.teams.votes}

    def dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        # Re-label every player row with its track's final majority team.
        tm = self.team_map()
        players = df.cls == "player"
        df.loc[players, "team"] = df.loc[players, "track_id"].map(tm).fillna(-1).astype(int)
        return df

    def save(self, out_dir: Path) -> pd.DataFrame:
        df = self.dataframe()
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tracks.csv", index=False)
        return df
