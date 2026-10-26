"""Export the tracks table to kloppy's standard tracking-data format.

Once in kloppy's format the data can go to other football libraries (floodlight,
mplsoccer, ...) without custom code. Needs `pip install -r requirements-extras.txt`.

Teams: team 0 is "home", team 1 is "away" (with bibs: 1 = the bib colour). Players whose team
is unknown (-1) are left out. Each tracked player is one kloppy player (jersey number = track id).
Coordinates are kloppy's normalized pitch: 0..1, origin top-left, y downwards. Our pitch has
the same orientation (origin at the far-left corner, y towards the camera), so only the scale changes.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .pitch import Pitch


def to_kloppy(df: pd.DataFrame, pitch: Pitch, fps: float, team_names=("Team 0", "Team 1")):
    from kloppy.domain import (BallState, DatasetFlag, Ground, KloppyCoordinateSystem, Metadata, Orientation,
                               Period, Player, PlayerData, Point, Point3D, Provider, Team, TrackingDataset, Frame)
    players_df = df[(df.cls == "player") & df.team.isin([0, 1])].dropna(subset=["x", "y"])
    if players_df.empty:
        raise ValueError("No players with a team and pitch position to export.")
    teams = {0: Team("0", team_names[0], Ground.HOME), 1: Team("1", team_names[1], Ground.AWAY)}
    people: dict[int, Player] = {}
    for tid, team in players_df.groupby("track_id").team.agg(lambda s: int(s.mode().iat[0])).items():
        pl = Player(f"track_{int(tid)}", teams[team], int(tid), name=f"Track {int(tid)}")
        teams[team].players.append(pl)
        people[int(tid)] = pl

    last = float(df.t.max())
    period = Period(1, timedelta(0), timedelta(seconds=last))
    coords = KloppyCoordinateSystem(pitch.length, pitch.width)
    meta = Metadata(periods=[period], teams=[teams[0], teams[1]], coordinate_system=coords,
                    pitch_dimensions=coords.pitch_dimensions, orientation=Orientation.NOT_SET,
                    flags=DatasetFlag(0), provider=Provider.KLOPPY, frame_rate=float(fps),
                    attributes={"source": "rsm-soccer-coach"})

    ball = df[df.cls == "ball"].drop_duplicates("frame").set_index("frame")[["x", "y"]].dropna()
    by_frame = {f: g for f, g in players_df.groupby("frame")}
    t_of = df.drop_duplicates("frame").set_index("frame").t
    records = []
    for f in sorted(by_frame):
        data = {people[int(r.track_id)]: PlayerData(coordinates=Point(r.x / pitch.length, r.y / pitch.width))
                for r in by_frame[f].itertuples()}
        bx, by = (ball.loc[f, "x"] / pitch.length, ball.loc[f, "y"] / pitch.width) if f in ball.index else (np.nan, np.nan)
        records.append(Frame(period=period, timestamp=timedelta(seconds=float(t_of[f])), statistics=[],
                             ball_owning_team=None, ball_state=BallState.ALIVE, frame_id=int(f),
                             players_data=data, other_data={}, ball_coordinates=Point3D(bx, by, np.nan)))
    return TrackingDataset(records=records, metadata=meta)
