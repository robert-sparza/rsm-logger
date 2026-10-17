import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer import analytics as A  # noqa: E402
from soccer_analyzer.pitch import Homography, Pitch  # noqa: E402
from soccer_analyzer.report import build_report  # noqa: E402


def synthetic_match(fps=10, seconds=6):
    """Player 1 (team 0) passes to player 2 (team 0), who loses it to player 3 (team 1)."""
    rows = []
    n = fps * seconds
    for f in range(n):
        t = f / fps
        p1 = (30 + t, 30)       # jogging at 1 m/s
        p2 = (50, 30)
        p3 = (70, 30)
        if t < 2:
            ball = p1
        elif t < 4:
            ball = p2
        else:
            ball = p3
        for tid, team, (x, y) in ((1, 0, p1), (2, 0, p2), (3, 1, p3)):
            rows.append(dict(frame=f, t=t, track_id=tid, cls="player", team=team, x=x, y=y))
        rows.append(dict(frame=f, t=t, track_id=-1, cls="ball", team=-1, x=ball[0] + 0.3, y=ball[1]))
    return pd.DataFrame(rows)


def test_homography_roundtrip():
    pitch = Pitch()
    pitch_pts = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], np.float32)
    img_pts = np.array([[200, 300], [1700, 300], [1900, 1000], [20, 1000]], np.float32)
    hom = Homography.fit(img_pts, pitch_pts, pitch)
    np.testing.assert_allclose(hom.to_pitch(img_pts), pitch_pts, atol=1e-3)
    np.testing.assert_allclose(hom.to_image(pitch_pts), img_pts, atol=1e-2)


def test_player_distance_and_speed():
    stats = A.player_stats(synthetic_match())
    assert stats.loc[1, "distance_m"] == pytest.approx(5.9, abs=0.3)  # smoothing trims the ends slightly
    assert stats.loc[2, "distance_m"] == pytest.approx(0, abs=1e-6)
    assert stats.loc[1, "top_speed_kmh"] == pytest.approx(3.6, abs=0.3)


def test_possession_and_passes():
    poss = A.possession(synthetic_match())
    share = A.possession_share(poss)
    assert share[0] == pytest.approx(4 / 6, abs=0.05)
    ev = A.passes(poss)
    assert list(ev.kind) == ["pass", "turnover"]
    assert list(ev.from_id) == [1, 2] and list(ev.to_id) == [2, 3]


def test_ball_interpolation_fills_short_gaps():
    df = synthetic_match()
    df = df[~((df.cls == "ball") & df.frame.between(10, 14))]
    ball = A.ball_track(df)
    assert set(range(10, 15)) <= set(ball.frame)


def test_report_writes_outputs(tmp_path):
    summary = build_report(synthetic_match(), tmp_path)
    assert (tmp_path / "heatmap_team0.png").exists()
    assert (tmp_path / "ball_path.png").exists()
    assert summary["passes"] == {"team0": 1}
    assert summary["turnovers"] == 1
