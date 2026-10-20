import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.ball import BallSelector  # noqa: E402
from soccer_analyzer.pipeline import MatchPipeline  # noqa: E402
from soccer_analyzer.pitch import Homography, Pitch  # noqa: E402
from soccer_analyzer.teams import BibClassifier  # noqa: E402
from soccer_analyzer.tracking import Detection  # noqa: E402

# Colours measured on real night footage (HSV, OpenCV ranges).
TURF_HSV = (46, 88, 135)
VEST_HSV = (40, 200, 200)
MAROON_HSV = (2, 150, 90)


def hsv_bgr(hsv):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def frame_with_players(players):
    """players: list of (box, shirt_hsv). Returns a 400x800 BGR frame."""
    img = np.full((400, 800, 3), hsv_bgr(TURF_HSV), np.uint8)
    for (x1, y1, x2, y2), shirt in players:
        h = y2 - y1
        cv2.rectangle(img, (x1 + 4, y1 + int(0.1 * h)), (x2 - 4, y1 + int(0.55 * h)), hsv_bgr(shirt), -1)
        cv2.rectangle(img, (x1 + 6, y1 + int(0.55 * h)), (x2 - 6, y2), (30, 30, 30), -1)
    return img


class FakeTracker:
    def __init__(self, script):
        self.script = script   # frame index -> list of Detection
        self.i = 0

    def step(self, frame):
        dets = self.script(self.i)
        self.i += 1
        return dets


def test_bib_classifier_on_measured_colours():
    vest_box, maroon_box = (100, 100, 140, 200), (300, 100, 340, 200)
    img = frame_with_players([(vest_box, VEST_HSV), (maroon_box, MAROON_HSV)])
    bib = BibClassifier("neon_yellow")
    assert bib.feature(img, vest_box) > 0.3
    assert bib.feature(img, maroon_box) < 0.01
    # Turf alone must not look like a neon vest.
    turf = np.full((50, 50, 3), hsv_bgr(TURF_HSV), np.uint8)
    assert bib.feature(turf, (0, 0, 50, 50)) == 0


def test_bib_track_counts_even_if_vest_seen_in_few_frames():
    bib = BibClassifier("neon_yellow", min_share=0.2)
    for f in [0.0] * 7 + [0.1] * 3:    # vest visible in 30% of frames (back turned otherwise)
        bib.vote(5, f)
    for f in [0.0] * 10:
        bib.vote(6, f)
    assert bib.team_of(5) == 1 and bib.team_of(6) == 0


def test_ball_selector_ignores_cones_and_follows_moving_ball():
    sel = BallSelector(static_frames=30, gap_frames=5, memory_frames=10)
    cone = Detection(-1, "ball", (500, 300, 510, 310), 0.9)
    picked = []
    for f in range(60):
        x = 100 + 5 * f
        ball = Detection(-1, "ball", (x, 200, x + 10, 210), 0.4)
        picked.append(sel.select(f, [(cone, True), (ball, True)]))
    # Once the cone has sat still for 30 frames, only the moving ball is picked.
    assert all(p.conf == 0.4 for p in picked[35:])
    off = Detection(-1, "ball", (0, 0, 10, 10), 0.99)
    assert sel.select(61, [(off, False)]) is None


def test_pipeline_drops_spectators_and_static_ball():
    # 10 px per metre, pitch 40 x 20 m drawn at (0,0)-(400,200) in a 800x400 frame.
    pitch = Pitch(length=40, width=20)
    pts = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], np.float32)
    hom = Homography.fit(pts * 10, pts, pitch)
    players = [((100, 60, 120, 120), VEST_HSV), ((250, 80, 270, 140), MAROON_HSV),
               ((600, 300, 620, 360), MAROON_HSV)]                      # last one: spectator off the pitch
    img = frame_with_players([(b, s) for b, s in players])

    def script(i):
        dets = [Detection(tid, "player", b, 0.9) for tid, (b, _) in enumerate(players, start=1)]
        dets.append(Detection(-1, "ball", (300, 150, 306, 156), 0.9))           # cone, never moves
        dets.append(Detection(-1, "ball", (50 + 2 * i, 100, 56 + 2 * i, 106), 0.3))  # real ball
        return dets

    pipe = MatchPipeline(FakeTracker(script), hom, teams=BibClassifier("neon_yellow"), fps=10)
    for i in range(60):
        pipe.step(i, i / 10, img)
    df = pipe.dataframe()
    assert set(df[df.cls == "player"].track_id) == {1, 2}                       # spectator dropped
    assert df[df.cls == "player"].groupby("track_id").team.first().to_dict() == {1: 1, 2: 0}
    late_ball = df[(df.cls == "ball") & (df.frame >= 40)]
    assert len(late_ball) == 20 and (late_ball.conf == 0.3).all()               # cone ignored
