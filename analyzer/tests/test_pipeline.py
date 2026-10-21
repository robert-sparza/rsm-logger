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


def test_stitching_joins_broken_tracks_but_not_across_teams():
    from soccer_analyzer.stitch import off_pitch_tracks, stitch_tracks
    import pandas as pd
    rows = []
    for f in range(100):                      # player A walks right, ID changes at t=5 s (gap 0.5 s)
        t = f / 10
        if 50 <= f < 55:
            continue
        rows.append(dict(t=t, track_id=1 if f < 50 else 7, x=5 + 0.5 * t, y=5))
    for f in range(100):                      # player B stands still far away, ID changes too
        t = f / 10
        if 50 <= f < 55:
            continue
        rows.append(dict(t=t, track_id=2 if f < 50 else 8, x=20, y=12))
    for f in range(100):                      # coach on the near touchline (y > width)
        rows.append(dict(t=f / 10, track_id=3, x=14, y=16.9))
    df = pd.DataFrame(rows)
    root = stitch_tracks(df)
    assert root[7] == 1 and root[8] == 2
    # Same geometry but sure of different teams -> must not be joined.
    root = stitch_tracks(df, teams={1: 0, 7: 1})
    assert root[7] == 7
    assert off_pitch_tracks(df, 28.5, 16.3) == {3}


def test_coach_standing_on_touchline_is_not_a_player():
    from soccer_analyzer.stitch import off_pitch_tracks
    import pandas as pd
    rows = [dict(t=f / 10, track_id=1, x=14 + 0.2 * np.sin(f / 7), y=16.2) for f in range(200)]     # coach on the line
    rows += [dict(t=f / 10, track_id=2, x=10 + f * 0.05, y=15.8) for f in range(200)]              # winger running along it
    rows += [dict(t=f / 10, track_id=3, x=0.3, y=8 + 0.3 * np.sin(f / 9)) for f in range(200)]     # keeper on the goal line
    assert off_pitch_tracks(pd.DataFrame(rows), 28.5, 16.3) == {1}


def test_lawn_behind_far_side_player_is_not_a_vest():
    lawn = (38, 140, 190)                       # lit park lawn behind the pitch, same hue as a neon vest
    img = np.full((200, 100, 3), hsv_bgr(lawn), np.uint8)
    cv2.rectangle(img, (34, 20), (66, 190), hsv_bgr(MAROON_HSV), -1)      # player fills the middle half of the box, as measured
    assert BibClassifier("neon_yellow").feature(img, (20, 10, 80, 190)) == 0


def test_split_where_tracker_swapped_vest_and_non_vest_players():
    from soccer_analyzer.stitch import split_on_team_change
    import pandas as pd
    bib = BibClassifier("neon_yellow")
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 0.1)
    # 0-15 s vest player (vest seen in half the samples), then the ID moves to a non-vest player.
    vote = np.where(t < 15, rng.random(len(t)) < 0.5, rng.random(len(t)) < 0.02).astype(float)
    df = pd.DataFrame(dict(t=t, track_id=5, vote=vote, x=10.0, y=5.0))
    ids = split_on_team_change(df, bib.decide, first_new_id=100)
    assert set(ids[t < 13]) == {5} and set(ids[t > 17]) == {100}
    # A steady vest player is never split even though the vest is only seen half the time.
    steady = df.assign(vote=(rng.random(len(t)) < 0.5).astype(float))
    assert set(split_on_team_change(steady, bib.decide)) == {5}


def test_cone_colours_are_not_the_ball():
    from soccer_analyzer.ball import cone_colored
    img = np.full((40, 120, 3), hsv_bgr(TURF_HSV), np.uint8)
    cv2.ellipse(img, (20, 20), (10, 6), 0, 0, 360, hsv_bgr((12, 220, 230)), -1)    # orange cone
    cv2.ellipse(img, (60, 20), (10, 6), 0, 0, 360, hsv_bgr((30, 200, 220)), -1)    # yellow cone
    cv2.circle(img, (100, 20), 9, (235, 235, 235), -1)                             # white ball
    cv2.circle(img, (100, 20), 3, (20, 20, 20), -1)
    assert cone_colored(img, (8, 12, 32, 28)) and cone_colored(img, (48, 12, 72, 28))
    assert not cone_colored(img, (90, 10, 110, 30))


def test_flickering_static_spot_becomes_static():
    sel = BallSelector(static_frames=30, gap_frames=50, memory_frames=10)
    spare = Detection(-1, "ball", (500, 300, 510, 310), 0.9)
    picks = []
    for f in range(120):
        cands = [(spare, True)] if f % 7 in (0, 1, 2) else []      # detected 3 frames out of 7
        x = 100 + 4 * f
        cands.append((Detection(-1, "ball", (x, 200, x + 10, 210), 0.3), True))
        picks.append(sel.select(f, cands))
    assert all(p.conf == 0.3 for p in picks[40:])


def test_hidden_legs_get_feet_from_head():
    from soccer_analyzer.occlusion import fix_occluded_feet
    import pandas as pd
    pitch = Pitch(length=40, width=20)
    pts = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], np.float32)
    hom = Homography.fit(pts * 10, pts, pitch)          # 10 px per metre, top-down for simplicity
    rng = np.random.default_rng(1)
    rows = []
    for i in range(300):                                 # fully visible players: height = 60 + 0.5 * top
        top = rng.uniform(0, 150)
        h = 60 + 0.5 * top
        rows.append(dict(px=200.0, by1=top, by2=top + h, bx1=180.0, bx2=180 + 0.4 * h))
    rows.append(dict(px=200.0, by1=100.0, by2=150.0, bx1=180.0, bx2=220.0))     # legs hidden: 50 px of 110
    rows.append(dict(px=200.0, by1=100.0, by2=150.0, bx1=150.0, bx2=250.0))     # lying down: wide box
    df = pd.DataFrame(rows)
    df["py"] = df.by2
    df[["x", "y"]] = hom.to_pitch(df[["px", "py"]].to_numpy())
    out = fix_occluded_feet(df, hom)
    hidden, lying = out.iloc[-2], out.iloc[-1]
    assert hidden.occluded and abs(hidden.py - 210) < 10       # feet put back near 100 + 110
    assert not lying.occluded and lying.py == 150
    assert out.iloc[:-2].occluded.mean() < 0.05


def test_colour_ignored_for_lying_or_overlapping_players():
    from soccer_analyzer.pipeline import reliable_colour_box
    upright, lying = (0, 0, 40, 100), (0, 0, 150, 60)
    other = (10, 0, 50, 100)                              # overlaps `upright` heavily
    assert reliable_colour_box(upright, [upright])
    assert not reliable_colour_box(lying, [lying])
    assert not reliable_colour_box(upright, [upright, other])


def test_frame_selection_prefers_hard_frames_and_spreads_the_rest():
    from soccer_analyzer.dataset import select_frames
    cands = [(i, i * 0.5, 5.0 if i in (10, 30, 50) else 0.0) for i in range(200)]   # 100 s of candidates
    chosen = select_frames(cands, n=20, min_gap_s=1.0)
    assert len(chosen) == 20 and {10, 30, 50} <= set(chosen)
    times = sorted(c * 0.5 for c in chosen)
    assert min(b - a for a, b in zip(times, times[1:])) >= 1.0


def test_crowded_frame_scores_higher():
    from soccer_analyzer.dataset import frame_score
    apart = [(0, 0, 40, 100), (300, 0, 340, 100)]
    bunched = [(0, 0, 40, 100), (30, 5, 70, 105)]
    ball_at_feet = [(42, 92, 52, 102)]
    assert frame_score(bunched, [], []) > frame_score(apart, [], [])
    assert frame_score(apart, ball_at_feet, []) > frame_score(apart, [], [])


def test_yolo_lines_and_roboflow_yaml(tmp_path):
    from soccer_analyzer.dataset import fix_roboflow_yaml, yolo_lines
    import yaml
    line = yolo_lines([(1, (90, 40, 110, 60)), (0, (-10, 0, 0, 5))], 200, 100)
    assert line == ["1 0.500000 0.500000 0.100000 0.200000"]          # box fully outside is dropped
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
                                        "nc: 3\nnames: ['ball', 'cone', 'player']\n")
    d = yaml.safe_load(fix_roboflow_yaml(tmp_path / "data.yaml").read_text())
    assert d["train"] == str(tmp_path / "train" / "images") and d["val"] == str(tmp_path / "valid" / "images")
    assert "test" not in d and d["names"] == ["ball", "cone", "player"]


def test_class_ids_by_name_for_stock_and_trained_models():
    from soccer_analyzer.tracking import BALL_NAMES, PLAYER_NAMES, class_ids
    coco = {0: "person", 32: "sports ball", 2: "car"}
    ours = {0: "ball", 1: "cone", 2: "player"}          # Roboflow sorts names alphabetically
    assert class_ids(coco, PLAYER_NAMES) == {0} and class_ids(coco, BALL_NAMES) == {32}
    assert class_ids(ours, PLAYER_NAMES) == {2} and class_ids(ours, BALL_NAMES) == {0}


def test_roboflow_yaml_found_one_folder_deeper_and_clear_error_for_label_batch(tmp_path):
    from soccer_analyzer.dataset import fix_roboflow_yaml
    import pytest
    import yaml
    (tmp_path / "Soccer.v1i.yolov11" / "train" / "images").mkdir(parents=True)
    (tmp_path / "Soccer.v1i.yolov11" / "valid" / "images").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: ['ball', 'cone', 'player']\nnc: 3\n")
    d = yaml.safe_load(fix_roboflow_yaml(tmp_path / "data.yaml").read_text())
    assert d["train"].endswith("train/images") and d["val"].endswith("valid/images")
    batch = tmp_path / "batch"
    (batch / "images").mkdir(parents=True)
    (batch / "labels").mkdir()
    (batch / "data.yaml").write_text("names: [player, ball, cone]\n")
    with pytest.raises(FileNotFoundError, match="extract_frames"):
        fix_roboflow_yaml(batch / "data.yaml")


def test_confirmed_spare_ball_stays_ignored_after_long_dropout():
    sel = BallSelector(static_frames=30, gap_frames=50, memory_frames=10, confirmed_memory_frames=1000)
    spare = Detection(-1, "ball", (500, 300, 520, 320), 0.9)
    for f in range(40):                                   # seen still for 40 frames -> confirmed
        sel.select(f, [(spare, True)])
    for f in range(40, 340):                              # gone for 300 frames (> gap_frames)
        sel.select(f, [])
    assert sel.select(340, [(spare, True)]) is None       # back again: still ignored


def test_big_near_ball_jitter_is_the_same_spot():
    sel = BallSelector(static_frames=30, gap_frames=50, memory_frames=10)
    picks = []
    for f in range(60):                                   # 60 px ball near the camera, jitters +-7 px
        dy = 7 if f % 2 else -7
        picks.append(sel.select(f, [(Detection(-1, "ball", (1000, 1500 + dy, 1060, 1560 + dy), 0.9), True)]))
    assert all(p is None for p in picks[35:])


def test_evaluate_drops_balls_sitting_still_in_hindsight():
    import importlib.util
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate", scripts / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    # A resting ball detected for 3 s, but picked only in its first 1.5 s (then the live filter
    # confirms it is still); a rolling ball picked for 3 s.
    resting = [(f, 500.0, 300.0) for f in range(0, 90, 3)]
    rolling = [(f, 100.0 + 10 * f, 200.0) for f in range(90, 180, 3)]
    picks = [(f, x, y, 20.0) for f, x, y in resting[:15] + rolling]
    flags = ev.still_picks(picks, resting + rolling, fps=30)
    assert flags[:15].all() and not flags[15:].any()


def test_moving_ball_beats_a_resting_ball_not_yet_confirmed():
    sel = BallSelector(static_frames=45, gap_frames=150, memory_frames=30)
    resting = Detection(-1, "ball", (880, 990, 900, 1010), 0.9)
    for f in range(20):                                     # only the resting ball for 20 frames
        assert sel.select(f, [(resting, True)]) is resting
    for f in range(20, 40):                                 # a ball starts rolling nearby
        x = 600 + 10 * f
        rolling = Detection(-1, "ball", (x, 700, x + 20, 720), 0.4)
        assert sel.select(f, [(resting, True), (rolling, True)]) is rolling
