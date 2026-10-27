import pytest
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
    assert d["train"].replace("\\", "/").endswith("train/images") and d["val"].replace("\\", "/").endswith("valid/images")
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


def _walkers(speed_ms: float = 3.0, seconds: float = 30, fps: int = 10):
    """Two players: one jogging at speed_ms along the pitch, one standing; plus a ball."""
    import pandas as pd
    rows = []
    for k in range(int(seconds * fps)):
        t = k / fps
        rows.append(dict(frame=k, t=t, track_id=1, cls="player", team=0, x=2 + speed_ms * t % 20, y=5.0, conf=0.9))
        rows.append(dict(frame=k, t=t, track_id=2, cls="player", team=1, x=14.0, y=8.0, conf=0.9))
        if k % 2 == 0:
            rows.append(dict(frame=k, t=t, track_id=-1, cls="ball", team=-1, x=10.0, y=6.0, conf=0.5))
    return pd.DataFrame(rows)


def test_player_stats_distance_per_minute():
    from soccer_analyzer.analytics import player_stats
    s = player_stats(_walkers(3.0, 30))
    assert abs(s.loc[1, "m_per_min"] - 3.0 * 60) < 12        # 3 m/s = 180 m per minute (wraps at 20 m once)
    assert s.loc[2, "m_per_min"] < 1.0
    short = player_stats(_walkers(3.0, 6))                    # followed under 10 s: no per-minute figure
    assert short.m_per_min.isna().all()


def test_kloppy_export_round_trip():
    pytest.importorskip("kloppy")
    from soccer_analyzer.export import to_kloppy
    pitch = Pitch(28.5, 16.3)
    ds = to_kloppy(_walkers(), pitch, 10.0)
    assert len(ds.records) == 300 and len(ds.metadata.teams) == 2
    f = ds.records[10]
    xs = sorted(round(pd_.coordinates.x * 28.5, 1) for pd_ in f.players_data.values())
    assert xs[-1] == 14.0 and 0 <= xs[0] <= 20
    df = ds.to_df(engine="pandas")
    assert len(df) == 300


def test_tile_grid_covers_region_and_stays_in_frame():
    from soccer_analyzer.tracking import tile_grid
    W, H = 3840, 2160
    for roi in (None, (500, 900, 3300, 1700), (0, 0, 1000, 700), (3000, 1800, 3840, 2160)):
        tiles = tile_grid(W, H, 1280, 0.2, roi)
        assert all(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H for x1, y1, x2, y2 in tiles)
        assert all(x2 - x1 <= 1280 and y2 - y1 <= 1280 for x1, y1, x2, y2 in tiles)
        r = roi or (0, 0, W, H)
        covered = np.zeros((H // 20 + 1, W // 20 + 1), bool)
        for x1, y1, x2, y2 in tiles:
            covered[y1 // 20:-(-y2 // 20), x1 // 20:-(-x2 // 20)] = True
        assert covered[int(r[1]) // 20:int(r[3]) // 20, int(r[0]) // 20:int(r[2]) // 20].all()
    assert len(tile_grid(W, H, 1280, 0.2, (500, 900, 3300, 1700))) < len(tile_grid(W, H, 1280, 0.2))
    assert tile_grid(800, 600, 1280) == [(0, 0, 800, 600)]       # frame smaller than a tile: one tile


class _FakeResult:
    def __init__(self, rows):
        import types
        a = np.array(rows, float).reshape(-1, 6)                  # x1 y1 x2 y2 conf cls
        arrays = types.SimpleNamespace(xyxy=a[:, :4], conf=a[:, 4], cls=a[:, 5])
        self.boxes = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: arrays))


class _FakeModel:
    """Finds a ball wherever the (full-frame) frame has a bright 8x8 square, by looking at each crop."""
    def predict(self, crops, **kw):
        out = []
        for c in crops:
            ys, xs = np.nonzero(c[..., 0] > 200)
            out.append(_FakeResult([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.6, 0]]) if len(xs) else _FakeResult([]))
        return out


def test_tiled_boxes_finds_small_ball_maps_to_frame_and_drops_cut_boxes():
    from soccer_analyzer.tracking import merge_boxes, tile_grid, tiled_boxes
    frame = np.zeros((2160, 3840, 3), np.uint8)
    frame[1000:1008, 2000:2008] = 255                              # an 8 px ball far from the left edge
    tiles = tile_grid(3840, 2160, 1280, 0.2)
    found = tiled_boxes(_FakeModel(), frame, {0}, 0.1, tiles)
    boxes = [b for b, _ in merge_boxes(found)]
    assert len(boxes) == 1                                         # seen by overlapping tiles, merged to one
    assert boxes[0] == (2000.0, 1000.0, 2008.0, 1008.0)
    # an object lying across a tile's cut edge is dropped from that tile
    cut = np.zeros((2160, 3840, 3), np.uint8)
    cut[100:108, 1276:1284] = 255                                  # straddles the first tile's right edge (x=1280)
    got = [b for b, _ in tiled_boxes(_FakeModel(), cut, {0}, 0.1, [(0, 0, 1280, 1280), (1024, 0, 2304, 1280)])]
    assert (1276.0, 100.0, 1284.0, 108.0) in got and len(got) == 1  # whole in the 2nd tile only


def test_standing_player_with_jittery_foot_point_does_not_run():
    import pandas as pd
    from soccer_analyzer.analytics import player_stats
    rng = np.random.default_rng(3)
    n = 30 * 30                                              # 30 s at 30 fps
    df = pd.DataFrame(dict(frame=np.arange(n), t=np.arange(n) / 30, track_id=1, cls="player", team=0,
                           x=14 + rng.normal(0, 0.12, n), y=8 + rng.normal(0, 0.12, n)))
    s = player_stats(df)
    assert s.loc[1, "m_per_min"] < 25                         # raw frame-to-frame jitter would read ~400 m/min


def _bib_scene(seconds=20, fps=10):
    """Two real bib wearers (tracks 1, 2: votes mostly 1) and a non-bib kid (track 3) who stands next to them and
    picks up bib colour from t=5 to t=12 (votes 1 about 40% of the time)."""
    import pandas as pd
    rng = np.random.default_rng(1)
    rows = []
    for k in range(seconds * fps):
        t = k / fps
        for tid, x, p1 in ((1, 5.0, 0.8), (2, 12.0, 0.8), (3, 20.0 if not 5 <= t < 12 else 12.8, 0.4 if 5 <= t < 12 else 0.0)):
            rows.append(dict(frame=k, t=t, track_id=tid, cls="player", x=x + 0.6 * np.sin(t + tid), y=5.0 + tid,
                             vote=float(rng.random() < p1)))
    return pd.DataFrame(rows)


def test_cap_team_size_demotes_the_weakest_bib_player_while_three_look_like_bibs():
    from soccer_analyzer.stitch import cap_team_size
    df = _bib_scene()
    teams = {1: 1, 2: 1, 3: 1}                                    # track 3 got the bib label from its borrowed votes
    ids, new_teams = cap_team_size(df, teams, 2)
    team_of_row = ids.map(new_teams)
    per_frame = df.assign(team=team_of_row).query("team == 1").groupby("frame").size()
    assert (per_frame <= 2).mean() > 0.9                          # at most 2 bibs in nearly every frame
    # the demoted rows belong to track 3 (the weak one), in its own new track
    changed = df[ids != df.track_id]
    assert set(changed.track_id) == {3} and set(ids[changed.index]).isdisjoint({1, 2, 3})
    assert all(new_teams[i] == 0 for i in set(ids[changed.index]))
    # nothing to do when there were never too many
    same, _ = cap_team_size(df[df.track_id != 3], {1: 1, 2: 1}, 2)
    assert (same == df[df.track_id != 3].track_id).all()


def test_stitch_joins_a_new_id_that_starts_one_frame_before_the_old_one_ends():
    import pandas as pd
    from soccer_analyzer.stitch import stitch_tracks
    a = pd.DataFrame(dict(t=np.arange(0, 5, 0.1), track_id=1, x=10.0, y=5.0))
    b = pd.DataFrame(dict(t=np.arange(4.9, 9, 0.1), track_id=2, x=10.3, y=5.0))   # overlaps by one sample
    df = pd.concat([a, b])
    assert stitch_tracks(df)[2] == 1


def test_pipeline_bib_count_limits_bib_players_per_frame():
    pitch = Pitch(28.5, 16.3)
    lm = pitch.landmarks()
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    hom = Homography.fit(np.array([(100, 100), (900, 100), (1000, 600), (0, 600)], float),
                         np.array([lm[n] for n in names], float), pitch)
    result = {}
    for k in (None, 2):
        pipe = MatchPipeline(None, hom, teams=BibClassifier("neon_yellow"), fps=10, bib_count=k)
        d = _bib_scene()
        d["conf"], d["team"], d["bib"] = 0.9, -1, d.vote * 0.1
        xy = hom.to_image(d[["x", "y"]].to_numpy(float))
        d["px"], d["py"] = xy[:, 0], xy[:, 1]
        h = 40 + 0.1 * d.py
        d["bx1"], d["bx2"], d["by2"], d["by1"] = d.px - h * 0.2, d.px + h * 0.2, d.py, d.py - h
        pipe.rows = d.to_dict("records")
        out = pipe.dataframe()
        result[k] = out[out.team == 1].groupby("frame").size().max()
    assert result[None] == 3 and result[2] == 2


def test_joined_overlapping_tracks_leave_one_row_per_player_and_frame():
    pitch = Pitch(28.5, 16.3)
    lm = pitch.landmarks()
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    hom = Homography.fit(np.array([(100, 100), (900, 100), (1000, 600), (0, 600)], float),
                         np.array([lm[n] for n in names], float), pitch)
    pipe = MatchPipeline(None, hom, teams=BibClassifier("neon_yellow"), fps=10)
    rows = []
    for tid, f0, f1 in ((1, 0, 50), (2, 49, 100)):                # id 2 starts one frame before id 1 ends
        for f in range(f0, f1):
            px, py = hom.to_image(np.array([[10.0, 8.0]]))[0]
            rows.append(dict(frame=f, t=f / 10, track_id=tid, cls="player", team=-1, conf=0.9 - 0.1 * (tid == 2),
                             px=px, py=py, x=10.0, y=8.0, bx1=px - 8, by1=py - 40, bx2=px + 8, by2=py,
                             bib=np.nan, vote=np.nan))
    pipe.rows = rows
    out = pipe.dataframe()
    assert out.track_id.nunique() == 1
    assert not out.duplicated(["frame", "track_id"]).any()


def test_team_hull_and_voronoi_control():
    import pandas as pd
    from soccer_analyzer.space import control_shares, hull, polygon_area, space_table
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)       # centre point is inside the hull
    assert abs(polygon_area(hull(square)) - 100) < 1e-6
    assert hull(np.array([[0, 0], [1, 1], [2, 2]], float)) is None                # all on a line: no polygon
    assert hull(np.array([[0, 0], [1, 1]], float)) is None
    # two players split a 20 x 10 pitch down the middle
    s = control_shares(np.array([[5, 5], [15, 5]], float), np.array([0, 1]), 20, 10, cell=0.5)
    assert abs(s[0] - 0.5) < 0.02 and abs(s[1] - 0.5) < 0.02
    # a table over frames: team 0 has 3 players in a corner (a triangle of 8 m²), team 1 has 3 spread out
    rows = []
    for f in range(3):
        for i, (x, y) in enumerate([(2, 2), (6, 2), (4, 6)]):
            rows.append(dict(frame=f, t=f / 10, track_id=i, cls="player", team=0, x=x, y=y))
        for i, (x, y) in enumerate([(14, 2), (18, 2), (16, 9)]):
            rows.append(dict(frame=f, t=f / 10, track_id=10 + i, cls="player", team=1, x=x, y=y))
    tab = space_table(pd.DataFrame(rows), 20, 10)
    assert len(tab) == 3 and abs(tab.hull0_m2.iloc[0] - 8) < 1e-6 and tab.hull1_m2.iloc[0] > 0
    assert abs(tab.control0.iloc[0] + tab.control1.iloc[0] - 1) < 1e-6


def test_calibration_clicks_can_skip_jump_and_undo():
    from soccer_analyzer.calib_ui import CalibState
    s = CalibState(["a1", "b2", "c3", "d4", "e5"])
    s.click((1, 1))                                   # a1 placed, now b2
    assert s.current == "b2"
    s.skip()                                          # b2 skipped, now c3
    assert s.current == "c3" and "b2" in s.skipped
    s.jump(s.key_of("e5"))                            # choose e5 directly
    s.click((5, 5))                                   # e5 placed; next unresolved is c3 (wraps round)
    assert s.current == "c3"
    s.click((3, 3))
    assert s.current == "d4"
    s.undo()                                          # takes back the c3 click
    assert s.current == "c3" and "c3" not in s.points
    s.jump(s.key_of("a1"))
    s.click((9, 9))                                   # re-clicking replaces a placed point
    assert s.points["a1"] == (9.0, 9.0) and len(s.points) == 2
    assert s.placed == ["a1", "e5"]
    s.undo()
    s.undo()                                          # undo the a1 click, then the jump has no history: stays safe
    s.undo()
    s.undo()
    s.undo()
    s.undo()


def test_off_pitch_tracks_reasons_and_option_to_keep_players_standing_on_the_line():
    import pandas as pd
    from soccer_analyzer.stitch import off_pitch_tracks
    t = np.arange(0, 20, 0.1)
    rows = []
    for tid, x, y in ((1, 14.0, 8.0), (2, 14.0, 16.0), (3, 14.0, -2.0)):      # in play / waiting on the near line / outside
        rows.append(pd.DataFrame(dict(t=t, track_id=tid, x=x + 0.6 * np.sin(t), y=y)))
    df = pd.concat(rows)
    why = {}
    assert off_pitch_tracks(df, 28.5, 16.3, reasons=why) == {2, 3}
    assert why["off_pitch"] == {3} and why["standing_on_line"] == {2}
    assert not why["ignore_zone"] and not why["static"] and not why["seated"]
    assert off_pitch_tracks(df, 28.5, 16.3, drop_standing_on_line=False) == {3}


def test_a_small_player_is_not_taken_for_an_occluded_one():
    from soccer_analyzer.occlusion import fix_occluded_feet
    import pandas as pd
    pitch = Pitch(length=40, width=20)
    pts = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], np.float32)
    hom = Homography.fit(pts * 10, pts, pitch)
    rng = np.random.default_rng(2)
    rows = []
    for tid, size in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 0.62)):        # track 4 is a small kid: 62% of the usual height
        for i in range(120):
            top = rng.uniform(0, 150)
            h = (60 + 0.5 * top) * size
            rows.append(dict(track_id=tid, px=200.0, by1=top, by2=top + h, bx1=180.0, bx2=180 + 0.4 * h))
    hidden = dict(track_id=1, px=200.0, by1=100.0, by2=150.0, bx1=180.0, bx2=220.0)     # a real hidden-legs frame
    df = pd.DataFrame(rows + [hidden])
    df["py"] = df.by2
    df[["x", "y"]] = hom.to_pitch(df[["px", "py"]].to_numpy())
    out = fix_occluded_feet(df, hom)
    assert out[out.track_id == 4].occluded.mean() < 0.02        # the small kid keeps his own feet
    assert out.iloc[-1].occluded and abs(out.iloc[-1].py - 210) < 12


def test_non_player_tracks_zone_static_and_seated():
    import pandas as pd
    from soccer_analyzer.stitch import off_pitch_tracks
    t = np.arange(0, 60, 0.1)
    rng = np.random.default_rng(0)

    def track(tid, x, y, sway, aspect=0.4):
        h = 200.0
        return pd.DataFrame(dict(t=t, track_id=tid, x=x + sway * np.sin(t * 0.7) + rng.normal(0, 0.05, len(t)),
                                 y=y + sway * np.cos(t * 0.5), bx1=0.0, bx2=aspect * h, by1=0.0, by2=h))
    df = pd.concat([
        track(1, 14, 8, 3.0),                  # a player, moving about
        track(2, 28.3, 8.0, 0.2),              # the coach beside the goal line, hardly moving
        track(3, 5.0, 8.0, 0.0),               # a post: not moving at all
        track(4, 20.0, 5.0, 3.0, aspect=1.1),  # seated: box as wide as tall
        track(5, 10.0, 4.0, 3.0),              # a player inside a zone someone listed
    ])
    why = {}
    assert off_pitch_tracks(df, 28.5, 16.3, reasons=why) == {3, 4}                       # post + seated by default
    assert off_pitch_tracks(df, 28.5, 16.3, static_s=30) == {2, 3, 4}                    # + the coach who stands still
    assert off_pitch_tracks(df, 28.5, 16.3, static_s=30, ignore_zones=[[8, 2, 12, 6]]) == {2, 3, 4, 5}


def test_overlap_duplicates_keeps_the_more_confident_of_two_boxes_on_one_player():
    from soccer_analyzer.tracking import overlap_duplicates
    boxes = np.array([[100, 100, 140, 220], [102, 110, 138, 215], [300, 100, 340, 220], [120, 100, 160, 220]], float)
    conf = np.array([0.5, 0.8, 0.6, 0.4])
    keep = overlap_duplicates(boxes, conf, 0.45)
    assert list(keep) == [1, 2, 3]            # box 0 is a duplicate of box 1; box 3 only half overlaps box 1
    torso = np.array([[100, 100, 140, 220], [104, 110, 136, 200]], float)      # a smaller box inside, about the same height
    assert list(overlap_duplicates(torso, np.array([0.6, 0.5]))) == [0]
    head = np.array([[100, 100, 140, 220], [105, 100, 135, 130]], float)       # a head box on the same body is another matter
    assert list(overlap_duplicates(head, np.array([0.6, 0.5]))) == [0, 1]


def test_ball_that_teleports_is_held_back_until_it_keeps_showing_up():
    pipe = MatchPipeline.__new__(MatchPipeline)
    pipe.fps, pipe.ball_max_speed, pipe._ball_last, pipe._challenger = 30.0, 18.0, None, None
    assert pipe._ball_plausible(0, 5.0, 5.0)                       # first ball: accepted
    assert pipe._ball_plausible(3, 6.0, 5.0)                       # moved 1 m in 0.1 s: fine
    assert not pipe._ball_plausible(6, 25.0, 12.0)                 # a boot 20 m away: held back
    assert pipe._ball_plausible(7, 6.5, 5.0)                       # the real ball is back: accepted, challenger forgotten
    accepted = [pipe._ball_plausible(10 + k, 25.0, 12.0) for k in range(20)]
    assert not accepted[0] and accepted[-1]                        # a ball that stays 20 m away becomes the ball


def test_local_config_is_laid_over_the_main_one(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze import load_config
    (tmp_path / "config.yaml").write_text("model:\n  weights: a.pt\n  imgsz: 1280\nteams:\n  bib_count: 2\n")
    assert load_config(tmp_path / "config.yaml")["model"]["weights"] == "a.pt"
    (tmp_path / "config.local.yaml").write_text("model:\n  weights: b.pt\nteams:\n  bib_count: 4\n")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg["model"] == {"weights": "b.pt", "imgsz": 1280} and cfg["teams"]["bib_count"] == 4


def test_pitch_without_goal_area_keeps_letters_and_skips_the_missing_landmarks():
    from soccer_analyzer.calib_ui import CalibState
    pitch = Pitch(28.5, 16.3, circle_radius=3.0, box_depth=5.0, box_width=9.1, goal_area_depth=0.0, penalty_spot=0.0)
    gone = pitch.absent_landmarks()
    assert "left_6yd_far_corner" in gone and "right_penalty_spot" in gone and "left_box_far_corner" not in gone
    s = CalibState(list(pitch.landmarks()), gone)
    s.jump(s.key_of("left_6yd_far_corner"))
    assert s.current == "far_left_corner"                              # a missing landmark can't be chosen
    names = list(pitch.landmarks())
    assert s.key_of("right_box_far_corner") == "t"                     # letters don't shift
    assert len(pitch.marking_polylines()) == 3 + 2                     # outline, halfway line, circle, 2 boxes (no goal areas)
