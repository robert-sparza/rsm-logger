# RSM Soccer Coach: analyzer (notes for a Claude session working on this PC)

Goal: analyze night-time youth football/training video from one fixed camera (XbotGo or phone on a 3-3.5 m tripod)
and give a coach: player tracks, team split (shirt colour or bibs), pitch maps, heat maps, team polygons / space control,
ball, passes, distance per minute. Everything runs locally on this PC (RTX 3060 Ti 8 GB, Windows, venv at `.venv`).
Read `README.md`, `PHASE2.md` and `PRODUCT.md` first. Spanish-speaking club, players are minors: video stays on this PC.

## Run it
```
.venv\Scripts\activate.bat
python scripts\calibrate.py data\videos\X.mp4 --out runs\calib_4v4.json --at 5 --length 28.5 --width 16.3 --circle-radius 3.0 --box-depth 5.0 --box-width 9.1 --goal-area-depth 0 --penalty-spot 0
python scripts\analyze.py data\videos\X.mp4 --calib runs\calib_4v4.json --out runs\<new folder> --bib neon_yellow
python scripts\run_quality.py runs\<folder> --players 8 --bibs 4
python -m pytest tests -q
```
Settings: `config.yaml` (comes with the code) and `config.local.yaml` (the user's own, never overwritten; layer on top).
The court is 28.5 x 16.3 m with one box per side 5.0 x 9.1 m and a centre circle of radius 3.0 m.
Calibration is only valid for one camera position; recalibrate when the tripod moves. Use a new `--out` folder per run.

## How to tell if a change helped (no ground truth needed)
`quality.txt` / `scripts/run_quality.py` score a run from tracks.csv: players per frame vs expected, tracks per expected
player (1.0 = no broken IDs), frames with exactly the right number of bib players, frames with two boxes on one player,
ball jumps per minute. Compare runs on the SAME video. Best known run on the 112 s 4 v 4 clip: 8.5 players/frame,
2.8 tracks per player, 71% frames with exactly 4 bib players, 0.3% overlapping boxes, 0 ball jumps.

## Known problems, most valuable first
1. IDs break: about 22 tracks for 8 players in 112 s. Try BoT-SORT with ReID (`trackers/botsort_moving_camera.yaml` has the
   shape of the config; `with_reid`), or link tracklets by position + bib/team + known player count.
2. The detector was trained on one 78 s session. More labelled sessions (Roboflow, see PHASE2.md) is the biggest lever:
   night 4 v 4 clips with overlapping players, coach and spectators as their own class, several balls.
3. Ball in play is found in about half the frames; try `model.tile_size: 1280` and compare with `scripts/evaluate.py --tile`.
4. Passes / turnovers are built on the ball and are unreliable until 3 improves.
5. Team polygons and space control (soccer_analyzer/space.py) are new: check them on real footage.

## Rules
- Never delete `data/`, `runs/` or the user's videos. Never overwrite a finished `runs\<name>` folder.
- Training uses the GPU for 1-2 hours (`scripts/train.py`, see PHASE2.md); ask before starting it.
- Keep tests passing (`python -m pytest tests -q`). Commit to the branch `claude/soccer-game-analyzer-cfe1q0`; no pull request unless asked.
- Model weights: the good one is `runs/train/soccer-v1b/weights/best.pt` at imgsz 1600. `soccer-v1` is a failed run: do not use it.
