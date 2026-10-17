# Soccer Match Analyzer

Turns footage from a fixed sideline camera (XbotGo Falcon on a 3–3.5 m tripod) into
player tracks, team heatmaps, ball paths, possession and pass maps. Runs on the
Omen PC (RTX 3060). It also has a live mode that reads the camera's RTMP stream.

```
video / RTMP ──► YOLO detection ──► ByteTrack IDs ──► team split (jersey colour)
                                                     │
                           calibration (homography) ─┴─► pitch metres ──► analytics + plots
```

## 1. Setup on the PC (Windows, one time)

1. Install **Python 3.11** (python.org, tick "Add to PATH"), **Git**, and an up-to-date **NVIDIA driver**.
2. Get the code:
   ```powershell
   git clone https://github.com/robert-sparza/rsm-logger.git
   cd rsm-logger\analyzer
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install PyTorch **with CUDA** first, then the rest:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The last line must print `True NVIDIA GeForce RTX 3060`. If it prints `False`, analysis
   still works on the CPU, just ~20× slower.
4. Run the tests: `pytest -q tests`

## 2. Analyze a recorded match (start here)

Copy a match video into `data/videos/` (this folder is git-ignored), then:

```powershell
# a) Calibrate once per camera position: click pitch landmarks on a frame
python scripts\calibrate.py data\videos\match.mp4 --out runs\calib.json --length 100 --width 64

# b) Quick 2-minute test run
python scripts\analyze.py data\videos\match.mp4 --calib runs\calib.json --out runs\test --max-seconds 120

# c) Full match (use --stride 2 to go twice as fast)
python scripts\analyze.py data\videos\match.mp4 --calib runs\calib.json --out runs\match1
```

Measure your actual pitch and pass `--length/--width`. Youth and amateur pitches are rarely 105×68 m.
`calibrate.py` saves a `calib.check.jpg` with the pitch lines drawn back onto the frame. If the
red lines don't sit on the real lines, redo the calibration with more points.

Output in `runs/match1/`: `annotated.mp4`, `tracks.csv`, `player_stats.csv`,
`heatmap_team0/1.png`, `ball_path.png`, `passes_team0/1.png`, `passes.csv`, `summary.json`.

## 3. Live mode (RTMP from the camera)

1. Download **MediaMTX** (github.com/bluenviron/mediamtx/releases, the `windows_amd64.zip`),
   unzip, and run `mediamtx.exe`. It listens for RTMP on port 1935.
2. Allow port 1935 in Windows Firewall. Find the PC's local IP with `ipconfig`.
3. In the XbotGo app choose Live → Custom RTMP. Set the URL to `rtmp://<PC-IP>:1935/live` and the key to `falcon`.
   The phone and the PC must be on the same network. At a field, a phone hotspot that both join is enough.
4. `python scripts\live.py rtmp://localhost:1935/live/falcon --calib runs\calib.json --record`

`--record` saves the raw stream too, so you can re-run the full offline analysis later.
You can open the stream in VLC at the same URL to check it before running the analyzer.

## 4. Camera tips (these matter more than the code)

- **Turn off auto-tracking and zoom while recording for analysis.** The pitch mapping assumes
  the camera doesn't move. If the Falcon pans, the homography is wrong for every frame after that.
  Lock it wide so it covers the whole pitch. Phase 3 removes this limit.
- Put the tripod at the **halfway line**, as high as it goes, a few metres back from the touchline.
  At 3.5 m the far side of the pitch is compressed and players overlap, so expect weaker tracking
  over there. The higher the camera, the better everything works.
- Record at the highest resolution the phone allows (4K if possible). Players on the far side are only a few pixels tall.
- Keep the sun behind the camera, and make sure the two teams wear clearly different colours (bibs help a lot).

## 5. Roadmap

| Phase | Goal | What to build |
|---|---|---|
| **1 – done here** | Offline analysis with stock models | Detection and tracking (YOLO11 + ByteTrack), team split by jersey colour, manual calibration, heatmaps, distance and speed, ball path, possession, passes |
| **2** | Better detection | Fine-tune YOLO on football data: Roboflow Universe "football-players-detection" plus ~300 labelled frames from **your** camera. Separate classes for ball, goalkeeper, player and referee. The ball is the weakest part of Phase 1. |
| **3** | Moving camera | Train a pitch-keypoint model (Roboflow sports `football-field-detection`) to re-estimate the homography every frame. This lets you keep the Falcon's auto-tracking on. |
| **4** | Tactics | Formation detection (cluster average positions), pressing intensity, line height, Voronoi space control, pass networks, sequences that lead to shots |
| **5** | Live dashboard | Web UI (FastAPI + WebSocket) showing the minimap, live stats and a timeline. Feed events into the RSM Match Logger (`../index.html`) |
| **6** | Players | Jersey-number OCR or manual tagging so stats belong to named players across matches |

Useful open-source references: `roboflow/sports` (football pipeline: pitch keypoints, team
classification with SigLIP embeddings, radar view), `ultralytics` (YOLO11 detection and
tracking), `supervision` (annotators, ByteTrack), `mplsoccer` (pitch plots), `kloppy` (tracking-data formats).

## 6. Working with coding agents

Each phase is a self-contained task you can give to a Claude Code session on the PC
(`claude` in this folder), or to several sessions in parallel on separate branches:

- *Agent A – data:* "Extract 300 frames from runs/*/raw.mp4, set up a Roboflow/CVAT labelling
  export, and fine-tune yolo11m on it (Phase 2)."
- *Agent B – camera motion:* "Implement per-frame pitch-keypoint homography (Phase 3) behind the same `Homography` interface."
- *Agent C – analytics:* "Add formation detection and pass networks to analytics.py with tests (Phase 4)."
- *Agent D – UI:* "Build a FastAPI + WebSocket dashboard for live.py (Phase 5)."

Code layout:

```
soccer_analyzer/
  tracking.py   YOLO + ByteTrack wrapper, frame reader (files and rtmp://)
  teams.py      jersey-colour team classifier
  pitch.py      pitch landmarks, homography
  pipeline.py   per-frame glue used by analyze.py and live.py
  analytics.py  stats: distance/speed, heatmaps, ball path, possession, passes, team shape
  report.py     writes CSVs and plots
  viz.py        overlays, minimap, matplotlib pitch plots
scripts/        calibrate.py, analyze.py, live.py
tests/          synthetic-data tests (no GPU needed)
```

## Known limitations (Phase 1)

- The stock COCO "sports ball" class often misses a small, fast or far ball. Passes and possession depend on it, so treat them as rough until Phase 2.
- Track IDs can swap when players cross or overlap, and one player can end up with several IDs. Per-team totals are more reliable than per-player stats for now.
- Goalkeepers and referees get pushed into one of the two teams.
- A ball in the air is projected as if it were on the ground.
