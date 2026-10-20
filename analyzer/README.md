# Soccer Match Analyzer

Turns footage from a fixed sideline camera (XbotGo Falcon on a 3–3.5 m tripod) into
player tracks, team heatmaps, ball paths, possession and pass maps. Runs on the
Omen PC (RTX 3060 Ti, 8 GB). It also has a live mode that reads the camera's RTMP stream.

```
video / RTMP ──► YOLO detection ──► ByteTrack IDs ──► team split (jersey colour)
                                                     │
                           calibration (homography) ─┴─► pitch metres ──► analytics + plots
```

## 1. Setup on the PC (Windows, one time)

Tested target: Omen 25L, Ryzen 7 5700G, 16 GB RAM, **RTX 3060 Ti (8 GB)**.

1. Install these three things:
   - **Python 3.12** from python.org. In the installer, tick **"Add python.exe to PATH"**.
     Newer Python versions may not have PyTorch builds yet.
   - **Git** from git-scm.com.
   - The latest **NVIDIA driver** (GeForce Experience or nvidia.com).
2. Get the code. Open "Command Prompt" and run:
   ```bat
   git clone -b claude/soccer-game-analyzer-cfe1q0 https://github.com/robert-sparza/rsm-logger.git
   cd rsm-logger\analyzer
   ```
3. **Double-click `setup_windows.bat`**, or type `setup_windows.bat` in that window.
   It creates `.venv`, installs PyTorch with CUDA (a download of about 3 GB), installs the
   requirements, checks the GPU and runs the tests. At the end it must print
   `GPU OK: NVIDIA GeForce RTX 3060 Ti`. If it shows an error, send the error lines to Claude.
4. Every time you open a new terminal, activate the environment first: `.venv\Scripts\activate.bat`
   (in PowerShell: `.venv\Scripts\Activate.ps1`; if PowerShell blocks scripts, use Command Prompt).

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

**Calibration tips.** "far" means the touchline away from the camera, and "near" means the one closest to the camera.
Five clicks are enough: far-left corner, far-right corner, the far end of the halfway line, the near end of the
halfway line, and the centre spot. Press `s` to skip corners that are outside the frame. Only click circle or box
points if you pass their real sizes (`--circle-radius`, `--box-depth`, `--box-width`). The script prints the error
of each click; anything over about 0.75 m is a wrong click or a wrong size.

Example: the small training pitch (28.5 × 16.3 m, circle radius 3.0 m, box 5.0 × 9.1 m):
`python scripts\calibrate.py data\videos\training.mp4 --out runs\calib_small.json --length 28.5 --width 16.3 --circle-radius 3.0 --box-depth 5.0 --box-width 9.1`

**Training with bibs.** Add `--bib neon_yellow` (or orange, pink, blue, red) to `analyze.py` or `live.py`.
Players in bibs become team 1 and everyone else team 0. A match with two kits needs no option.

Measure your actual pitch and pass `--length/--width`. Youth and amateur pitches are rarely 105×68 m.
`calibrate.py` saves a `calib.check.jpg` with the pitch lines drawn back onto the frame. If the
red lines don't sit on the real lines, redo the calibration with more points.

Output in `runs/match1/`: `annotated.mp4`, `tracks.csv`, `player_stats.csv`,
`heatmap_team0/1.png`, `ball_path.png`, `passes_team0/1.png`, `passes.csv`, `summary.json`.

## 3. Live mode (RTMP from the camera)

Any phone app that can stream RTMP works: the XbotGo app, or the free **Larix Broadcaster**
(Android and iPhone) for a plain phone on a tripod. Recorded phone videos also work with `analyze.py`.

1. Download **MediaMTX** (github.com/bluenviron/mediamtx/releases, the `windows_amd64.zip`),
   unzip, and run `mediamtx.exe`. It listens for RTMP on port 1935.
2. Allow port 1935 in Windows Firewall. Find the PC's local IP with `ipconfig`.
3. In the XbotGo app choose Live → Custom RTMP. Set the URL to `rtmp://<PC-IP>:1935/live` and the key to `falcon`.
   The phone and the PC must be on the same network. At a field, a phone hotspot that both join is enough.
4. `python scripts\live.py rtmp://localhost:1935/live/falcon --calib runs\calib.json --record`

`--record` saves every frame of the raw stream, so you can re-run the full offline analysis later.
The live view always analyzes the newest frame. The PC analyzes about 16 frames per second while the phone
sends 30, so the view skips frames rather than falling behind the game. `live.py` waits up to
5 minutes for the phone to start streaming, and reconnects if the stream drops for up to 30 seconds.
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

- The stock COCO "sports ball" class often misses a small, fast or far ball, and it detects cones as balls.
  Candidates that don't move for 3 seconds, or that are off the pitch, are ignored, but passes and possession are still rough until Phase 2.
- Track IDs can swap when players cross or overlap. After the run, broken tracks are joined: a track that
  starts within 2 s of another ending, at a spot the player could have reached, is the same player (never
  across teams). `annotated.mp4` is drawn after that, with the final IDs (1, 2, 3...) and teams. The live view
  shows the tracker's raw IDs.
- People whose usual position is outside the lines (coach, bench) are dropped after calibration.
- Goalkeepers and referees get pushed into one of the two teams.
- Anyone more than 1.5 m outside the lines is ignored (`analysis.pitch_margin` in config.yaml). This only works after calibration.
- A ball in the air is projected as if it were on the ground.
