# Phase 2: train a detector on your own footage

The stock detector was trained on everyday photos, not on night-time kids' football
seen from a sideline tripod. It misses the ball about 3 times out of 4, calls cones
"balls", and merges players who overlap. A model trained on a few hundred of **your**
frames fixes most of that.

Steps: **extract frames → fix boxes in Roboflow → export → train → compare → switch.**

## 1. Extract frames (PC, 5–10 min)

```bat
.venv\Scripts\activate.bat
python scripts\extract_frames.py data\videos\training.mp4 --calib runs\calib_small.json --out runs\label_batch1
```

List several videos to mix sessions (each gets frames in proportion to its length):

```bat
python scripts\extract_frames.py data\videos\training.mp4 data\videos\match1.mp4 data\videos\day_training.mp4 --n 300
```

This creates `runs\label_batch1.zip`. The images are frames chosen so that about half are
the hard moments (players bunched up or colliding, ball near someone's feet, cones) and the
rest are spread at random. Each one already has boxes drawn by the current detector.
Frames are at least 1 s apart, so a 78-second video gives about 70 frames. More sessions
give more variety, and variety is what makes the model good.

## 2. Roboflow: create the project and upload (once per batch)

1. Go to roboflow.com and sign up (the free plan is enough).
2. **Create New Project**: choose **Object Detection** and name it e.g. `rsm-soccer`.
3. **Upload Data**: drag `runs\label_batch1.zip` in. Roboflow reads the boxes from the zip
   (YOLO format). Check that the classes shown are **player, ball, cone**, then save or
   continue the upload.

## 3. Fix the boxes (the important part, 1–2 h for ~300 images)

Open each image in **Annotate**. The boxes are only a first guess. Make every image follow
these rules:

| Class | Draw a box around | Don't box |
|---|---|---|
| `player` | every person **on the pitch or standing on its lines**: players, goalkeepers, a coach on the line | people on the bench, in the stands, on other pitches |
| `ball` | **every ball**, including partly hidden ones and balls lying still at the side. **Add every ball the detector missed.** This matters most. | |
| `cone` | every training cone or disc | |

- Two players under one box: delete it and draw one box per player.
- Partly hidden player: box only the visible part.
- A cone or anything else labelled `ball`: change its class or delete it.
- Keep boxes tight around the object.
- Shortcuts: press the class number, drag to draw, `Delete` removes the selected box,
  and the arrow keys go to the next or previous image.

Be consistent: the model copies whatever you do. If you skip some balls, it learns that
skipping balls is fine.

## 4. Generate a version and export

1. **Generate** → **Create New Version**.
2. **Train/Test Split**: about 80% train, 15% valid, 5% test.
3. **Preprocessing**: keep *Auto-Orient* and **remove *Resize*** if it's there. Resizing to
   640 px shrinks the ball to almost nothing.
4. **Augmentation**: none. The training script adds its own.
5. **Export Dataset** → format **YOLOv11** → **Download zip to computer**.
6. Unzip it into `analyzer\data\datasets\soccer-v1\`. That folder must contain `data.yaml`,
   `train\` and `valid\`.

## 5. Train (PC, about 1–2 hours on the RTX 3060 Ti)

```bat
python scripts\train.py data\datasets\soccer-v1\data.yaml
```

- The default is `yolo11s.pt` at 1600 px with 2 images per step, which fits in 8 GB (1920 px
  needs 4 GB per image and doesn't fit 2). For the bigger `--model yolo11m.pt`, add `--imgsz 1280`.
- If you get `CUDA out of memory`, close other GPU programs or lower to `--imgsz 1280`.
- Small datasets train with the pretrained backbone frozen (`--freeze 10`) and a low learning
  rate (AdamW, 0.001). Training the whole network on 30 images diverged after 8 epochs: the losses
  shot up and mAP fell to 0. Once you have ~500+ images, try `--freeze 0`.
- If `results.png` still shows the losses suddenly jumping up, add `--no-amp --imgsz 1280`
  (full precision, more stable, uses more memory).
- Training stops early once it stops improving. The result is
  `runs\train\soccer-v1\weights\best.pt`, and the script prints what to put in `config.yaml`.
- `runs\train\soccer-v1\results.png` shows the learning curves. In
  `runs\train\soccer-v1\val_batch0_pred.jpg` you can see its predictions on unseen frames.

## 6. Compare with the stock detector

```bat
python scripts\evaluate.py data\videos\training.mp4 --models yolo11m.pt runs\train\soccer-v1\weights\best.pt --imgsz 1280 1600 --calib runs\calib_small.json
```

It runs each model over the video with the same ball filter as the analyzer and prints:
- **ball in play**: % of frames where a moving ball on the pitch was found. Spare balls, the coach's
  ball and cones don't count. This is the number to improve.
- **near a player**: of those, how many were within 2 m of a player (a check that it's the play ball).
- **players/frame**: people on the pitch per frame. Steadier (small ±) is better.
- **cones as ball**: lower is better.

The table is also saved to `runs\evaluate_<video>.txt`. Ideally test on a video that was **not**
used for labelling.

## 7. Switch the analyzer to your model

In `config.yaml`:

```yaml
  weights: runs/train/soccer-v1/weights/best.pt
  imgsz: 1600
```

Run `analyze.py` as usual. Class ids don't need changing: the analyzer finds `player` and
`ball` by name.

## Next rounds

After each new session: extract a new batch from it (pre-labelled by **your** model now, so
there's less to fix), upload it to the same Roboflow project, fix it, generate a new
version, and train again as `soccer-v2`. Two or three rounds with day and night, matches
and training sessions, and different kits make a big difference.
