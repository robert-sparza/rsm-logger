"""Building a labelling batch for training our own detector (Phase 2).

We sample frames from match/training videos, pre-draw boxes with the current
detector, and write them in YOLO format so Roboflow imports them as editable
annotations. The person labelling only fixes boxes instead of drawing them all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# Order = class id in the label files. Roboflow may re-sort them on export;
# the analyzer finds classes by name, so that is fine.
CLASSES = ["player", "ball", "cone"]
PLAYER, BALL, CONE = range(3)


def frame_score(player_boxes, ball_boxes, cone_boxes) -> float:
    """How useful a frame is for teaching the detector what it gets wrong.

    Crowding (players overlapping or close: where boxes merge and IDs swap) and a
    ball near a player's feet (the ball the stock model misses) count most;
    cones count a little (so the model learns they are not balls).
    """
    score = 0.0
    boxes = [np.asarray(b, float) for b in player_boxes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            gap_x = max(a[0], b[0]) - min(a[2], b[2])            # < 0 means overlapping horizontally
            width = max(a[2] - a[0], b[2] - b[0])
            if gap_x < 0.5 * width and abs(a[3] - b[3]) < 0.5 * (a[3] - a[1]):
                score += 1.0
    for ball in ball_boxes:
        bx, by = (ball[0] + ball[2]) / 2, ball[3]
        for p in boxes:
            h = p[3] - p[1]
            if p[0] - 0.5 * h <= bx <= p[2] + 0.5 * h and abs(by - p[3]) < 0.3 * h:
                score += 2.0
                break
    return score + 0.25 * min(len(cone_boxes), 4)


def select_frames(candidates, n: int, min_gap_s: float = 1.0, top_share: float = 0.5, seed: int = 0):
    """Pick n candidates: about half the highest-scoring ones, the rest spread at random.

    candidates: list of (key, t_seconds, score); key is returned for the chosen ones.
    Chosen frames are at least `min_gap_s` apart (within one video) so we don't
    label near-identical images.
    """
    rng = np.random.default_rng(seed)
    chosen: list = []
    times: list[float] = []

    def take(c) -> bool:
        if len(chosen) >= n or any(abs(c[1] - t) < min_gap_s for t in times):
            return False
        chosen.append(c)
        times.append(c[1])
        return True

    by_score = sorted(candidates, key=lambda c: -c[2])
    n_top = int(round(n * top_share))
    for c in by_score:
        if len(chosen) >= n_top:
            break
        if c[2] > 0:
            take(c)
    rest = [c for c in candidates if c not in chosen]
    for i in rng.permutation(len(rest)):
        take(rest[i])
    return sorted(c[0] for c in chosen)


def yolo_lines(labelled_boxes, width: int, height: int) -> list[str]:
    """labelled_boxes: (class_id, (x1, y1, x2, y2)) in pixels -> YOLO txt lines."""
    lines = []
    for cls, (x1, y1, x2, y2) in labelled_boxes:
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def write_data_yaml(out_dir: Path) -> Path:
    path = out_dir / "data.yaml"
    path.write_text(yaml.safe_dump({"names": CLASSES, "nc": len(CLASSES)}, sort_keys=False))
    return path


def fix_roboflow_yaml(data_yaml: Path) -> Path:
    """Roboflow's export points to '../train/images' etc., which only works from one
    particular folder. Write data_fixed.yaml next to it with absolute paths."""
    data_yaml = Path(data_yaml).resolve()
    root = data_yaml.parent
    d = yaml.safe_load(data_yaml.read_text())

    def find(names):
        """<root>/<name>/images, also one or two folders deeper (extra layer from unzipping)."""
        for depth in ("", "*/", "*/*/"):
            for name in names:
                hits = sorted(root.glob(f"{depth}{name}/images"))
                if hits:
                    return hits[0]
        return None

    for split, names in (("train", ("train",)), ("val", ("valid", "val")), ("test", ("test",))):
        images = find(names)
        if images is not None:
            d[split] = str(images)
        elif split in d and split != "train":
            d.pop(split)
    d.pop("path", None)
    if "train" not in d or not Path(d["train"]).is_dir():
        if (root / "images").is_dir() and (root / "labels").is_dir():
            raise FileNotFoundError(
                f"{root} looks like the label batch made by extract_frames.py (images/ + labels/), "
                "not a Roboflow export. Export from Roboflow (Versions > Export Dataset > YOLOv11), "
                "unzip that into its own folder and pass its data.yaml.")
        found = ", ".join(sorted(p.name for p in root.iterdir())) or "nothing"
        raise FileNotFoundError(f"No train/images folder found under {root} (it contains: {found}). "
                                "Unzip the Roboflow YOLOv11 export there and pass its data.yaml.")
    d.setdefault("val", d["train"])
    out = root / "data_fixed.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return out
