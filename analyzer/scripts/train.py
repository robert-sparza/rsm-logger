"""Train our own detector on the dataset exported from Roboflow (Phase 2).

Usage:
    python scripts/train.py data/datasets/soccer-v1/data.yaml
    python scripts/train.py data/datasets/soccer-v1/data.yaml --model yolo11m.pt --epochs 150

Export the dataset from Roboflow in "YOLOv11" format and unzip it into
data/datasets/<name>/ (the folder that contains data.yaml, train/, valid/).
The result is runs/train/<name>/weights/best.pt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.dataset import fix_roboflow_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data.yaml from the Roboflow export")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="starting weights: yolo11s.pt (fits 8 GB) or yolo11m.pt (better, needs --imgsz 1280)")
    # Measured on an RTX 3060 Ti (8 GB) with yolo11s: 1920 px uses 4 GB per image, so 2 images
    # don't fit and Ultralytics' automatic batch search crashes. 1600 px x 2 images: ~5.5 GB (estimated).
    ap.add_argument("--imgsz", type=int, default=1600,
                    help="training resolution; high, because the ball is only a few pixels wide")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2,
                    help="images per step; 2 fits 8 GB at 1600 px. Out of memory: lower --imgsz")
    # Small datasets (tens to a few hundred images): the first run on 30 images with the
    # defaults (whole network, auto optimizer) learned for ~8 epochs and then diverged (losses
    # x10, NaN in validation). Freezing the pretrained backbone and a low AdamW learning rate
    # is the usual stable recipe; unfreeze (--freeze 0) once you have ~500+ images.
    ap.add_argument("--freeze", type=int, default=10,
                    help="freeze the first N layers (10 = the backbone); 0 trains everything")
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    ap.add_argument("--no-amp", action="store_true",
                    help="train in full precision (more stable, ~1.5-2x memory; use with --imgsz 1280)")
    ap.add_argument("--name", help="run name (default: the dataset folder name)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    data = fix_roboflow_yaml(Path(args.data))
    name = args.name or Path(args.data).resolve().parent.name
    model = YOLO(args.model)
    model.train(data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                patience=30, device=args.device, project=str(ROOT / "runs" / "train"), name=name,
                exist_ok=True, workers=4, close_mosaic=10, plots=True,
                freeze=args.freeze or None, optimizer=args.optimizer, lr0=args.lr0, cos_lr=True,
                warmup_epochs=3, amp=not args.no_amp)

    best = ROOT / "runs" / "train" / name / "weights" / "best.pt"
    print(f"\nDone. Best model: {best}")
    print("Use it: in config.yaml set")
    print(f"  weights: {best.relative_to(ROOT).as_posix()}")
    print(f"  imgsz: {args.imgsz}")
    print("then compare: python scripts/evaluate.py data/videos/<video>.mp4 "
          f"--models yolo11m.pt {best.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
