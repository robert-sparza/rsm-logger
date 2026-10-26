"""Export an analyzed session to kloppy's standard tracking format.

Usage:
    python scripts/export_kloppy.py runs/training --calib runs/calib_small.json

Reads <run>/tracks.csv and writes <run>/tracking_kloppy.csv (one row per frame, one column pair per
player and the ball, positions on a 0..1 pitch). Python code can use the dataset directly:

    from soccer_analyzer.export import to_kloppy
    dataset = to_kloppy(df, pitch, fps)       # a kloppy TrackingDataset
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.export import to_kloppy  # noqa: E402
from soccer_analyzer.pitch import Homography  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="output folder of analyze.py (has tracks.csv)")
    ap.add_argument("--calib", required=True, help="the calibration json used for that run (gives the pitch size)")
    ap.add_argument("--fps", type=float, help="frames per second of the analyzed video (default: from tracks.csv)")
    args = ap.parse_args()

    run = Path(args.run)
    df = pd.read_csv(run / "tracks.csv")
    pitch = Homography.load(args.calib).pitch
    d = df.drop_duplicates("frame").sort_values("frame")
    fps = args.fps or ((len(d) - 1) / (d.t.iloc[-1] - d.t.iloc[0]) if len(d) > 1 else 30.0)
    dataset = to_kloppy(df, pitch, fps)
    out = run / "tracking_kloppy.csv"
    dataset.to_df(engine="pandas").to_csv(out, index=False)
    print(f"{len(dataset.records)} frames, {sum(len(t.players) for t in dataset.metadata.teams)} players -> {out}")


if __name__ == "__main__":
    main()
