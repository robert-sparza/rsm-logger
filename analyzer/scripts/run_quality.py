"""Score a finished run from its tracks.csv.

Usage:
    python scripts/run_quality.py runs/run4v4e_3 --players 8 --bibs 4

--players  how many players were on the pitch (leave out if it varies)
--bibs     how many wore the bib (leave out if none or it varies)
Prints the checks and saves them to <run>/quality.txt, so runs can be compared.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_analyzer.quality import format_quality, run_quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--players", type=int)
    ap.add_argument("--bibs", type=int)
    args = ap.parse_args()
    run = Path(args.run)
    text = format_quality(run_quality(pd.read_csv(run / "tracks.csv"), args.players, args.bibs))
    print(f"Quality of {run.name}:\n{text}")
    (run / "quality.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
