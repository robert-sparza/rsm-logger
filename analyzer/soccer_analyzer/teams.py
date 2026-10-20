"""Team assignment from shirt colour.

Two modes:

* TeamClassifier (matches): for every player box we take the torso region, drop
  grass-coloured pixels and average the rest in Lab colour space. KMeans (k=2)
  on those colours gives the two teams; each track then takes the majority vote
  of its frames, which smooths over occlusions and bad crops.
* BibClassifier (training with bibs/vests): counts how much of the torso has
  the bib colour. A track is in the bib team if enough of its frames show the
  bib, so players who mostly face away from the camera are still caught.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

# HSV ranges (OpenCV: H 0-180) for common training bibs. Neon yellow was tuned
# on night footage under floodlights where the turf hue is ~44-47.
BIB_COLORS = {
    "neon_yellow": ((22, 100, 160), (42, 255, 255)),
    "orange": ((8, 140, 140), (20, 255, 255)),
    "pink": ((150, 90, 140), (172, 255, 255)),
    "blue": ((95, 120, 90), (125, 255, 255)),
    "red": ((0, 150, 110), (6, 255, 255)),
}
BIB_DRAW_BGR = {"neon_yellow": (0, 255, 220), "orange": (0, 140, 255), "pink": (180, 100, 255),
                "blue": (255, 120, 0), "red": (0, 0, 230)}


def _torso(frame: np.ndarray, box, top: float = 0.15, bottom: float = 0.5, side: float = 0.2):
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < 12 or w < 6:
        return None
    crop = frame[max(y1 + int(top * h), 0): y1 + int(bottom * h), max(x1 + int(side * w), 0): x2 - int(side * w)]
    return crop if crop.size else None


def _grass_hue(frame: np.ndarray, box) -> float:
    """Turf hue measured in the bottom corners of the player's own box."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    rows = frame[max(y2 - int(0.2 * h), 0): y2]
    corners = np.concatenate([rows[:, max(x1, 0): x1 + int(0.25 * w)].reshape(-1, 3),
                              rows[:, x2 - int(0.25 * w): x2].reshape(-1, 3)])
    if len(corners) < 5:
        return 60.0
    hsv = cv2.cvtColor(corners.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    green = hsv[(hsv[:, 1] > 40) & (hsv[:, 0] > 25) & (hsv[:, 0] < 95)]
    return float(np.median(green[:, 0])) if len(green) >= 5 else 60.0


def jersey_color(frame: np.ndarray, box) -> np.ndarray | None:
    crop = _torso(frame, box)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
    grass = (np.abs(hsv[:, 0] - _grass_hue(frame, box)) <= 5) & (hsv[:, 1] > 40)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass]
    if len(lab) < 10:
        return None
    return lab.astype(np.float32).mean(axis=0)


class TeamClassifier:
    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self.kmeans: KMeans | None = None
        self.samples: list[np.ndarray] = []
        self.votes: dict[int, Counter] = defaultdict(Counter)

    feature = staticmethod(jersey_color)

    @property
    def fitted(self) -> bool:
        return self.kmeans is not None

    def add_sample(self, color: np.ndarray) -> None:
        self.samples.append(color)

    def fit(self) -> None:
        if len(self.samples) < self.n_teams * 5:
            raise ValueError("Not enough jersey samples to fit team colours")
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=0)
        self.kmeans.fit(np.stack(self.samples))

    def predict(self, color: np.ndarray) -> int:
        assert self.kmeans is not None
        return int(self.kmeans.predict(color.reshape(1, -1))[0])

    def vote(self, track_id: int, color: np.ndarray) -> int:
        self.votes[track_id][self.predict(color)] += 1
        return self.team_of(track_id)

    def team_of(self, track_id: int) -> int:
        return self.decide(self.votes.get(track_id))

    @staticmethod
    def decide(votes: Counter | None) -> int:
        return votes.most_common(1)[0][0] if votes else -1

    def team_colors_bgr(self) -> list[tuple[int, int, int]]:
        assert self.kmeans is not None
        lab = self.kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        return [tuple(int(v) for v in c) for c in bgr]


class BibClassifier:
    """Team 1 = wearing the bib, team 0 = everyone else. Needs no fitting."""

    def __init__(self, color: str = "neon_yellow", min_fraction: float = 0.04, min_share: float = 0.2):
        if color not in BIB_COLORS:
            raise ValueError(f"Unknown bib colour {color!r}; choose from {sorted(BIB_COLORS)}")
        self.color = color
        self.lo, self.hi = BIB_COLORS[color]
        self.min_fraction = min_fraction    # share of torso pixels that must be bib-coloured
        self.min_share = min_share          # share of a track's frames that must show the bib
        self.votes: dict[int, Counter] = defaultdict(Counter)
        self.samples: list = []

    fitted = True

    def feature(self, frame: np.ndarray, box) -> float | None:
        crop = _torso(frame, box, top=0.1, bottom=0.55, side=0.0)
        if crop is None:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float((cv2.inRange(hsv, self.lo, self.hi) > 0).mean())

    def add_sample(self, _):
        pass

    def fit(self):
        pass

    def vote(self, track_id: int, fraction: float) -> int:
        self.votes[track_id][int(fraction >= self.min_fraction)] += 1
        return self.team_of(track_id)

    def team_of(self, track_id: int) -> int:
        return self.decide(self.votes.get(track_id))

    def decide(self, votes: Counter | None) -> int:
        if not votes or votes[0] + votes[1] == 0:
            return -1
        return int(votes[1] / (votes[0] + votes[1]) >= self.min_share)

    def team_colors_bgr(self) -> list[tuple[int, int, int]]:
        return [(60, 60, 60), BIB_DRAW_BGR[self.color]]
