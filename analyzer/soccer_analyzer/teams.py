"""Unsupervised team assignment from jersey colour.

For every player box we take the torso region, drop grass-coloured pixels and
average the rest in Lab colour space. KMeans (k=2) on those colours gives the
two teams; each track then takes the majority vote of its frames, which smooths
over occlusions and bad crops. Goalkeepers / referees usually end up as
outliers and can be relabelled later.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


def jersey_color(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < 12 or w < 6:
        return None
    # Torso: roughly 15%-50% of the box height, middle 60% of the width.
    crop = frame[y1 + int(0.15 * h): y1 + int(0.5 * h), x1 + int(0.2 * w): x2 - int(0.2 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) > 0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass.reshape(-1)]
    if len(lab) < 10:
        return None
    return lab.astype(np.float32).mean(axis=0)


class TeamClassifier:
    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self.kmeans: KMeans | None = None
        self.samples: list[np.ndarray] = []
        self.votes: dict[int, Counter] = defaultdict(Counter)

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
        c = self.votes.get(track_id)
        return c.most_common(1)[0][0] if c else -1

    def team_colors_bgr(self) -> list[tuple[int, int, int]]:
        assert self.kmeans is not None
        lab = self.kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        return [tuple(int(v) for v in c) for c in bgr]
