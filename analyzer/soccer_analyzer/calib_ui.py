"""The clicking logic of scripts/calibrate.py, without any window, so it can be tested.

Landmarks are offered in order, but any of them can be skipped, or chosen directly with the letter shown
next to its name, so you only click the ones that are visible and sure.
"""
from __future__ import annotations

# Letters that choose a landmark. s (skip), u (undo) and q (finish) are left out: they are commands.
KEYS = "abcdefghijklmnoprtvwxyz"


class CalibState:
    def __init__(self, names: list[str]):
        if len(names) > len(KEYS):
            raise ValueError("more landmarks than keys")
        self.names = list(names)
        self.points: dict[str, tuple[float, float]] = {}
        self.skipped: set[str] = set()
        self.history: list[tuple[str, str]] = []        # ("click" | "skip", landmark)
        self.current: str | None = self.names[0] if self.names else None

    def key_of(self, name: str) -> str:
        return KEYS[self.names.index(name)]

    def _advance(self, after: str) -> None:
        """Next landmark that is neither placed nor skipped, after `after`, wrapping round; None if all are done."""
        start = self.names.index(after)
        for k in range(1, len(self.names) + 1):
            n = self.names[(start + k) % len(self.names)]
            if n not in self.points and n not in self.skipped:
                self.current = n
                return
        self.current = None

    def click(self, pt) -> None:
        if self.current is None:
            return
        name = self.current
        self.points[name] = (float(pt[0]), float(pt[1]))
        self.skipped.discard(name)
        self.history.append(("click", name))
        self._advance(name)

    def skip(self) -> None:
        if self.current is None:
            return
        name = self.current
        self.skipped.add(name)
        self.history.append(("skip", name))
        self._advance(name)

    def jump(self, key: str) -> bool:
        """Make the landmark with this letter current (re-clicking a placed one replaces its point)."""
        i = KEYS.find(key)
        if 0 <= i < len(self.names):
            self.current = self.names[i]
            return True
        return False

    def undo(self) -> None:
        if not self.history:
            return
        kind, name = self.history.pop()
        if kind == "click":
            self.points.pop(name, None)
        else:
            self.skipped.discard(name)
        self.current = name

    @property
    def placed(self) -> list[str]:
        """Landmarks with a point, in landmark order."""
        return [n for n in self.names if n in self.points]
