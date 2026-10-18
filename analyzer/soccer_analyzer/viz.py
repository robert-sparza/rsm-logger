"""Drawing helpers: video overlays, a live minimap and report plots."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pitch import Pitch

TEAM_FALLBACK = [(255, 255, 0), (255, 0, 255), (160, 160, 160)]  # BGR: cyan, magenta, grey (unknown)
BALL_COLOR = (0, 230, 255)


def draw_pitch(pitch: Pitch, scale: float = 6.0, margin: int = 10) -> np.ndarray:
    """Top-down pitch image; returns BGR canvas. 1 m = `scale` px."""
    W = int(pitch.length * scale) + 2 * margin
    H = int(pitch.width * scale) + 2 * margin
    img = np.full((H, W, 3), (60, 130, 60), np.uint8)
    p = lambda x, y: (int(margin + x * scale), int(margin + y * scale))
    white, L, Wd = (235, 235, 235), pitch.length, pitch.width
    cv2.rectangle(img, p(0, 0), p(L, Wd), white, 2)
    cv2.line(img, p(L / 2, 0), p(L / 2, Wd), white, 2)
    cv2.circle(img, p(L / 2, Wd / 2), int(pitch.circle_radius * scale), white, 2)
    bw, gw = pitch.box_width / 2, pitch.goal_area_width / 2
    for x0, sgn in ((0, 1), (L, -1)):
        cv2.rectangle(img, p(x0, Wd / 2 - bw), p(x0 + sgn * pitch.box_depth, Wd / 2 + bw), white, 2)
        cv2.rectangle(img, p(x0, Wd / 2 - gw), p(x0 + sgn * pitch.goal_area_depth, Wd / 2 + gw), white, 2)
    return img


def to_canvas(xy, scale: float = 6.0, margin: int = 10) -> tuple[int, int]:
    return int(margin + xy[0] * scale), int(margin + xy[1] * scale)


def annotate_frame(frame, dets, teams: dict[int, int], team_colors, jersey_colors=None) -> np.ndarray:
    out = frame.copy()
    if jersey_colors:
        # Legend: drawing colour -> the jersey colour the classifier learned for that team.
        for i, (draw, jersey) in enumerate(zip(team_colors, jersey_colors)):
            y = 30 + 34 * i
            cv2.rectangle(out, (20, y - 20), (48, y + 6), draw, -1)
            cv2.rectangle(out, (54, y - 20), (82, y + 6), jersey, -1)
            cv2.rectangle(out, (54, y - 20), (82, y + 6), (255, 255, 255), 1)
            cv2.putText(out, f"Team {i}", (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        if d.cls == "ball":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(out, (cx, cy - 12), BALL_COLOR, cv2.MARKER_TRIANGLE_DOWN, 14, 2)
            continue
        team = teams.get(d.track_id, -1)
        color = team_colors[team] if 0 <= team < len(team_colors) else TEAM_FALLBACK[2]
        cv2.ellipse(out, (int(d.foot[0]), int(d.foot[1])), (max(8, (x2 - x1) // 2), 6), 0, -45, 235, color, 3)
        cv2.putText(out, str(d.track_id), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def minimap(pitch: Pitch, points, team_colors, scale: float = 4.0) -> np.ndarray:
    """points: iterable of (x_m, y_m, team, is_ball)."""
    img = draw_pitch(pitch, scale=scale)
    for x, y, team, is_ball in points:
        if not (-5 <= x <= pitch.length + 5 and -5 <= y <= pitch.width + 5):
            continue
        c = to_canvas((x, y), scale)
        if is_ball:
            cv2.circle(img, c, 4, BALL_COLOR, -1)
        else:
            color = team_colors[team] if 0 <= team < len(team_colors) else TEAM_FALLBACK[2]
            cv2.circle(img, c, 6, color, -1)
            cv2.circle(img, c, 6, (0, 0, 0), 1)
    return img


def paste_minimap(frame: np.ndarray, mm: np.ndarray, width_frac: float = 0.3) -> np.ndarray:
    h, w = frame.shape[:2]
    tw = int(w * width_frac)
    th = int(mm.shape[0] * tw / mm.shape[1])
    small = cv2.resize(mm, (tw, th))
    out = frame.copy()
    out[h - th - 10:h - 10, (w - tw) // 2:(w - tw) // 2 + tw] = small
    return out


# --- static report plots (matplotlib) ---------------------------------------

def _mpl_pitch(ax, pitch: Pitch):
    from matplotlib.patches import Circle, Rectangle
    L, W = pitch.length, pitch.width
    ax.set_facecolor("#3c823c")
    kw = dict(fill=False, ec="white", lw=1.5)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1.5)
    ax.add_patch(Circle((L / 2, W / 2), pitch.circle_radius, **kw))
    bd, bw, gd, gw = pitch.box_depth, pitch.box_width, pitch.goal_area_depth, pitch.goal_area_width
    for x0, s in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if s > 0 else x0 - bd, W / 2 - bw / 2), bd, bw, **kw))
        ax.add_patch(Rectangle((x0 if s > 0 else x0 - gd, W / 2 - gw / 2), gd, gw, **kw))
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(W + 3, -3)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_heatmap(h: np.ndarray, pitch: Pitch, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.8))
    _mpl_pitch(ax, pitch)
    ax.imshow(h, extent=(0, pitch.length, pitch.width, 0), cmap="hot", alpha=0.65,
              interpolation="bicubic", zorder=0)
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ball_path(ball, pitch: Pitch, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.8))
    _mpl_pitch(ax, pitch)
    if len(ball):
        sc = ax.scatter(ball.x, ball.y, c=ball.t, cmap="viridis", s=4)
        ax.plot(ball.x, ball.y, color="white", alpha=0.25, lw=0.8)
        fig.colorbar(sc, ax=ax, label="seconds", shrink=0.7)
    ax.set_title("Ball path")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_passes(passes, df, pitch: Pitch, team: int, path: Path):
    """Arrows from passer to receiver positions at the moment of each pass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.8))
    _mpl_pitch(ax, pitch)
    players = df[df.cls == "player"].set_index(["t", "track_id"])[["x", "y"]]
    for p in passes[(passes.kind == "pass") & (passes.team_from == team)].itertuples():
        try:
            a = players.loc[(p.t_start, p.from_id)]
            b = players.loc[(p.t_end, p.to_id)]
        except KeyError:
            continue
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.4, alpha=0.8))
    ax.set_title(f"Team {team} passes")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
