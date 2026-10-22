"""Render every part in arm.PARTS to renders/<name>.png (shaded, two views)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import arm

OUT = Path(__file__).parent / "renders"


def mesh(shape):
    verts, tris = shape.val().tessellate(0.1, 0.2)
    v = np.array([(p.x, p.y, p.z) for p in verts])
    return v[np.array(tris)]


def draw(ax, tri, elev, azim):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    col = np.outer(shade, [0.95, 0.55, 0.15])
    ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in arm.PARTS.items():
        tri = mesh(fn())
        fig = plt.figure(figsize=(15, 5))
        for i, (e, a) in enumerate([(30, -60), (90, -90), (0, -90)]):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            ax.set_proj_type("ortho")
            draw(ax, tri, e, a)
            ax.set_title(["iso", "top", "front"][i])
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=80)
        plt.close(fig)
        print("rendered", name)
