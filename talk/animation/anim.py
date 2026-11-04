"""Goofy explainer animation for the GR00T talk -> timothy_explainer.mp4.

Everything is drawn in code to look like a crude MS Paint doodle (wobbly lines
that "boil" a few times a second), except Timothy, who is the real photo
(timothy.png) with googly eyes stuck on. Nothing is ever shown being drawn:
things pop in, move, shake and get cut. Timed to voice.m4a with words.json
(made by transcribe.py).

    uv run python anim.py                     # the whole video
    uv run python anim.py --from 20 --to 50   # a piece, for checking
    uv run python anim.py --stills 5,12.5,30  # PNG stills in stills/
"""
import argparse
import math
import os
import zlib
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
DURATION = 148.0
BOIL = 4          # frames per wobble pose (~7 poses a second)
NVAR = 3          # wobble poses per sprite

FONTS = {"hand": r"C:\Windows\Fonts\Inkfree.ttf", "comic": r"C:\Windows\Fonts\comicbd.ttf"}

BLACK = (25, 25, 25, 255)
WHITE = (255, 255, 255, 255)
RED = (230, 40, 40, 255)
YEL = (255, 214, 40, 255)
GOLD = (245, 190, 20, 255)
NVG = (118, 185, 0, 255)          # NVIDIA-ish green
TEAL = (16, 163, 127, 255)
ORANGE = (217, 119, 87, 255)
BLUE = (40, 90, 230, 255)
LBLUE = (120, 190, 255, 255)
GREEN = (40, 170, 60, 255)
GREY = (160, 160, 160, 255)
DGREY = (70, 70, 70, 255)
PINK = (255, 140, 180, 255)
PURPLE = (150, 70, 200, 255)
BROWN = (150, 95, 45, 255)

# ---------------------------------------------------------------- state
FR = 0                 # frame being drawn
CAN = None             # the canvas (RGB) of that frame
SHAKES = []            # (t0, dur, amp) camera shakes


def var():
    return FR // BOIL


def vrng(*key):
    """A random generator that changes with the wobble pose (for direct drawing)."""
    return np.random.default_rng(zlib.crc32(repr((key, var())).encode()))


@lru_cache(None)
def font(kind, size):
    return ImageFont.truetype(FONTS[kind], size)


# ---------------------------------------------------------------- timing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return ease((t - a) / (b - a))


def lerp(a, b, u):
    return a + (b - a) * u


def pop(t, t0, d=0.22):
    """0 before t0, then springs up to 1 with an overshoot."""
    if t < t0:
        return 0.0
    x = (t - t0) / d
    if x >= 1:
        return 1.0
    c = 2.4
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def wig(t, f=1.0, a=1.0, ph=0.0):
    return a * math.sin(2 * math.pi * f * t + ph)


def thud(t, t0, amp=1.0, f=6.0, decay=6.0):
    """Damped wobble after an impact at t0."""
    if t < t0:
        return 0.0
    x = t - t0
    return amp * math.exp(-decay * x) * math.sin(2 * math.pi * f * x)


# ---------------------------------------------------------------- wobbly drawing
def wob(pts, rng, amp=2.5, step=16):
    pts = [(float(x), float(y)) for x, y in pts]
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * j / k, y0 + (y1 - y0) * j / k) for j in range(k)]
    out.append(pts[-1])
    a = np.array(out)
    n = rng.normal(0, amp, a.shape)
    if len(a) > 3:
        n[1:-1] = (n[:-2] + n[1:-1] + n[2:]) / 3
    return [tuple(p) for p in a + n]


def line(d, pts, rng, w=7, col=BLACK, amp=2.5):
    p = wob(pts, rng, amp)
    d.line(p, fill=col, width=w, joint="curve")
    r = w / 2 - 0.5
    for x, y in (p[0], p[-1]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)


def poly(d, pts, rng, fill=None, w=7, col=BLACK, amp=2.5):
    if fill:
        d.polygon(wob(list(pts) + [pts[0]], rng, amp * 0.5), fill=fill)
    if w:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        over = (x0 + (x1 - x0) * 0.15, y0 + (y1 - y0) * 0.15)
        line(d, list(pts) + [pts[0], over], rng, w, col, amp)


def ell(d, cx, cy, rx, ry, rng, fill=None, w=7, col=BLACK, rot=0.0, amp=1.8):
    n = max(24, int((rx + ry) / 4))
    ph = rng.uniform(0, 6.28, 2)
    a0 = rng.uniform(0, 6.28)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def pt(th):
        r = 1 + 0.035 * math.sin(2 * th + ph[0]) + 0.02 * math.sin(3 * th + ph[1])
        x, y = rx * r * math.cos(th), ry * r * math.sin(th)
        return (cx + x * c - y * s, cy + x * s + y * c)

    if fill:
        d.polygon([pt(a0 + 2 * math.pi * i / n) for i in range(n)], fill=fill)
    if w:
        extra = int(n * 0.07) + 1
        line(d, [pt(a0 + 2 * math.pi * i / n) for i in range(n + extra)], rng, w, col, amp)


def arrow(d, p0, p1, rng, w=9, col=BLACK, head=36):
    line(d, [p0, ((p0[0] + p1[0]) / 2 + 12, (p0[1] + p1[1]) / 2 - 12), p1], rng, w, col)
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (2.6, -2.6):
        line(d, [p1, (p1[0] + head * math.cos(a + da), p1[1] + head * math.sin(a + da))], rng, w, col)


def eye(d, x, y, r, rng, look=(0.0, 0.0), mode="normal", pupil=0.48):
    """A googly eye. mode: normal, x, spiral, closed, tiny, heart."""
    ell(d, x, y, r, r, rng, fill=WHITE, w=max(3, int(r * 0.13)))
    lw = max(3, int(r * 0.14))
    if mode == "x":
        k = r * 0.55
        line(d, [(x - k, y - k), (x + k, y + k)], rng, lw)
        line(d, [(x - k, y + k), (x + k, y - k)], rng, lw)
        return
    if mode == "spiral":
        pts = [(x + r * 0.8 * (i / 40) * math.cos(i * 0.55 + FR * 0.5),
                y + r * 0.8 * (i / 40) * math.sin(i * 0.55 + FR * 0.5)) for i in range(41)]
        line(d, pts, rng, max(2, lw - 1), amp=0.6)
        return
    if mode == "closed":
        line(d, [(x - r * 0.8, y + r * 0.05), (x, y + r * 0.3), (x + r * 0.8, y + r * 0.05)], rng, lw)
        return
    if mode == "heart":
        heart(d, x, y + r * 0.1, r * 0.75, rng)
        return
    if mode == "tiny":
        pupil = 0.18
    lx, ly = look
    m = math.hypot(lx, ly)
    if m > 1:
        lx, ly = lx / m, ly / m
    pr = r * pupil
    reach = r - pr - r * 0.12
    px, py = x + lx * reach, y + ly * reach
    d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=BLACK)
    g = pr * 0.3
    d.ellipse([px - pr * 0.35 - g, py - pr * 0.35 - g, px - pr * 0.35 + g, py - pr * 0.35 + g], fill=WHITE)


def heart(d, x, y, r, rng, col=RED):
    pts = []
    for i in range(40):
        th = 2 * math.pi * i / 40
        hx = 16 * math.sin(th) ** 3
        hy = -(13 * math.cos(th) - 5 * math.cos(2 * th) - 2 * math.cos(3 * th) - math.cos(4 * th))
        pts.append((x + hx * r / 17, y + hy * r / 17))
    poly(d, pts, rng, fill=col, w=max(3, int(r * 0.1)))


def wander(t, i, speed=1.0):
    """Googly pupils that drift around on their own."""
    t *= speed
    return (0.75 * math.sin(t * 1.7 + i * 2.1) + 0.3 * math.sin(t * 4.3 + i),
            0.6 * math.cos(t * 1.3 + i * 1.4) + 0.2 * math.sin(t * 5.1 + i * 3))


# ---------------------------------------------------------------- sprites (cached)
SPR = {}
_cache = {}


def S(fn):
    SPR[fn.__name__] = fn
    return fn


def sprite(name, *args):
    key = (name,) + args
    if key not in _cache:
        _cache[key] = [SPR[name](np.random.default_rng(zlib.crc32(repr((key, v)).encode())), *args)
                       for v in range(NVAR)]
    return _cache[key][(var() + zlib.crc32(repr(key).encode())) % NVAR]


def new(w, h):
    im = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


@S
def text(rng, s, size, col=BLACK, kind="comic", stroke=0, scol=WHITE, tilt=0.0):
    f = font(kind, size)
    lines = s.split("\n")
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [probe.textbbox((0, 0), ln, font=f, stroke_width=stroke)[2] for ln in lines]
    lh = int(size * 1.18)
    im, d = new(max(widths) + 60 + 2 * stroke, lh * len(lines) + 50 + 2 * stroke)
    for i, ln in enumerate(lines):
        d.text((im.width / 2, 25 + stroke + lh * i + lh / 2), ln, font=f, fill=col, anchor="mm",
               stroke_width=stroke, stroke_fill=scol)
    return im.rotate(tilt + rng.uniform(-1.6, 1.6), resample=Image.BICUBIC, expand=True)


@S
def gpt_body(rng):
    im, d = new(440, 440)
    ell(d, 220, 220, 190, 190, rng, fill=TEAL, w=9)
    for k in range(6):
        a = math.radians(k * 60)
        ell(d, 220 + 58 * math.cos(a), 220 + 58 * math.sin(a), 92, 40, rng, w=11, col=WHITE, rot=k * 60 + 90)
    return im


@S
def cheese_body(rng):
    im, d = new(520, 400)
    poly(d, [(40, 150), (400, 150), (490, 70), (130, 70)], rng, fill=(255, 232, 110, 255))
    poly(d, [(400, 150), (490, 70), (490, 290), (400, 370)], rng, fill=(222, 170, 20, 255))
    poly(d, [(40, 150), (400, 150), (400, 370), (40, 370)], rng, fill=(250, 200, 35, 255))
    hole = (200, 145, 15, 255)
    for x, y, rx, ry in [(95, 325, 22, 15), (340, 190, 17, 12), (355, 330, 14, 10), (300, 105, 22, 8),
                         (190, 90, 14, 6), (448, 170, 10, 20), (455, 260, 8, 14)]:
        ell(d, x, y, rx, ry, rng, fill=hole, w=3)
    line(d, [(150, 318), (200, 336), (255, 322)], rng, 6)
    ell(d, 214, 340, 15, 12, rng, fill=PINK, w=4)
    return im


@S
def groot_body(rng):
    im, d = new(500, 500)
    dark, lite = (88, 145, 0, 255), (160, 215, 50, 255)
    poly(d, [(110, 70), (40, 130), (5, 75), (80, 25)], rng, fill=lite)          # flaps
    poly(d, [(380, 130), (450, 70), (495, 115), (425, 175)], rng, fill=lite)
    poly(d, [(40, 130), (380, 130), (450, 70), (110, 70)], rng, fill=lite)
    poly(d, [(380, 130), (450, 70), (450, 410), (380, 470)], rng, fill=dark)
    poly(d, [(40, 130), (380, 130), (380, 470), (40, 470)], rng, fill=NVG)
    d.text((210, 300), "ISAAC", font=font("comic", 52), fill=WHITE, anchor="mm", stroke_width=3,
           stroke_fill=BLACK)
    d.text((210, 368), "GR00T", font=font("comic", 70), fill=WHITE, anchor="mm", stroke_width=4,
           stroke_fill=BLACK)
    d.text((330, 440), "N1.7", font=font("hand", 30), fill=BLACK, anchor="mm")
    return im


@S
def me_body(rng, tired=False):
    im, d = new(300, 560)
    ell(d, 150, 110, 64, 72, rng, fill=WHITE)
    line(d, [(92, 75), (105, 40), (118, 70), (132, 30), (148, 65), (165, 28), (178, 66), (195, 38),
             (210, 78)], rng, 7)
    line(d, [(150, 182), (150, 370)], rng, 8)
    line(d, [(150, 232), (72, 312)], rng, 8)
    line(d, [(150, 232), (232, 300)], rng, 8)
    line(d, [(150, 370), (95, 522)], rng, 8)
    line(d, [(150, 370), (212, 522)], rng, 8)
    if tired:
        line(d, [(118, 158), (150, 150), (182, 160)], rng, 5)
        for x in (124, 174):
            line(d, [(x - 16, 124), (x, 130), (x + 16, 124)], rng, 3, col=PURPLE)
    else:
        line(d, [(120, 148), (150, 160), (182, 146)], rng, 5)
    return im


@S
def claude_body(rng):
    im, d = new(440, 440)
    pts = []
    n = 12
    for i in range(2 * n):
        r = 195 if i % 2 == 0 else 80
        a = math.pi * i / n + rng.uniform(-0.04, 0.04)
        pts.append((220 + r * math.cos(a), 220 + r * math.sin(a)))
    poly(d, pts, rng, fill=ORANGE, w=8)
    line(d, [(185, 262), (220, 278), (258, 260)], rng, 6)
    return im


@S
def ball(rng, col=RED, r=50):
    im, d = new(2 * r + 30, 2 * r + 30)
    c = r + 15
    ell(d, c, c, r, r, rng, fill=col, w=7)
    ell(d, c - r * 0.35, c - r * 0.35, r * 0.18, r * 0.12, rng, fill=WHITE, w=0, rot=-40)
    return im


@S
def block(rng, col=GREEN):
    im, d = new(170, 170)
    poly(d, [(20, 55), (115, 55), (150, 25), (55, 25)], rng, fill=(120, 220, 120, 255), w=6)
    poly(d, [(115, 55), (150, 25), (150, 115), (115, 150)], rng, fill=(20, 120, 40, 255), w=6)
    poly(d, [(20, 55), (115, 55), (115, 150), (20, 150)], rng, fill=col, w=6)
    return im


@S
def cup(rng):
    im, d = new(240, 240)
    poly(d, [(25, 45), (215, 45), (190, 225), (50, 225)], rng, fill=(45, 45, 45, 255))
    ell(d, 120, 45, 95, 22, rng, fill=(95, 95, 95, 255), w=6)
    return im


@S
def watch(rng, alarm=False):
    im, d = new(440, 500)
    if alarm:
        ell(d, 105, 105, 62, 45, rng, fill=RED, rot=-35)
        ell(d, 335, 105, 62, 45, rng, fill=RED, rot=35)
        line(d, [(120, 440), (80, 490)], rng, 9)
        line(d, [(320, 440), (360, 490)], rng, 9)
    else:
        poly(d, [(195, 30), (245, 30), (245, 80), (195, 80)], rng, fill=GREY)
        poly(d, [(170, 10), (270, 10), (270, 38), (170, 38)], rng, fill=GREY)
    ell(d, 220, 280, 190, 190, rng, fill=(250, 250, 240, 255) if not alarm else WHITE, w=9,
        col=BLACK if not alarm else RED)
    ell(d, 220, 280, 190, 190, rng, w=9)
    for k in range(12):
        a = 2 * math.pi * k / 12
        r0 = 150 if k % 3 else 132
        line(d, [(220 + r0 * math.cos(a), 280 + r0 * math.sin(a)),
                 (220 + 170 * math.cos(a), 280 + 170 * math.sin(a))], rng, 6)
    return im


@S
def nn(rng):
    im, d = new(440, 280)
    layers = [[60, 140, 220], [50, 110, 170, 230], [100, 180]]
    xs = [60, 220, 380]
    for li in range(2):
        for y0 in layers[li]:
            for y1 in layers[li + 1]:
                line(d, [(xs[li], y0), (xs[li + 1], y1)], rng, 3, amp=1.2)
    for li, ys in enumerate(layers):
        for y in ys:
            ell(d, xs[li], y, 24, 24, rng, fill=[LBLUE, PINK, YEL][li], w=5)
    return im


@S
def card(rng, i, ang):
    im, d = new(120, 150)
    poly(d, [(10, 10), (110, 10), (110, 140), (10, 140)], rng, fill=WHITE, w=5)
    d.text((28, 30), str(i + 1), font=font("comic", 26), fill=BLACK, anchor="mm")
    a = math.radians(ang)
    c = (60, 85)
    p0 = (c[0] - 32 * math.cos(a), c[1] - 32 * math.sin(a))
    p1 = (c[0] + 32 * math.cos(a), c[1] + 32 * math.sin(a))
    arrow(d, p0, p1, rng, w=6, col=BLUE, head=16)
    return im


@S
def check(rng, col=GREEN, size=120):
    im, d = new(size * 1.3, size * 1.2)
    line(d, [(size * 0.15, size * 0.6), (size * 0.45, size * 0.95), (size * 1.15, size * 0.15)], rng,
         int(size * 0.16), col)
    return im


@S
def xmark(rng, size=110):
    im, d = new(size, size)
    k = size * 0.15
    line(d, [(k, k), (size - k, size - k)], rng, int(size * 0.14), RED)
    line(d, [(k, size - k), (size - k, k)], rng, int(size * 0.14), RED)
    return im


@S
def trophy(rng):
    im, d = new(420, 520)
    ell(d, 95, 170, 70, 80, rng, w=16, col=GOLD)
    ell(d, 325, 170, 70, 80, rng, w=16, col=GOLD)
    ell(d, 95, 170, 70, 80, rng, w=5)
    ell(d, 325, 170, 70, 80, rng, w=5)
    poly(d, [(80, 60), (340, 60), (300, 260), (240, 320), (180, 320), (120, 260)], rng, fill=GOLD)
    poly(d, [(185, 315), (235, 315), (245, 400), (175, 400)], rng, fill=GOLD)
    poly(d, [(110, 400), (310, 400), (310, 480), (110, 480)], rng, fill=BROWN)
    d.text((210, 170), "#1", font=font("comic", 110), fill=WHITE, anchor="mm", stroke_width=4,
           stroke_fill=BLACK)
    return im


@S
def crown(rng):
    im, d = new(300, 200)
    poly(d, [(25, 175), (25, 60), (85, 120), (150, 25), (215, 120), (275, 60), (275, 175)], rng, fill=GOLD)
    for x, c in [(80, RED), (150, BLUE), (220, GREEN)]:
        ell(d, x, 145, 16, 16, rng, fill=c, w=4)
    return im


@S
def gradcap(rng):
    im, d = new(320, 220)
    poly(d, [(80, 120), (240, 120), (235, 190), (85, 190)], rng, fill=BLACK)
    poly(d, [(10, 95), (160, 30), (310, 95), (160, 160)], rng, fill=BLACK)
    line(d, [(160, 95), (270, 120), (275, 185)], rng, 5, col=YEL)
    ell(d, 275, 192, 12, 16, rng, fill=YEL, w=3)
    return im


@S
def shades(rng):
    im, d = new(340, 130)
    poly(d, [(15, 30), (150, 30), (140, 105), (40, 110)], rng, fill=BLACK)
    poly(d, [(190, 30), (325, 30), (300, 110), (200, 105)], rng, fill=BLACK)
    line(d, [(150, 45), (170, 38), (190, 45)], rng, 8)
    line(d, [(40, 45), (75, 50)], rng, 4, col=WHITE)
    line(d, [(215, 45), (250, 50)], rng, 4, col=WHITE)
    return im


@S
def paper(rng, k):
    im, d = new(130, 160)
    poly(d, [(10, 10), (120, 10), (120, 150), (10, 150)], rng, fill=WHITE, w=4)
    if k % 3 == 0:          # tiny robot
        poly(d, [(40, 40), (90, 40), (90, 80), (40, 80)], rng, fill=GREY, w=3)
        line(d, [(65, 80), (65, 120)], rng, 3)
        line(d, [(65, 95), (95, 110)], rng, 3)
    else:
        for j in range(5):
            line(d, [(25, 35 + j * 22), (105 - (j * 13) % 40, 35 + j * 22)], rng, 3, col=DGREY, amp=1)
    return im


@S
def bulb(rng):
    im, d = new(220, 300)
    ell(d, 110, 110, 85, 90, rng, fill=YEL)
    poly(d, [(75, 190), (145, 190), (140, 250), (80, 250)], rng, fill=GREY)
    line(d, [(85, 215), (135, 215)], rng, 4)
    line(d, [(90, 115), (110, 85), (130, 115)], rng, 5, col=(200, 120, 0, 255))
    return im


@S
def warn(rng):
    im, d = new(420, 380)
    poly(d, [(210, 20), (400, 355), (20, 355)], rng, fill=YEL, w=14, col=RED)
    line(d, [(210, 120), (210, 250)], rng, 26)
    ell(d, 210, 300, 16, 16, rng, fill=BLACK, w=0)
    return im


@S
def boom(rng, s, col=YEL):
    im, d = new(640, 420)
    pts = []
    n = 14
    for i in range(2 * n):
        r = (1.0 if i % 2 == 0 else 0.62) * rng.uniform(0.9, 1.08)
        a = math.pi * i / n
        pts.append((320 + 300 * r * math.cos(a), 210 + 195 * r * math.sin(a)))
    poly(d, pts, rng, fill=col, w=9, col=RED)
    d.text((320, 210), s, font=font("comic", 76), fill=RED, anchor="mm", stroke_width=4, stroke_fill=WHITE)
    return im


@S
def tape(rng):
    im, d = new(240, 130)
    poly(d, [(10, 10), (230, 10), (230, 120), (10, 120)], rng, fill=(35, 35, 35, 255), w=5)
    poly(d, [(35, 22), (205, 22), (205, 52), (35, 52)], rng, fill=WHITE, w=3)
    d.text((120, 38), "DEMO", font=font("comic", 22), fill=BLACK, anchor="mm")
    ell(d, 80, 86, 18, 18, rng, fill=WHITE, w=3)
    ell(d, 160, 86, 18, 18, rng, fill=WHITE, w=3)
    return im


@S
def servo(rng):
    im, d = new(200, 140)
    poly(d, [(15, 35), (185, 35), (185, 125), (15, 125)], rng, fill=BLUE, w=6)
    poly(d, [(40, 10), (90, 10), (90, 35), (40, 35)], rng, fill=WHITE, w=5)
    d.text((110, 85), "FANCY", font=font("hand", 30), fill=WHITE, anchor="mm")
    return im


@S
def bill(rng):
    im, d = new(520, 260)
    poly(d, [(15, 15), (505, 15), (505, 245), (15, 245)], rng, fill=(150, 210, 140, 255), w=8)
    poly(d, [(40, 40), (480, 40), (480, 220), (40, 220)], rng, w=4, col=(40, 110, 40, 255))
    ell(d, 260, 130, 70, 80, rng, fill=(200, 240, 190, 255), w=5, col=(40, 110, 40, 255))
    d.text((260, 135), "$", font=font("comic", 80), fill=(40, 110, 40, 255), anchor="mm")
    d.text((95, 75), "20", font=font("comic", 50), fill=(40, 110, 40, 255), anchor="mm")
    d.text((425, 190), "20", font=font("comic", 50), fill=(40, 110, 40, 255), anchor="mm")
    return im


@S
def smug_arm(rng):
    """A different, fancier arm that GR00T has seen before (drawn, made up)."""
    im, d = new(560, 680)
    lite, acc = (240, 240, 240, 255), (255, 140, 30, 255)
    poly(d, [(130, 560), (370, 560), (390, 660), (110, 660)], rng, fill=lite)
    ell(d, 250, 560, 120, 30, rng, fill=lite)
    poly(d, [(215, 560), (285, 560), (250, 330), (180, 320)], rng, fill=lite)
    ell(d, 215, 320, 42, 42, rng, fill=acc)
    poly(d, [(200, 290), (230, 350), (420, 245), (395, 195)], rng, fill=lite)
    ell(d, 410, 215, 40, 40, rng, fill=acc)
    line(d, [(440, 195), (530, 140)], rng, 14)
    line(d, [(445, 240), (540, 230)], rng, 14)
    sh = sprite("shades")
    im.alpha_composite(sh.resize((200, 76)).rotate(18, expand=True), (300, 100))
    line(d, [(345, 290), (385, 285), (410, 265)], rng, 6)
    return im


@S
def code_panel(rng, k):
    lines = ["# inverse_kinematics.py", "def ik(x, y, z):", "    base = atan2(y, x)",
             "    r = hypot(x, y)", "    elbow = acos((r^2 + z^2 - L1^2 - L2^2)",
             "                 / (2 * L1 * L2))", "    # pls work", "    return angles  # ?????"]
    im, d = new(920, 560)
    poly(d, [(15, 15), (905, 15), (905, 545), (15, 545)], rng, fill=(30, 30, 45, 255), w=8)
    for i, s in enumerate(lines[:k]):
        col = (120, 220, 120, 255) if s.strip().startswith("#") else (235, 235, 235, 255)
        d.text((50, 60 + i * 60), s, font=font("comic", 34), fill=col, anchor="lm")
    return im


@S
def protractor(rng):
    im, d = new(330, 190)
    pts = [(165 + 150 * math.cos(math.pi + math.pi * i / 30), 175 + 150 * math.sin(math.pi + math.pi * i / 30))
           for i in range(31)]
    poly(d, pts, rng, fill=(200, 230, 255, 200), w=6)
    for k in range(1, 12):
        a = math.pi + math.pi * k / 12
        line(d, [(165 + 125 * math.cos(a), 175 + 125 * math.sin(a)),
                 (165 + 148 * math.cos(a), 175 + 148 * math.sin(a))], rng, 3)
    return im


@S
def sign(rng, s, col=BROWN):
    f = font("comic", 34)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = s.split("\n")
    w = max(probe.textbbox((0, 0), ln, font=f)[2] for ln in lines) + 50
    h = 44 * len(lines) + 30
    im, d = new(w + 20, h + 20)
    poly(d, [(10, 10), (w + 10, 10), (w + 10, h + 10), (10, h + 10)], rng, fill=(235, 200, 140, 255), w=6,
         col=col)
    for i, ln in enumerate(lines):
        d.text((w / 2 + 10, 10 + 15 + 44 * i + 22), ln, font=f, fill=BLACK, anchor="mm")
    return im


@S
def loop_arrow(rng):
    im, d = new(640, 640)
    pts = [(320 + 260 * math.cos(math.radians(a)), 320 + 260 * math.sin(math.radians(a)))
           for a in range(20, 330, 8)]
    line(d, pts, rng, 18, col=PURPLE)
    p1 = pts[-1]
    a = math.radians(330)
    tang = a + math.pi / 2
    for da in (2.5, -2.5):
        line(d, [p1, (p1[0] + 70 * math.cos(tang + da), p1[1] + 70 * math.sin(tang + da))], rng, 18,
             col=PURPLE)
    return im


@S
def skull(rng):
    im, d = new(140, 150)
    ell(d, 70, 62, 52, 50, rng, fill=WHITE, w=6)
    poly(d, [(42, 95), (98, 95), (92, 135), (48, 135)], rng, fill=WHITE, w=6)
    ell(d, 50, 62, 13, 15, rng, fill=BLACK, w=0)
    ell(d, 90, 62, 13, 15, rng, fill=BLACK, w=0)
    return im


@S
def scene_frame(rng):
    im, d = new(680, 460)
    poly(d, [(15, 15), (665, 15), (665, 445), (15, 445)], rng, fill=(245, 245, 235, 255), w=9)
    line(d, [(30, 390), (650, 395)], rng, 4, col=GREY)
    for (bx, by), b in [((200, 330), sprite("ball", RED, 50)), ((490, 300), sprite("cup")),
                        ((330, 330), sprite("block", GREEN))]:
        bb = b.resize((int(b.width * 0.7), int(b.height * 0.7)))
        im.alpha_composite(bb, (int(bx - bb.width / 2), int(by - bb.height / 2)))
    d.text((80, 50), "CAMERA", font=font("hand", 32), fill=RED, anchor="lm")
    ell(d, 50, 50, 10, 10, rng, fill=RED, w=0)
    return im


@S
def prompt_box(rng, s, w=720):
    im, d = new(w, 130)
    poly(d, [(10, 15), (w - 10, 15), (w - 10, 115), (10, 115)], rng, fill=WHITE, w=7)
    d.text((40, 65), "> " + s, font=font("comic", 44), fill=BLACK, anchor="lm")
    poly(d, [(w - 90, 35), (w - 30, 35), (w - 30, 95), (w - 90, 95)], rng, fill=TEAL, w=5)
    return im


@S
def bubble(rng, full, k, w, h, tail):
    """Speech bubble with the first k words of `full`; tail = tip point relative to the box."""
    im, d = new(w + 400, h + 400)
    ox, oy = 200, 200
    tx, ty = tail
    cx, cy = ox + w / 2, oy + h / 2
    base_dx = (tx - w / 2) * 0.25
    poly(d, [(cx + base_dx - 50, cy), (ox + tx, oy + ty), (cx + base_dx + 50, cy)], rng, fill=WHITE, w=7)
    ell(d, cx, cy, w / 2, h / 2, rng, fill=WHITE, w=8)
    poly(d, [(cx + base_dx - 44, cy), (ox + tx, oy + ty), (cx + base_dx + 44, cy)], rng, fill=WHITE, w=0)
    f = font("comic", 60)
    words = full.split()
    lines, cur = [], []
    for wd in words:                      # wrap the full sentence so words don't jump
        trial = " ".join(cur + [wd])
        if cur and d.textlength(trial, font=f) > w * 0.78:
            lines.append(cur)
            cur = [wd]
        else:
            cur.append(wd)
    lines.append(cur)
    lh = 74
    y = cy - lh * (len(lines) - 1) / 2
    n = 0
    for ln in lines:
        x = cx - d.textlength(" ".join(ln), font=f) / 2
        for wd in ln:
            if n < k:
                d.text((x, y), wd, font=f, fill=BLACK, anchor="lm")
            x += d.textlength(wd + " ", font=f)
            n += 1
        y += lh
    return im


@S
def halo(rng):
    im, d = new(300, 110)
    ell(d, 150, 55, 125, 32, rng, w=12, col=GOLD)
    return im


@S
def ground(rng, w=1800):
    im, d = new(w, 40)
    line(d, [(10, 20), (w - 10, 22)], rng, 6, col=DGREY)
    return im


# ---------------------------------------------------------------- characters
@lru_cache(None)
def tim_base():
    im = Image.open(os.path.join(HERE, "timothy.png")).convert("RGBA")
    box = im.getchannel("A").getbbox()
    im = im.crop(box)
    sc = 640 / im.height
    return im.resize((round(im.width * sc), 640), Image.LANCZOS), sc, box


TIM_EYES = [(655, 238, 62), (752, 305, 44)]     # on the wrist ring, in timothy.png pixels
TIM_PIVOT = (0.303, 1.0)                        # bottom middle of the round base


def tim_img(t, mode="normal", look=None, eye_s=1.0, pupil=0.48):
    base, sc, box = tim_base()
    im = base.copy()
    d = ImageDraw.Draw(im)
    for i, (x, y, r) in enumerate(TIM_EYES):
        lk = look if look is not None else wander(t, i)
        eye(d, (x - box[0]) * sc, (y - box[1]) * sc, r * sc * eye_s, vrng("tim", i), lk, mode, pupil)
    return im


def tim_local(x, y):
    """A timothy.png pixel -> offset from the pivot, at scale 1, no rotation."""
    base, sc, box = tim_base()
    return ((x - box[0]) * sc - TIM_PIVOT[0] * base.width, (y - box[1]) * sc - base.height)


def tim_head(px, py, s=1.0):
    ox, oy = tim_local(700, 270)
    return px + ox * s, py + oy * s


def put_tim(x, y, s=1.0, rot=0.0, t=0.0, mode="normal", look=None, eye_s=1.0, pivot=TIM_PIVOT, sxy=None,
            pupil=0.48):
    put(tim_img(t, mode, look, eye_s, pupil), x, y, sxy or s, rot, pivot)


def char_img(body, eyes, t, look=None, mode="normal", eye_s=1.0, seed=0, pupil=0.48):
    im = body.copy()
    d = ImageDraw.Draw(im)
    for i, (x, y, r) in enumerate(eyes):
        lk = look[i] if isinstance(look, list) else (look if look is not None else wander(t, i + seed))
        md = mode[i] if isinstance(mode, list) else mode
        eye(d, x, y, r * eye_s, vrng(seed, i), lk, md, pupil)
    return im


def gpt(t, look=None, mode="normal", eye_s=1.0, pupil=0.48):
    return char_img(sprite("gpt_body"), [(160, 215, 62), (285, 200, 74)], t, look, mode, eye_s, 1, pupil)


def cheese(t, look=None, mode="normal", eye_s=1.0):
    lk = look if look is not None else [wander(t, 0, 1.6), (-wander(t, 3)[0], 0.6)]
    return char_img(sprite("cheese_body"), [(150, 228, 44), (272, 218, 30)], t, lk, mode, eye_s, 2)


def groot(t, look=None, mode="normal", eye_s=1.0):
    return char_img(sprite("groot_body"), [(150, 200, 40), (270, 195, 48)], t, look, mode, eye_s, 3)


def me(t, tired=False, look=None):
    mode = "closed" if tired and int(t * 2) % 3 else "normal"
    return char_img(sprite("me_body", tired), [(126, 102, 18), (174, 99, 15)], t, look, mode, 1.0, 4)


def claude(t, look=None, mode="normal"):
    return char_img(sprite("claude_body"), [(178, 200, 38), (258, 192, 46)], t, look, mode, 1.0, 5)


# ---------------------------------------------------------------- placing things
def put(im, x, y, s=1.0, rot=0.0, pivot=(0.5, 0.5), alpha=1.0):
    sx, sy = (s, s) if not isinstance(s, tuple) else s
    if sx <= 0.01 or sy <= 0.01 or alpha <= 0:
        return
    if sx != 1 or sy != 1:
        im = im.resize((max(1, int(im.width * sx)), max(1, int(im.height * sy))), Image.BILINEAR)
    px, py = pivot[0] * im.width, pivot[1] * im.height
    if rot:
        vx, vy = px - im.width / 2, py - im.height / 2
        a = math.radians(rot)
        im = im.rotate(rot, resample=Image.BILINEAR, expand=True)
        px = im.width / 2 + vx * math.cos(a) + vy * math.sin(a)
        py = im.height / 2 - vx * math.sin(a) + vy * math.cos(a)
    if alpha < 1:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    CAN.paste(im, (int(round(x - px)), int(round(y - py))), im)


def txt(t, t0, s, x, y, size=80, col=BLACK, kind="comic", stroke=0, tilt=0.0, scol=WHITE, rot=0.0):
    p = pop(t, t0)
    if p:
        put(sprite("text", s, size, col, kind, stroke, scol, tilt), x, y, p, rot)


def D():
    return ImageDraw.Draw(CAN)


def shake(t0, dur=0.4, amp=22):
    SHAKES.append((t0, dur, amp))


def fill(col):
    CAN.paste(col[:3], (0, 0, W, H))


def sweat(t, x, y, n=2, every=0.6):
    d = D()
    for i in range(n):
        ph = ((t + i * every / n) % every) / every
        sx = x + (i - (n - 1) / 2) * 50
        sy = y + ph * 120
        ell(d, sx, sy, 10, 16, vrng("sw", i), fill=LBLUE, w=3)


def sparkles(t, pts, size=28, col=YEL):
    d = D()
    for i, (x, y) in enumerate(pts):
        k = size * (0.6 + 0.4 * abs(math.sin(t * 5 + i * 1.7)))
        poly(d, [(x, y - k), (x + k * 0.25, y - k * 0.25), (x + k, y), (x + k * 0.25, y + k * 0.25),
                 (x, y + k), (x - k * 0.25, y + k * 0.25), (x - k, y), (x - k * 0.25, y - k * 0.25)],
             vrng("spk", i), fill=col, w=4)


def confetti(t, t0, n=70):
    if t < t0:
        return
    d = D()
    cols = [RED, BLUE, YEL, GREEN, PINK, PURPLE]
    rng = np.random.default_rng(7)
    for i in range(n):
        x0, sp, ph, delay = rng.uniform(0, W), rng.uniform(250, 450), rng.uniform(0, 6), rng.uniform(0, 1.5)
        tt = t - t0 - delay
        if tt < 0:
            continue
        y = -30 + sp * tt
        if y > H + 30:
            continue
        x = x0 + 40 * math.sin(tt * 3 + ph)
        a = tt * 4 + ph
        c, s = math.cos(a) * 14, math.sin(a) * 14
        d.polygon([(x - c, y - s), (x + s * 0.5, y - c * 0.5), (x + c, y + s), (x - s * 0.5, y + c * 0.5)],
                  fill=cols[i % len(cols)])


def rec_dot(t, x=110, y=80):
    if int(t * 2) % 2 == 0:
        ell(D(), x, y, 22, 22, vrng("rec"), fill=RED, w=0)
    put(sprite("text", "REC", 60, RED), x + 95, y)


def clock_hands(cx, cy, s, t, speed=1.0):
    d = D()
    for L, w, f in [(120, 9, 1.0), (85, 12, 1 / 12)]:
        a = t * speed * f * 2 * math.pi - math.pi / 2
        line(d, [(cx, cy), (cx + L * s * math.cos(a), cy + L * s * math.sin(a))], vrng("hand", L), int(w * s))


# ---------------------------------------------------------------- scenes
def sc_title(t):
    put_tim(700, 1030, 1.0, wig(t, 0.5, 3), t)
    txt(t, 0.5, "THIS IS TIMOTHY", 1380, 280, 104, tilt=-3)
    if t >= 1.5:
        arrow(D(), (1200, 560), (980, 530), vrng("a0"), w=11)
    txt(t, 2.5, "(he's trying his best)", 1380, 400, 66, kind="hand", tilt=2)


def sc_jam(t):
    px, py, s = 700, 1050, 0.95
    hx, hy = tim_head(px, py, s)
    # the box drops in, hovers, then gets crammed into his head
    if t < 6.1:
        by = lerp(-300, 230, ramp(t, 4.75, 5.35)) + thud(t, 5.35, 25)
        bs = 0.6
    else:
        u = ramp(t, 6.1, 6.42)
        by, bs = lerp(230, hy, u), lerp(0.6, 0.12, u)
    sq = 1 - 0.14 * math.exp(-6 * (t - 6.42)) * math.cos(14 * (t - 6.42)) if t >= 6.42 else 1.0
    mode = "spiral" if 6.42 <= t < 7.4 else "normal"
    look = (0.2, -1) if t < 6.42 else None
    put_tim(px, py, rot=wig(t, 0.4, 2), t=t, mode=mode, look=look, sxy=(s * (2 - sq), s * sq),
            eye_s=1.25 if t >= 7.4 else 1.0)
    if t < 6.42:
        put(groot(t, look=(0.3, 1)), hx, by, bs, wig(t, 1.5, 4))
    if 6.42 <= t < 7.3:
        put(sprite("boom", "CRAMMED"), hx, hy - 60, pop(t, 6.42, 0.15) * 0.8, -6)
    if t >= 6.42 and not any(sh[0] == 6.42 for sh in SHAKES):
        shake(6.42, 0.45, 28)
    txt(t, 7.22, "VISION", 1450, 260, 110, BLUE, tilt=-2)
    txt(t, 7.46, "LANGUAGE", 1450, 390, 110, RED, tilt=1)
    txt(t, 7.82, "ACTION", 1450, 520, 110, GREEN, tilt=-1)
    txt(t, 8.26, "= VLA", 1450, 680, 140, tilt=3)
    txt(t, 9.3, "(2 systems??)", 1450, 860, 80, kind="hand", tilt=-2)


def sc_versus(t):
    p = pop(t, 10.5)
    look_g = (1, 0.1) if t > 15.3 else None
    put(gpt(t, look_g, eye_s=1.0), 480, 560, p, wig(t, 0.7, 3))
    txt(t, 10.6, "VLM", 480, 150, 110, tilt=-2)
    if t >= 11.74:
        put(sprite("gradcap"), 450, 345, pop(t, 11.74), -14)
        sparkles(t, [(250, 380), (700, 330), (720, 720), (240, 760)])
    txt(t, 11.9, "knows everything", 480, 870, 70, kind="hand", tilt=-2)
    txt(t, 12.2, "IQ: 9000", 480, 970, 62, BLUE)
    pc = pop(t, 13.1)
    put(cheese(t), 1430, 580, pc, wig(t, 0.9, 2, 1))
    txt(t, 13.28, "ACTION HEAD", 1430, 220, 100, tilt=2)
    txt(t, 14.82, "IQ: 3", 1430, 880, 84, RED, tilt=-4)
    txt(t, 16.4, "(literally cheese)", 1430, 985, 62, kind="hand")
    if t >= 13.3:                       # a fly buzzing around the cheese
        a = t * 5
        fx, fy = 1430 + 260 * math.cos(a), 500 + 120 * math.sin(a * 1.3)
        d = D()
        ell(d, fx, fy, 9, 9, vrng("fly"), fill=BLACK, w=0)
        ell(d, fx - 9, fy - 12, 9, 6, vrng("fw1"), fill=(220, 240, 255, 255), w=2)
        ell(d, fx + 9, fy - 12, 9, 6, vrng("fw2"), fill=(220, 240, 255, 255), w=2)


def sc_fusion(t):
    txt(t, 18.0, "how do they work together??", 960, 100, 72, kind="hand", tilt=-1)
    u = ramp(t, 20.42, 21.0)
    if t < 21.0:
        j1 = thud(t, 18.92, 12, 8, 5)
        j2 = thud(t, 19.76, 12, 8, 5)
        put(gpt(t, (1, 0) if t > 20.3 else None), lerp(480, 860, u), 560, lerp(1, 0.75, u), j1 + u * 30)
        put(cheese(t, [(-1, 0), (-1, 0.3)] if t > 20.3 else None), lerp(1430, 1060, u), 580, lerp(1, 0.7, u),
            j2 - u * 30)
    hx, hy = tim_head(1350, 1060, 0.8)
    if 21.0 <= t < 22.05:
        v = ramp(t, 21.55, 22.0)
        cx, cy = lerp(960, hx, v), lerp(560, hy, v)
        sc = lerp(0.62, 0.08, v)
        a = (t - 21.0) * 14
        put(gpt(t, mode="spiral"), cx + 60 * math.cos(a), cy + 60 * math.sin(a), sc, a * 50)
        put(cheese(t, mode="spiral"), cx - 60 * math.cos(a), cy - 60 * math.sin(a), sc, -a * 50)
        if t < 21.5:
            put(sprite("boom", "FUSION!!"), 960, 560, pop(t, 21.0, 0.12) * 0.85, 5)
    if t >= 21.0 and not any(sh[0] == 21.0 for sh in SHAKES):
        shake(21.0, 0.35, 18)
        shake(22.0, 0.3, 24)
    if t >= 21.24:
        big = t >= 22.0
        put_tim(1350, 1060, pop(t, 21.24) * 0.8, thud(t, 22.0, 8), t, "tiny" if big else "normal",
                None if big else (-1, -0.5), 1.4 if big else 1.0)
        txt(t, 22.05, "?!", 1560, 380, 130, RED, tilt=8)


def sc_timer(t):
    p = pop(t, 22.56)
    put(sprite("watch"), 700, 560, p * 1.05, wig(t, 1.2, 4))
    if p >= 1:
        clock_hands(700, 560 + 30 * 1.05, 1.05, t - 22.56, 2.0)
    txt(t, 22.88, "300 ms", 1360, 470, 170, RED, stroke=6, tilt=-5)
    txt(t, 23.4, "(~3 times a second)", 1360, 660, 62, kind="hand")


def sc_look(t):
    big = ramp(t, 25.2, 25.45)
    look = (1, -0.1) if t < 25.92 else (1, 0.55)
    put(gpt(t, look if t > 24.6 else None, eye_s=1 + 0.55 * big), 430, 520, pop(t, 24.32) * 1.0, wig(t, 0.8, 3))
    put(sprite("scene_frame"), 1260, 420, pop(t, 24.76))
    if 25.3 <= t < 25.95:
        d = D()
        for i, (ex, ey) in enumerate([(430 - 60, 515), (430 + 65, 500)]):
            for k in range(6):
                u0, u1 = k / 6, k / 6 + 0.08
                x0, y0 = lerp(ex, 940, u0), lerp(ey, 420 + (i - 0.5) * 120, u0)
                x1, y1 = lerp(ex, 940, u1), lerp(ey, 420 + (i - 0.5) * 120, u1)
                line(d, [(x0, y0), (x1, y1)], vrng("beam", i, k), 8, col=YEL)
    txt(t, 25.3, "*looks*", 430, 180, 64, kind="hand")
    put(sprite("prompt_box", "pick up the ball"), 1260, 820, pop(t, 25.92))
    txt(t, 26.5, "*reads*", 430, 870, 64, kind="hand", tilt=3)


BUM = "this unemployed bum wants the round thing in the cup"
BUM_T = [29.16, 29.58, 30.2, 30.64, 31.32, 31.52, 31.72, 32.0, 32.34, 32.46]


def sc_whisper(t):
    put(gpt(t, (1, 0.2)), 420, 700, 0.85, -8 + wig(t, 1.5, 2))
    cx = lerp(2300, 1480, ramp(t, 27.4, 27.85))
    put(cheese(t, [(-1, -0.4), (-1, 0.2)] if t < 31 else None), cx, 760, 0.9, wig(t, 0.8, 2))
    if t >= 29.16:
        k = sum(1 for w in BUM_T if t >= w)
        put(sprite("bubble", BUM, k, 1300, 300, (220, 420)), 870, 380, pop(t, 29.16))
    if t >= 30.2:
        put(me(t), 960, 800, pop(t, 30.2) * 0.62)
        txt(t, 30.4, "(me)", 1110, 640, 60, kind="hand", tilt=6)
    txt(t, 31.0, "?", 1580, 520, 120, tilt=10)


def sc_think(t):
    if t < 33.95:
        u = ramp(t, 33.3, 33.9)
        put(sprite("bubble", BUM, 10, 1300, 300, (220, 420)), lerp(870, 960, u), lerp(380, 600, u),
            lerp(0.6, 0.02, u))
    mode = "spiral" if 39.4 <= t else "normal"
    put(cheese(t, mode=mode), 960, 640, 1.1, wig(t, 3 if t > 39.3 else 0.8, 3 if t > 39.3 else 2))
    if t < 38.5:
        txt(t, 34.28, "*receives prompt*", 960, 150, 64, kind="hand")
    put_tim(330, 1070, pop(t, 36.36) * 0.6, 0, t, look=(1, -0.3))
    hx, hy = tim_head(330, 1070, 0.6)
    for k, (s, col) in enumerate([("90°", BLUE), ("45°", RED), ("132°", GREEN), ("77°", PURPLE),
                                  ("open", ORANGE)]):
        t0 = 37.0 + 0.16 * k
        if t0 <= t < t0 + 0.7:
            u = ease((t - t0) / 0.7)
            x, y = lerp(hx, 960, u), lerp(hy, 620, u) - 200 * math.sin(math.pi * u)
            put(sprite("text", s, 64, col, "comic", 0, WHITE, 0.0), x, y, lerp(1, 0.3, u))
    txt(t, 36.9, "servo angles", 330, 380, 56, kind="hand")
    if t >= 39.12:
        put(sprite("nn"), 960, 290, pop(t, 39.12) * 0.9, wig(t, 4, 4))
        txt(t, 39.36, "neural network (cheese edition)", 960, 110, 60, kind="hand", tilt=-1)
        sweat(t, 1240, 470, 2)
    if t >= 39.6:
        d = D()
        for i in range(3):
            ph = ((t - 39.6) * 0.9 + i / 3) % 1
            ell(d, 1150 + i * 40 + 30 * ph, 380 - 160 * ph, 30 + 40 * ph, 26 + 30 * ph, vrng("smoke", i),
                fill=(205, 205, 205, 255), w=4, col=GREY)


def sc_moves(t):
    angs = [20, -60, 170, 90, -120, 45, 0, 200, -30, 130, 70, -90, 160, 10, -150, 110]
    for i in range(16):
        x = 120 + 112 * i
        y = 230
        rot = 0
        if i >= 8 and t >= 43.9:
            dt = t - 43.9 - 0.04 * (i - 8)
            if dt > 0:
                y += 1800 * dt * dt
                rot = 120 * dt * (1 if i % 2 else -1)
        p = pop(t, 40.44 + 0.04 * i, 0.18)
        put(sprite("card", i, angs[i]), x, y, p, rot)
        if i < 8 and t >= 42.94 + 0.08 * i:
            put(sprite("check", GREEN, 80), x + 10, y + 10, pop(t, 42.94 + 0.08 * i, 0.15))
        if i >= 8 and t >= 43.55:
            put(sprite("xmark", 90), x, y, pop(t, 43.55, 0.12), rot)
    txt(t, 40.64, "16 MOVES", 960, 450, 120, tilt=-2)
    txt(t, 43.4, "(only the first 8 lol)", 960, 570, 60, kind="hand", tilt=2)
    put(cheese(t), 1650, 930, 0.5, wig(t, 0.8, 3))
    txt(t, 41.2, "nailed it", 1360, 1000, 56, kind="hand", tilt=-4)
    if t >= 42.02:
        n = sum(1 for i in range(8) if t >= 42.94 + 0.08 * i)
        rot = 7 * (1 if n % 2 else -1) if 0 < n < 8 or t < 43.7 and n else 0
        put_tim(450, 1070, pop(t, 42.02) * 0.72, rot, t)
    if t >= 43.84:
        x = lerp(2200, 1770, ramp(t, 43.84, 44.2))
        put(gpt(t, (-1, 0), eye_s=1.5), x, 650, 0.62, 18)


def sc_loop(t):
    sp = (t - 45.66) * lerp(1.5, 6.0, ramp(t, 45.66, 47.3))
    put(sprite("loop_arrow"), 960, 540, pop(t, 45.66), -sp * 60)
    for k, im, s in [(0, gpt(t), 0.42), (1, cheese(t), 0.38), (2, tim_img(t), 0.3)]:
        a = -sp + k * 2 * math.pi / 3
        put(im, 960 + 330 * math.cos(a), 540 + 330 * math.sin(a), pop(t, 45.7 + 0.08 * k) * s)
    txt(t, 46.14, "REPEAT", 960, 520, 96, PURPLE)
    txt(t, 46.5, "x ∞", 960, 620, 80)


def sc_nothing(t):
    put(sprite("ground"), 960, 1050)
    rot = lerp(0, -40, ramp(t, 47.5, 48.0))
    rot = lerp(rot, -45, ramp(t, 48.0, 48.3))
    rot = lerp(rot, -8, ramp(t, 48.55, 49.2))
    put(sprite("ball", RED, 55), 1660, 990)
    if t >= 49.2:
        d = D()
        eye(d, 1642, 975, 14, vrng("be1"), (-0.6, -1))
        eye(d, 1678, 972, 11, vrng("be2"), (-0.6, -1))
    proud = t >= 49.48
    put_tim(820, 1050, 1.08, rot, t, look=(0.2, -1) if proud else None)
    txt(t, 48.86, "absolutely", 600, 140, 80, kind="hand", tilt=-3)
    txt(t, 49.3, "NOTHING", 600, 260, 140, RED, stroke=5, tilt=-3)
    if t >= 49.48:
        sparkles(t, [(330, 220), (880, 170), (900, 330)])
    txt(t, 49.9, "*proud*", 1250, 230, 60, kind="hand", tilt=8)


def sc_supposed(t):
    if t < 52.04:
        txt(t, 51.2, "PERFECT.", 960, 520, 180, GREEN, stroke=6, tilt=-4)
        return
    fall = ramp(t, 55.0, 55.35)
    rot = -84 * fall + thud(t, 55.35, 6, 5, 7)
    dead = t >= 55.3
    put_tim(1077 - 100 * fall, lerp(746, 820, fall), 0.95, rot, t, mode="x" if dead else "normal",
            look=(0, -0.6), pivot=(0.5, 0.5))
    if t < 54.84:
        hx, hy = tim_head(960, 1050, 0.95)
        put(sprite("halo"), hx, hy - 150, pop(t, 52.1))
        sparkles(t, [(620, 500), (1350, 380), (1380, 760), (560, 820)], 34)
        txt(t, 52.7, "how he was SUPPOSED to work", 960, 120, 76, kind="hand")
    else:
        txt(t, 54.84, "why he DIDN'T", 960, 140, 140, RED, stroke=5, tilt=-3)
    if t >= 55.35 and not any(sh[0] == 55.35 for sh in SHAKES):
        shake(55.35, 0.4, 26)


def sc_cosmos(t):
    put(groot(t, (0, -1) if t >= 58.0 else None), 960, 690, pop(t, 56.32) * 1.1, thud(t, 57.6, 6, 7, 4))
    if t >= 58.0:
        up = pop(t, 58.0, 0.3)
        gy = lerp(500, 270, up)
        d = D()
        zz = [(960 + (25 if i % 2 else -25), lerp(430, gy + 90, i / 8)) for i in range(9)]
        line(d, zz, vrng("spring"), 7, col=DGREY)
        put(gpt(t, (0, 0.3) if t < 61.5 else None), 960, gy, 0.55, wig(t, 1.2, 4))
        if t >= 61.5:
            put(sprite("crown"), 960, gy - 125, pop(t, 61.5) * 0.6, wig(t, 1.2, 4))
    txt(t, 59.24, "COSMOS REASON 2", 960, 75, 84, WHITE, stroke=6, scol=BLACK)
    put(sprite("trophy"), 1560, 600, pop(t, 60.78) * 0.95, wig(t, 0.9, 4))
    txt(t, 62.34, "#1", 380, 470, 220, GOLD, stroke=8, scol=BLACK, tilt=-6)
    txt(t, 63.0, "open VLM for\nphysical AI", 380, 700, 58, kind="hand")
    confetti(t, 62.34)


def sc_data(t):
    gulp = 1 + 0.15 * math.exp(-4 * max(0, t - 69.0)) * math.sin(12 * max(0, t - 69.0)) if t > 69 else 1
    put(groot(t, (1, 0.2)), 450, 640, pop(t, 66.44) * 0.9 * gulp)
    rng = np.random.default_rng(3)
    for i in range(70):
        v = rng.uniform(0, 1) ** 0.8
        x = 1350 + rng.uniform(-1, 1) * 380 * (1 - v)
        y = 950 - v * 520
        put(sprite("paper", i % 6), x, y, pop(t, 67.8 + 0.02 * i, 0.15) * 0.8, rng.uniform(-30, 30))
    txt(t, 68.68, "ROBOT DATA", 1350, 180, 96)
    txt(t, 68.9, "(a LOT)", 1350, 290, 64, kind="hand")
    if t >= 69.0:
        arrow(D(), (1000, 700), (720, 660), vrng("a1"), w=12)


def panel_img(kind, t):
    im = Image.new("RGBA", (520, 380), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    rng = vrng("panel", kind)
    if kind == 0:
        cols = [(255, 190, 210), (180, 220, 255), (200, 255, 190), (255, 240, 160), (220, 190, 255)]
        d.rectangle([0, 0, 520, 380], fill=cols[int(t / 0.3) % len(cols)])
        for k in range(0, 600, 70):      # stripes on every other colour
            if int(t / 0.3) % 2:
                line(d, [(k, 0), (k - 200, 380)], rng, 10, col=(255, 255, 255, 255))
    bcol = [RED, BLUE, YEL, PURPLE, GREEN, PINK][int(t / 0.3) % 6] if kind == 2 else RED
    b = sprite("ball", bcol, 50)
    im.alpha_composite(b, (150 - b.width // 2, 280 - b.height // 2))
    c = sprite("cup").resize((170, 170))
    im.alpha_composite(c, (300, 190))
    if kind == 1:
        bl = sprite("bulb").resize((110, 150))
        im.alpha_composite(bl, (205, 5))
        if int(t / 0.25) % 2:
            dark = Image.new("RGBA", im.size, (0, 0, 40, 150))
            im = Image.alpha_composite(im, dark)
            d = ImageDraw.Draw(im)
    poly(d, [(6, 6), (514, 6), (514, 374), (6, 374)], rng, w=9)
    return im


def sc_ez(t):
    put(groot(t, (0, 0.5)), 960, 300, 0.65, wig(t, 0.7, 3))
    if t >= 70.3:
        put(sprite("shades"), 960 - 18, 300 - 80, pop(t, 70.3) * 0.62, -4)
    txt(t, 70.56, "ez", 1260, 230, 110, tilt=10)
    for k, (t0, label) in enumerate([(73.54, "new background"), (74.84, "new lighting"),
                                     (75.88, "new ball colour")]):
        x = 360 + 600 * k
        p = pop(t, t0)
        if p:
            put(panel_img(k, t), x, 760, p * 0.95)
            txt(t, t0 + 0.1, label, x, 990, 56, kind="hand")
            if t >= t0 + 0.7:
                put(sprite("check", GREEN, 120), x + 200, 620, pop(t, t0 + 0.7))


def sc_eyes(t):
    big = ramp(t, 80.4, 80.65)
    put(gpt(t, None if t < 80.4 else (math.sin(t * 9) * 0.8, 0), eye_s=1 + 0.6 * big), 650, 560, 1.15,
        wig(t, 0.8, 3))
    txt(t, 79.68, "chat gpt", 1470, 340, 120, TEAL, tilt=-3)
    txt(t, 80.4, "WITH EYES", 1470, 560, 130, RED, stroke=5, tilt=2)
    if t >= 80.6:
        d = D()
        arrow(d, (1060, 650), (870, 560), vrng("ae1"), w=11)
        arrow(d, (1060, 720), (820, 660), vrng("ae2"), w=11)


def sc_understand(t):
    look = (1, 0.45) if t >= 83.76 else None
    put(gpt(t, look), 400, 620, 0.85, wig(t, 0.8, 3))
    put(sprite("ball", RED, 55), 1200, 780)
    put(sprite("block", GREEN), 1600, 780)
    put(sprite("ground", 1300), 1300, 860)
    if 82.86 <= t < 84.2:
        k = sum(1 for w in [82.86, 83.3, 83.44, 83.56, 83.76] if t >= w)
        put(sprite("bubble", "pick up the round thing", k, 900, 220, (700, -40)), 1150, 260, pop(t, 82.86))
    elif t >= 84.2:
        k = sum(1 for w in [84.36, 84.64, 84.84, 85.04] if t >= w)
        put(sprite("bubble", "get that red one", k, 800, 220, (650, -40)), 1150, 260, pop(t, 84.2))
    if t >= 83.9:
        arrow(D(), (600, 640), (1110, 760), vrng("a2"), w=10, col=TEAL)
        put(sprite("check", GREEN, 110), 1240, 640, pop(t, 83.95))
    if t >= 84.95:
        put(sprite("check", GREEN, 110), 1300, 650, pop(t, 84.95))
    txt(t, 85.3, "big brain", 400, 300, 66, kind="hand", tilt=-5)


def sc_problem(t):
    fill(RED if int((t - 86.32) / 0.12) % 2 == 0 and t < 87.2 else WHITE)
    put(sprite("warn"), 960, 470, pop(t, 86.32, 0.15) * 1.3, wig(t, 6, 5))
    txt(t, 86.7, "PROBLEM", 960, 880, 170, BLACK, stroke=7, tilt=-2)
    if not any(sh[0] == 86.32 for sh in SHAKES):
        shake(86.32, 0.8, 26)


def sc_stranger(t):
    put(groot(t, (1, 0.4)), 500, 640, pop(t, 87.74) * 0.9, wig(t, 0.7, 3))
    spin = ramp(t, 91.3, 92.9) * 720
    if spin and t < 92.95:
        put_tim(1485, 790, 0.85, spin, t, mode="spiral", pivot=(0.5, 0.55))
    else:
        put_tim(1380, 1050, pop(t, 87.74) * 0.85, 0, t, mode="spiral" if t >= 92.9 else "normal",
                look=(-1, 0) if t < 90 else None)
    if 89.9 <= t < 91.04:
        k = sum(1 for w in [89.9, 90.1, 90.3, 90.5] if t >= w)
        put(sprite("bubble", "who is this guy??", k, 640, 190, (-120, 230)), 760, 200, pop(t, 89.9))
    for i, (x, y, t0) in enumerate([(380, 230, 91.5), (640, 260, 91.8), (520, 160, 92.1), (300, 380, 92.4)]):
        txt(t, t0, "?", x, y, 120 + 20 * (i % 2), RED if i % 2 else BLACK, tilt=15 - i * 10)
    txt(t, 92.04, "no idea.", 500, 960, 90, tilt=-3)


def sc_demos(t):
    put(me(t), 330, 640, pop(t, 93.8) * 0.85)
    if t >= 94.78:
        rec_dot(t)
    n = int(clamp((t - 95.3) / 4.2) * 18)
    for i in range(n):
        jx = ((i * 37) % 23 - 11) * 2
        put(sprite("tape"), 960 + jx, 950 - i * 38, 0.8, ((i * 53) % 13 - 6))
    if t >= 95.14:
        cnt = int(clamp((t - 95.14) / 4.4) * 150)
        put(sprite("text", f"demos: {cnt}", 70, BLACK, "comic", 0, WHITE, 0.0), 960, 140)
    put(groot(t, (-1, 0.2) if t < 97.6 else (0, -1)), 1580, 650, pop(t, 96.12) * 0.72)
    if t >= 96.3:
        arrow(D(), (1110, 640), (1400, 650), vrng("a3"), w=11)
    if t >= 97.6:
        put(sprite("bulb"), 1580, 330, pop(t, 97.6) * 0.75, wig(t, 1, 5))
    put(sprite("prompt_box", "pick up the ball", 620), 1520, 990, pop(t, 99.12) * 0.85)


def sc_two(t):
    if t < 104.42:
        put(me(t), 500, 640, pop(t, 100.88) * 0.9)
        txt(t, 101.76, "REAL.", 1300, 300, 120, BLUE, tilt=-3)
        txt(t, 102.46, "PHYSICAL.", 1300, 450, 120, RED, tilt=2)
        txt(t, 102.94, "LIVE.", 1300, 600, 120, GREEN, tilt=-2)
        return
    put(sprite("warn"), 260, 230, pop(t, 104.5) * 0.6, wig(t, 4, 6))
    put(sprite("warn"), 1680, 860, pop(t, 104.6) * 0.6, wig(t, 4, 6, 1))
    txt(t, 104.42, "2", 700, 520, 420, RED, stroke=10, tilt=-8)
    txt(t, 104.68, "MORE\nPROBLEMS", 1250, 540, 120, tilt=3)
    if not any(sh[0] == 104.42 for sh in SHAKES):
        shake(104.42, 0.5, 26)


def sc_claude(t):
    txt(t, 106.08, "PROBLEM #1", 960, 85, 90, RED, tilt=-2)
    put(claude(t, (1, 0.3) if t > 107.5 else None), 430, 560, pop(t, 106.76), wig(t, 0.5, 6))
    txt(t, 106.9, "Claude", 430, 860, 76, ORANGE, kind="hand")
    if t >= 107.54:
        k = min(8, 1 + int((t - 107.54) / 0.4))
        put(sprite("code_panel", k), 1290, 530, pop(t, 107.54) * 0.95)
    if t >= 108.12:
        for i, s in enumerate(["θ", "π", "cos", "√", "atan2", "sin"]):
            a = (t - 108.12) * 1.5 + i * 2 * math.pi / 6
            put(sprite("text", s, 54, [BLUE, RED, GREEN, PURPLE, BLACK, ORANGE][i], "comic", 0, WHITE, 0.0),
                430 + 270 * math.cos(a), 560 + 185 * math.sin(a), pop(t, 108.12 + 0.07 * i))
    if t >= 110.94:
        put(sprite("text", "GOOD DATA", 100, GREEN, "comic", 6, WHITE, -8.0), 1250, 920, pop(t, 110.94, 0.15))
        put(sprite("check", GREEN, 110), 1620, 900, pop(t, 111.2))
        sparkles(t, [(900, 870), (1700, 990)], 30, GOLD)


def sc_calibrate(t):
    txt(t, 112.64, "PROBLEM #2", 960, 85, 90, RED, tilt=2)
    put(me(t, tired=t >= 117.8), 430, 640, pop(t, 112.4) * 0.9)
    if t >= 114.16:
        sweat(t, 430, 470, 2)
        put(sprite("watch", True), 1520, 330, pop(t, 114.16) * 0.7, wig(t, 5, 4))
        clock_hands(1520, 330 + 30 * 0.7, 0.7, t - 114.16, 6.0)
    txt(t, 114.8, "2-3 HOURS", 900, 260, 120, RED, stroke=5, tilt=-4)
    txt(t, 116.3, "+ building Timothy", 900, 400, 66, kind="hand")
    jit = wig(t, 9, 3) if t >= 117.8 else 0
    put_tim(1450, 1060, pop(t, 116.5) * 0.6, jit, t)
    if t >= 117.8:
        put(sprite("protractor"), 1090, 860, pop(t, 117.8) * 0.9, wig(t, 1, 8))
        dots = "." * (1 + int((t - 117.8) * 3) % 3)
        put(sprite("text", "calibrating" + dots, 62, BLACK, "hand", 0, WHITE, 0.0), 820, 990)


def sc_ohwait(t):
    txt(t, 120.58, "oh wait.", 960, 420, 170, kind="hand", tilt=-4)
    txt(t, 121.72, "I FORGOT", 960, 660, 120, RED, tilt=3)


def sc_record(t):
    rec_dot(t)
    tired = t >= 127.08
    put(me(t, tired=tired), 380, 650, pop(t, 122.84) * 0.85)
    if tired:
        sweat(t, 380, 480, 3, 0.4)
    rot = -22 * (0.5 - 0.5 * math.cos(2 * math.pi * (t - 122.84) * 0.9))
    put_tim(1150, 1060, 0.8, rot, t)
    cnt = 1 + int(max(0, t - 122.84) * 21)
    put(sprite("text", f"demo #{min(cnt, 150)}", 74, BLACK, "comic", 0, WHITE, 0.0), 1600, 480)
    put(sprite("watch", True), 1690, 290, pop(t, 123.0) * 0.55, wig(t, 5, 4))
    clock_hands(1690, 290 + 30 * 0.55, 0.55, t - 123.0, 8.0)
    txt(t, 125.04, "ANOTHER 2-3 HOURS", 900, 140, 100, RED, stroke=5, tilt=-2)
    txt(t, 127.9, "(send help)", 380, 990, 60, kind="hand", tilt=-4)


PATHS = [  # end point, sign text, sign position
    ((1500, 170), "buy an arm\nGR00T knows", (1700, 170)),
    ((1500, 370), "simulation only", (1700, 370)),
    ((1500, 560), "touch grass", (1700, 560)),
    ((1500, 750), "give up", (1700, 750)),
    ((1500, 950), "record demos\nby hand", (1700, 950)),
]


def path_pts(k):
    (ex, ey), _, _ = PATHS[k]
    sx, sy = 330, 560
    return [(lerp(sx, ex, u) + 40 * math.sin(u * 7 + k), lerp(sy, ey, u * u * (3 - 2 * u)) +
             (60 * math.sin(u * 9 + k * 2) if k == 4 else 0)) for u in np.linspace(0, 1, 30)]


def sc_paths(t):
    chosen = t >= 134.66
    d = D()
    for k in range(5):
        if t < 130.78 + 0.3 * k:
            continue
        bad = k == 4
        col = RED if (bad and chosen) else ((200, 200, 200, 255) if chosen else BLACK)
        line(d, path_pts(k), vrng("path", k), 14 if (bad and chosen) else 8, col=col)
        put(sprite("sign", PATHS[k][1]), *PATHS[k][2], pop(t, 130.78 + 0.3 * k) * (1.1 if bad and chosen else 0.9))
    if chosen:
        pts = path_pts(4)
        for u, kind in [(0.35, "skull"), (0.55, "clock"), (0.75, "skull")]:
            x, y = pts[int(u * 29)]
            if kind == "skull":
                put(sprite("skull"), x, y - 70, pop(t, 135.2 + u) * 0.7)
            else:
                put(sprite("watch", True), x, y - 90, pop(t, 135.2 + u) * 0.3)
        u = clamp((t - 135.3) / 4.5)
        i = u * 29
        x0, y0 = pts[int(i)]
        x1, y1 = pts[min(29, int(i) + 1)]
        f = i - int(i)
        put(me(t, tired=u > 0.5), lerp(x0, x1, f), lerp(y0, y1, f) - 90, 0.38, wig(t, 2, 5))
    else:
        put(me(t), 230, 470, pop(t, 129.88) * 0.55)
    txt(t, 130.78, "all the possible paths:", 620, 90, 70, kind="hand")
    txt(t, 138.42, "bye bye daily life", 760, 1000, 64, kind="hand", tilt=-3)


def sc_twenty(t):
    txt(t, 140.68, "could have just spent", 700, 200, 70, kind="hand")
    put(sprite("bill"), 700, 520 + wig(t, 0.7, 15), pop(t, 141.32) * 1.15, wig(t, 0.5, 5))
    if t >= 141.32:
        sparkles(t, [(400, 380), (1010, 400), (960, 690), (420, 700)], 30, GREEN)
    for i, (x, y) in enumerate([(1400, 330), (1560, 560), (1400, 790)]):
        put(sprite("servo"), x, y, pop(t, 142.74 + 0.1 * i) * 0.9, wig(t, 1.5, 6, i))
    txt(t, 143.36, "better servos", 1500, 980, 66, kind="hand")


def sc_smug(t):
    put(sprite("smug_arm"), 1150, 600, pop(t, 144.84) * 1.0, wig(t, 0.6, 3))
    txt(t, 145.2, "an arm GR00T already knows", 1150, 1010, 58, kind="hand")
    put(groot(t, mode="heart"), 1700, 760, pop(t, 145.72) * 0.45, wig(t, 1.3, 5))
    if t >= 146.12:
        k = sum(1 for w in [146.12, 146.3, 146.5, 146.7, 146.85] if t >= w)
        put(sprite("bubble", "oh I know this guy", k, 720, 170, (600, 465)), 1450, 300, pop(t, 146.12))
    tear = t >= 147.0
    put_tim(330, 1070, 0.6, 0, t, look=(1, -0.3))
    if tear:
        hx, hy = tim_head(330, 1070, 0.6)
        ph = ((t - 147.0) % 0.8) / 0.8
        ell(D(), hx - 20, hy + 40 + 80 * ph, 9, 14, vrng("tear"), fill=LBLUE, w=3)
    txt(t, 147.2, "...", 470, 640, 90)


SCENES = [
    (0.0, 4.66, sc_title), (4.66, 10.46, sc_jam), (10.46, 17.96, sc_versus), (17.96, 22.56, sc_fusion),
    (22.56, 24.28, sc_timer), (24.28, 27.3, sc_look), (27.3, 33.26, sc_whisper), (33.26, 40.44, sc_think),
    (40.44, 45.66, sc_moves), (45.66, 47.36, sc_loop), (47.36, 51.2, sc_nothing), (51.2, 56.32, sc_supposed),
    (56.32, 66.44, sc_cosmos), (66.44, 69.48, sc_data), (69.48, 77.98, sc_ez), (77.98, 81.26, sc_eyes),
    (81.26, 86.32, sc_understand), (86.32, 87.74, sc_problem), (87.74, 93.8, sc_stranger),
    (93.8, 100.88, sc_demos), (100.88, 105.94, sc_two), (105.94, 112.32, sc_claude),
    (112.32, 120.58, sc_calibrate), (120.58, 122.84, sc_ohwait), (122.84, 129.88, sc_record),
    (129.88, 140.18, sc_paths), (140.18, 143.96, sc_twenty), (143.96, DURATION + 1, sc_smug),
]


def render(f):
    """Frame f -> raw RGB bytes."""
    global FR, CAN
    FR = f
    t = f / FPS
    CAN = Image.new("RGB", (W, H), (255, 255, 255))
    for t0, t1, fn in SCENES:
        if t0 <= t < t1:
            fn(t)
            break
    dx = dy = 0.0
    for t0, dur, amp in SHAKES:
        if t0 <= t < t0 + dur:
            k = amp * (1 - (t - t0) / dur)
            rng = np.random.default_rng(f)
            dx += rng.uniform(-k, k)
            dy += rng.uniform(-k, k)
    if dx or dy:
        moved = Image.new("RGB", (W, H), (255, 255, 255))
        moved.paste(CAN, (int(dx), int(dy)))
        CAN = moved
    return CAN.tobytes()


def _register_shakes():
    """Shakes are added the first time a scene reaches them; pre-run each scene's
    shake times so every worker process knows them from frame 0."""
    for t0, dur, amp in [(6.42, 0.45, 28), (21.0, 0.35, 18), (22.0, 0.3, 24), (55.35, 0.4, 26),
                         (86.32, 0.8, 26), (104.42, 0.5, 26)]:
        if not any(s[0] == t0 for s in SHAKES):
            SHAKES.append((t0, dur, amp))


_register_shakes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=DURATION)
    ap.add_argument("--stills", help="comma-separated times -> stills/*.png")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    if a.stills:
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        for s in a.stills.split(","):
            render(int(round(float(s) * FPS)))
            CAN.save(os.path.join(HERE, "stills", f"t{float(s):06.2f}.png"))
        print("stills saved")
        return

    import imageio_ffmpeg
    from multiprocessing import Pool

    full = a.t0 == 0 and a.t1 >= DURATION
    out = a.out or os.path.join(HERE, "timothy_explainer.mp4" if full else f"preview_{a.t0:g}-{a.t1:g}.mp4")
    frames = range(int(a.t0 * FPS), int(min(a.t1, DURATION) * FPS))
    audio = os.path.join(HERE, "voice.m4a")
    if not full:                          # cut the audio to match the piece
        import subprocess
        audio = os.path.join(HERE, "_piece.m4a")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(a.t0),
                        "-t", str(len(frames) / FPS), "-i", os.path.join(HERE, "voice.m4a"), "-c:a", "aac",
                        audio], check=True)
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                    audio_path=audio, audio_codec="aac", output_params=["-shortest"],
                                    ffmpeg_log_level="error")
    w.send(None)
    with Pool(a.jobs) as pool:
        for i, buf in enumerate(pool.imap(render, frames, chunksize=6)):
            w.send(buf)
            if i % 300 == 0:
                print(f"{frames[0] / FPS + i / FPS:6.1f} s", flush=True)
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
