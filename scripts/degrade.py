"""Make labelled degraded variants of a clean image.

    python scripts/degrade.py clean.jpg out_dir [--kind id|webcam]

Writes out_dir/<stem>__<label>.jpg for: blur, dark, backlight, glare, edge_crop, thumb, partial_face.
The functions are importable so tests and evaluate.py can build cases in memory.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def blur(img, sigma=9.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def darken(img, factor=0.2):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def backlight(img, face_box=None):
    """Bright background gradient and a darkened subject. face_box = (x, y, w, h); default centre ellipse."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if face_box is None:
        cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)
    else:
        x, y, bw, bh = face_box
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 15).astype(np.float32)[..., None] / 255.0
    grad = np.linspace(1.0, 0.7, w, dtype=np.float32)[None, :, None]
    bright = np.clip(img.astype(np.float32) * 0.4 + 215 * grad, 0, 255)
    dark = img.astype(np.float32) * 0.3
    return np.clip(mask * dark + (1 - mask) * bright, 0, 255).astype(np.uint8)


def glare(img, centre=None, radius=None):
    h, w = img.shape[:2]
    centre = centre or (int(w * 0.6), int(h * 0.45))
    radius = radius or int(min(h, w) * 0.28)
    overlay = img.copy()
    cv2.circle(overlay, centre, radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 6)
    return cv2.addWeighted(overlay, 0.95, img, 0.05, 0)


def edge_crop(img, fraction=0.18):
    """Cut off the left edge so a document corner leaves the frame."""
    w = img.shape[1]
    return img[:, int(w * fraction):].copy()


def thumb(img, side="bottom"):
    """Skin-coloured thumb-shaped ellipse entering from the bottom edge, covering the lower middle."""
    h, w = img.shape[:2]
    out = img.copy()
    cv2.ellipse(out, (int(w * 0.35), h), (int(w * 0.09), int(h * 0.38)), 0, 180, 360, (120, 150, 205), -1)
    cv2.ellipse(out, (int(w * 0.35), int(h * 0.64)), (int(w * 0.045), int(h * 0.03)), 0, 180, 360, (140, 170, 220), -1)
    return out


def partial_face(img, face_box=None, fraction=0.5):
    """Shift so the frame cuts through the face. face_box = (x, y, w, h); default the image centre."""
    h, w = img.shape[:2]
    if face_box is None:
        cut = int(w * 0.5)
    else:
        x, y, bw, bh = face_box
        cut = int(x + fraction * bw)
    return img[:, cut:].copy()


VARIANTS = {"blur": blur, "dark": darken, "backlight": backlight, "glare": glare,
            "edge_crop": edge_crop, "thumb": thumb, "partial_face": partial_face}
APPLIES = {"id": ["blur", "dark", "glare", "edge_crop", "thumb"],
           "webcam": ["blur", "dark", "backlight", "partial_face"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("outdir")
    ap.add_argument("--kind", choices=["id", "webcam", "all"], default="all")
    a = ap.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    names = list(VARIANTS) if a.kind == "all" else APPLIES[a.kind]
    for n in names:
        p = os.path.join(a.outdir, f"{stem}__{n}.jpg")
        cv2.imwrite(p, VARIANTS[n](img))
        print(p)


if __name__ == "__main__":
    main()
