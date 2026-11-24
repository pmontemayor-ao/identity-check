"""Synthetic images and a fake face engine so quality tests need no models or real photos."""
import cv2
import numpy as np

W, H = 800, 600


def background(value=90):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), value, np.uint8)
    noise = rng.integers(-6, 7, (H, W, 1)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def id_card(margin=100, card_value=225, text=True):
    """A card-like rectangle with text-ish detail on a darker table."""
    img = background(60)
    cv2.rectangle(img, (margin, margin), (W - margin, H - margin), (card_value,) * 3, -1)
    if text:
        for i in range(8):
            y = margin + 60 + i * 38
            cv2.putText(img, f"NAME FIELD {i} ABCDEFG 12345", (margin + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.rectangle(img, (margin + 20, margin + 40), (margin + 170, margin + 230), (120, 110, 100), -1)
    return img


def webcam(brightness=140):
    img = background(brightness)
    cv2.circle(img, (W // 2, H // 2), 120, (max(0, brightness - 10),) * 3, -1)
    cv2.putText(img, "texture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for i in range(0, W, 20):
        cv2.line(img, (i, 0), (i, H), (brightness + 30,) * 3, 1)
    return img


class FakeFaces:
    """Returns a configurable face row (x,y,w,h, 5 landmarks, score)."""

    def __init__(self, faces=None):
        self.faces = faces if faces is not None else [self.row(300, 180, 200, 240)]

    @staticmethod
    def row(x, y, w, h, score=0.95):
        lm = [x + .3 * w, y + .35 * h, x + .7 * w, y + .35 * h, x + .5 * w, y + .55 * h, x + .35 * w, y + .75 * h, x + .65 * w, y + .75 * h]
        return np.array([x, y, w, h, *lm, score], np.float32)

    def detect(self, img):
        return np.array(self.faces, np.float32).reshape(-1, 15)
