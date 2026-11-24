"""Step 2 helpers: YuNet detection and SFace embeddings (OpenCV only)."""
from __future__ import annotations

import os
import urllib.request
from typing import List, Optional

import cv2
import numpy as np

import config


def ensure_models(model_dir: str = config.MODEL_DIR) -> tuple[str, str]:
    """Download the two ONNX models on first run. Returns (yunet_path, sface_path)."""
    os.makedirs(model_dir, exist_ok=True)
    out = []
    for fname, url in ((config.YUNET_FILE, config.YUNET_URL), (config.SFACE_FILE, config.SFACE_URL)):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 100_000:
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        out.append(path)
    return out[0], out[1]


class FaceEngine:
    def __init__(self, model_dir: str = config.MODEL_DIR, score_threshold: float = config.DETECT_SCORE_THRESHOLD):
        yunet, sface = ensure_models(model_dir)
        self._yunet_path = yunet
        self._score = score_threshold
        self._det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), score_threshold, 0.3, 5000)
        self._rec = cv2.FaceRecognizerSF.create(sface, "")

    # -- detection
    def detect(self, img: np.ndarray) -> np.ndarray:
        """Returns an (N,15) float array in original-image pixels: x,y,w,h, 5 landmarks, score."""
        h, w = img.shape[:2]
        scale = min(1.0, config.DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._det.detect(small)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= scale
        return faces

    # -- portrait crop from an ID
    def crop_portrait(self, id_img: np.ndarray, face: Optional[np.ndarray] = None, margin: float = 0.6) -> Optional[np.ndarray]:
        """Crop the portrait region (face plus margin), upscaled to a workable size."""
        if face is None:
            faces = self.detect(id_img)
            if len(faces) == 0:
                return None
            face = max(faces, key=lambda f: f[2] * f[3])
        h, w = id_img.shape[:2]
        x, y, bw, bh = face[:4]
        x0, y0 = int(max(0, x - margin * bw)), int(max(0, y - margin * bh))
        x1, y1 = int(min(w, x + bw * (1 + margin))), int(min(h, y + bh * (1 + margin)))
        crop = id_img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        if crop.shape[1] < 200:
            s = 200.0 / crop.shape[1]
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return crop

    # -- embeddings
    def embed_face(self, img: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self._rec.alignCrop(img, face.astype(np.float32))
        return self._rec.feature(aligned)

    def embed_largest(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect(img)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        return self.embed_face(img, face)

    def embed_portrait(self, id_img: np.ndarray, face: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Crop the portrait first, then re-detect on the crop for a clean alignment."""
        crop = self.crop_portrait(id_img, face)
        if crop is None:
            return None
        emb = self.embed_largest(crop)
        if emb is None and face is not None:
            emb = self.embed_face(id_img, face)
        return emb

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
