"""Streamlit UI. Run: streamlit run app.py"""
from __future__ import annotations

import dataclasses
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from faces import FaceEngine
from ocr import make_ocr
from pipeline import plain_explanation, run_pipeline

st.set_page_config(page_title="Identity check (prototype)", layout="wide")

BADGE = {"green": "🟢 Quality OK", "amber": "🟠 OK, a check was skipped", "red": "🔴 Retake needed"}
RESULT_COLOURS = {"MATCH": "background-color:#c8f0c8", "MISMATCH": "background-color:#f6c6c6",
                  "UNCERTAIN": "background-color:#ffe3a3"}
VERDICT_COLOURS = {"PASS": "green", "FAIL": "red", "NEEDS HUMAN REVIEW": "orange", "RETAKE NEEDED": "blue"}


@st.cache_resource(show_spinner="Downloading face models on first run...")
def get_faces() -> FaceEngine:
    return FaceEngine()


def decode(file) -> np.ndarray | None:
    """Decode an upload or camera capture straight from memory. Nothing is written to disk."""
    if file is None:
        return None
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Thresholds")
sb.caption("Cosine similarity: at or above the high value = MATCH, below the low value = MISMATCH, between = UNCERTAIN.")
m = config.MATCH
q = config.QUALITY
wm = sb.slider("Webcam vs webcam: match at", 0.0, 1.0, m.webcam_match, 0.01)
wl = sb.slider("Webcam vs webcam: mismatch below", 0.0, 1.0, m.webcam_mismatch, 0.01)
im_ = sb.slider("ID vs ID: match at", 0.0, 1.0, m.id_match, 0.01)
il = sb.slider("ID vs ID: mismatch below", 0.0, 1.0, m.id_mismatch, 0.01)
xm = sb.slider("Webcam vs ID: match at", 0.0, 1.0, m.id_webcam_match, 0.01)
xl = sb.slider("Webcam vs ID: mismatch below", 0.0, 1.0, m.id_webcam_mismatch, 0.01)
nm = sb.slider("Name: match at (0-100)", 0, 100, int(m.name_match))
nl = sb.slider("Name: mismatch below (0-100)", 0, 100, int(m.name_mismatch))
sb.header("Quality")
blur = sb.slider("Blur: minimum Laplacian variance", 5, 300, int(q.blur_min))
glare = sb.slider("Glare: max share of near-white pixels (%)", 0.5, 20.0, q.glare_max_fraction * 100, 0.5)
sb.caption(f"OCR backend: {config.OCR_BACKEND} (set IDCHECK_OCR=tesseract for local-only).")

match_t = dataclasses.replace(m, webcam_match=wm, webcam_mismatch=min(wl, wm), id_match=im_, id_mismatch=min(il, im_),
                              id_webcam_match=xm, id_webcam_mismatch=min(xl, xm),
                              name_match=float(nm), name_mismatch=float(min(nl, nm)))
quality_t = dataclasses.replace(q, blur_min=float(blur), glare_max_fraction=glare / 100.0)

# ------------------------------------------------------------------ header + privacy
st.title("Identity check: work-from-home agents")
if config.OCR_BACKEND == "claude":
    st.warning("Privacy: ID images are sent to the Anthropic Claude API to read the ID text and to check for a finger "
               "over the ID. Face comparison runs locally. Nothing is stored; images stay in memory. "
               "Set IDCHECK_OCR=tesseract to keep everything local.")
else:
    st.info("Local mode: ID text is read with Tesseract on this machine. The finger/thumb check is skipped in this mode.")

# ------------------------------------------------------------------ inputs
st.subheader("Inputs")
c1, c2 = st.columns(2)
id_files = [c1.file_uploader("ID upload 1", type=["jpg", "jpeg", "png"], key="id1"),
            c2.file_uploader("ID upload 2", type=["jpg", "jpeg", "png"], key="id2")]
cam_files = []
cols = st.columns(3)
for col, label in zip(cols, "ABC"):
    with col:
        mode = st.radio(f"Webcam {label}", ["Upload", "Camera"], horizontal=True, key=f"mode{label}")
        if mode == "Upload":
            cam_files.append(st.file_uploader(f"Webcam image {label}", type=["jpg", "jpeg", "png"], key=f"up{label}"))
        else:
            cam_files.append(st.camera_input(f"Webcam image {label}", key=f"cam{label}"))

if not st.button("Run identity check", type="primary"):
    st.stop()

id_imgs = [decode(f) for f in id_files]
cam_imgs = [decode(f) for f in cam_files]

try:
    ocr = make_ocr(config.OCR_BACKEND)
except Exception as e:
    ocr = None
    st.error(f"OCR unavailable ({e}). Text matching and the finger check are skipped, so results will go to review.")

with st.spinner("Running checks..."):
    res = run_pipeline(id_imgs, cam_imgs, get_faces(), ocr, match_t, quality_t)

# ------------------------------------------------------------------ 1. quality
st.header("1. Image quality")
st.caption("Quality is reported separately from matching. A poor image is never counted as a mismatch.")
imgs = id_imgs + cam_imgs
cols = st.columns(5)
for col, qr, im in zip(cols, res.qualities, imgs):
    with col:
        st.markdown(f"**{qr.image_id}**  \n{BADGE[qr.status]}")
        if im is not None:
            st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), use_container_width=True)
        for p in qr.retake_prompts:
            st.markdown(f":red[{p}]")
        with st.expander("Metric details"):
            rows = [{"Metric": mt.name, "Value": "" if mt.value is None else round(mt.value, 3), "Rule": mt.threshold,
                     "Result": "skipped" if mt.passed is None else ("pass" if mt.passed else "FAIL"), "Note": mt.detail}
                    for mt in qr.metrics]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ 2. comparisons
st.header("2. Comparisons")
if res.reference_sentence:
    st.info(res.reference_sentence)
if res.comparisons:
    df = pd.DataFrame([{"Pair": c.pair, "Type": c.kind, "Score": "" if c.score is None else round(c.score, 3),
                        "Result": c.result, "Detail": c.detail} for c in res.comparisons])
    st.dataframe(df.style.map(lambda v: RESULT_COLOURS.get(v, ""), subset=["Result"]), hide_index=True, use_container_width=True)
else:
    st.write("No comparisons yet: fix the images flagged above first.")
if res.id_fields:
    with st.expander("Text read from the IDs"):
        st.json(res.id_fields)
for n in res.notes:
    st.warning(n)

# ------------------------------------------------------------------ 3. verdict
st.header("3. Overall verdict")
v = res.verdict
st.markdown(f"## :{VERDICT_COLOURS[v.verdict]}[{v.verdict}]")
for flag in v.special_flags:
    st.error(flag)
if v.retake_prompts:
    for p in v.retake_prompts:
        st.markdown(f":red[{p}]")
with st.spinner("Writing summary..."):
    st.write(plain_explanation(res, ocr))
