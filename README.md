# Identity check prototype (hackathon)

Compares 2 ID images and 3 webcam images for a work-from-home call centre agent. Image quality is reported
separately from matching. A poor image gives a specific retake prompt and is skipped. It is never counted as a mismatch.

**This is a prototype. Thresholds are starting values, not calibrated. Do not use it as the only identity control.**

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # not needed with IDCHECK_OCR=tesseract
streamlit run app.py
```

The two ONNX models (YuNet face detector, SFace recognizer, about 39 MB) download into `models/` on first run.
Run the tests with `python -m pytest -q tests` (the face tests skip themselves if the models cannot be downloaded).

Settings (`config.py`, environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Required for the Claude backend |
| `IDCHECK_OCR` | `claude` | `claude` or `tesseract` (local, needs the `tesseract` binary) |
| `IDCHECK_CLAUDE_MODEL` | `claude-sonnet-4-5` | Vision model for ID text, finger check, and the summary |
| `IDCHECK_MODEL_DIR` | `./models` | Where the ONNX models live |

## Pipeline

1. **Quality gate** (`quality.py`) on every image. Each metric, its rule, and pass/fail is recorded.
   Blur (Laplacian variance), brightness, backlighting (face vs background), glare (IDs), cropping (IDs: four corners
   found with a margin), obstruction (IDs: asked of the vision model), face present / single / big enough / landmarks visible.
   Each failure maps to a retake prompt in `RETAKE_PROMPTS`.
2. **Webcam vs webcam**: A-B, A-C, B-C, SFace cosine similarity.
3. **ID vs ID**: crop each portrait, compare faces. Extract name, date of birth, ID number, expiry. Name is fuzzy
   (rapidfuzz `token_sort_ratio`), date of birth and ID number are exact after normalising. The reference portrait is
   the one from the ID with the higher quality score (only IDs that passed quality are eligible), and the UI says which and why.
4. **Webcam vs ID reference**: each passing webcam image against the reference portrait.

Every comparison is MATCH, MISMATCH, or UNCERTAIN with its score (`compare.classify`: at or above high = match, below
low = mismatch, between = uncertain). Thresholds live in `config.py` with comments, and the sidebar sliders change them live.

### Verdict (`decide.overall_verdict`)

1. Any image failed quality: **RETAKE NEEDED**, with one prompt per failure.
2. Else any MISMATCH: **FAIL**.
3. Else any UNCERTAIN (or nothing to compare): **NEEDS HUMAN REVIEW**. Uncertain is never auto-rejected.
4. Else **PASS**.

"Webcam images match each other but not the ID" is flagged whenever all webcam pairs match and no webcam image matches the
reference portrait (mismatch or uncertain). The verdict still follows the rules above.

## Case matrix

| Case | Setup | Expected |
|---|---|---|
| Same person | IDs and webcams all the same person, IDs agree | PASS |
| Different person | IDs the same person, webcams someone else | FAIL (or REVIEW if the score lands in the uncertain band) |
| Mismatched IDs | The two IDs show different people / names / numbers | FAIL, reference portrait sentence says the IDs disagree |
| Webcam-consistent, ID different | Webcams match each other, ID is someone else | FAIL or REVIEW, with the special flag |
| Poor image | Any degraded image (see `scripts/degrade.py`) | RETAKE NEEDED, never FAIL |

## Test harness

```bash
python scripts/degrade.py clean_id.jpg out/ --kind id         # blur, dark, glare, edge_crop, thumb
python scripts/degrade.py selfie.jpg out/ --kind webcam       # blur, dark, backlight, partial_face
python scripts/make_demo_cases.py demo_cases                   # synthetic cases from the two sample portraits
python scripts/evaluate.py demo_cases -v                       # confusion matrix + rates
```

`evaluate.py` takes a folder whose subfolders are named for the expected outcome (`PASS`, `FAIL`,
`NEEDS_HUMAN_REVIEW`, `RETAKE_NEEDED`), each case holding `id1`, `id2`, `a`, `b`, `c` images. With `--ocr sidecar`
(default) ID text comes from optional `id1.json` / `id2.json`, so it runs offline. Use `--ocr claude` for the real engine.
It prints the confusion matrix, false-match rate, false-reject rate, retake rate, and the review rate.

Demo result on the six synthetic cases: 6 of 6 outcomes as designed (one "FAIL" style case lands in review by design).
Six synthetic cases prove the plumbing only. Measure real rates on real, consented, labelled data before relying on anything.

The `thumb` degrade variant is for the Claude obstruction check, which needs a live API key, so it is not unit tested.

## Privacy

- Uploads and camera captures are decoded from memory. The app never writes an image to disk (stricter than a cleared temp dir).
  Tesseract mode hands pixels to `pytesseract`, which uses its own short-lived temp file.
- With the Claude backend, ID images are sent to the Anthropic API (banner shown in the UI), and so is the structured result
  for the summary (scores and verdicts, no images). `IDCHECK_OCR=tesseract` turns all of that off. The finger check is skipped in that mode.
- If the API call for the finger check fails, the check is "skipped" (amber badge), not failed.

## Known limits

- Thresholds are uncalibrated. SFace's 0.363 is a general-population operating point, not a fraud-control one.
- Blur, glare, and backlight thresholds were tuned on synthetic images and a few samples only.
- Document corner detection assumes a card that contrasts with its background. A white ID on a white desk will ask for a retake.
- Slash dates such as 03/04/2000 are read day-first. Two IDs with different conventions can disagree on date of birth.
- Two different document types (passport and licence) have different ID numbers by design, and will FAIL the ID number check.
- No liveness or anti-spoofing: a photo of a photo can pass. This only checks image quality and similarity.
- Demographic bias in face recognition is not measured here.
- Tested on Python 3.10 in the build sandbox; code uses nothing newer than 3.10 features, so 3.11 should work unchanged.
