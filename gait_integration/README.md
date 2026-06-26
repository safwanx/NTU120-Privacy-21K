# Gait identity adversary (OpenGait / GaitBase)

Adds a real clip-level gait recognizer to the multi-cue threat model
(remaining.md item 3), replacing the weak silhouette/motion proxy (step 08).
GaitBase is the cue most-cited as surviving body anonymization, so this directly
hardens the paper's "multi-cue" claim.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `gait_paths.py` | CPU | Path glue; locates OpenGait repo + GaitBase checkpoint; auto-sets PILOT_ROOT. |
| `gaitbase_model.py` | CPU/GPU | Standalone GaitBase (no OpenGait training framework); load + embed a silhouette sequence -> 4096-d. |
| `gait_silhouette.py` | CPU | Detection-box -> binary silhouette -> OpenGait `cut_img` (64x44) ordered sequence. |
| `extract_silhouette_masks.py` | **GPU** | YOLO-seg person masks per method (authoritative silhouettes). |
| `eval_gait_identity.py` | **GPU** | The adversary across all protocols x methods; cosine retrieval + coverage. |

## Status (validated 2026-06-14, CPU)

- OpenGait `Baseline` blocks constructed standalone; **strict load** of
  `GaitBase_DA-60000.pt` (all keys matched).
- `embed([S,64,44]) -> 4096-d` L2-normalized: verified.
- `cut_img` normalization -> (64,44): verified.
- All four files compile.
- Not yet run on real data: needs extracted frames + detections (the
  pixel-space prerequisites), which live on SLURM, not locally.

## Assets present

- Code: `repos/adversaries/OpenGait`.
- Checkpoint: `models/gait_opengait/.../CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt`
  (auto-located; override with `GAITBASE_CKPT`). Other checkpoints (GaitSet,
  DeepGaitV2, etc.) are also present if you want to swap models.

## Prerequisites (same as the other pixel-space attackers)

1. Extracted frames: `pipeline/02_extract_frames.py` (CPU).
2. Detections: `pipeline/03_detect_people_faces.py` (GPU) — provides
   the person boxes the silhouette extractor crops from.
3. Anonymized frames per method (face_blur/body_blur/body_pixel already done;
   deepprivacy2 via `04_deepprivacy2_probe_only.py`).

## Run (on SLURM, GPU)

```
# all methods that have complete anonymized frames, all four protocols
python gait_integration/eval_gait_identity.py --methods auto --max-videos 0

# or an explicit method list incl. the learned baseline
python gait_integration/eval_gait_identity.py \
    --methods original,face_blur,body_blur,body_pixel,deepprivacy2 --max-videos 0
```

Writes `features/gait_identity_results.csv` with Rank-1/5/mAP, probe coverage,
and coverage-adjusted Rank-1 per (protocol, method) — same schema as the face /
re-ID tables, so it drops straight into the privacy table and Fig. 4.

`--max-videos` defaults to `SIL_MAX_EVAL` (800) for a quick pass; use `0` for the
full protocol numbers in the paper.

## Silhouette quality (a reviewer-facing choice)

Two silhouette sources:

1. **Default (Otsu).** Otsu on the grayscale person crop, then `cut_img`. No extra
   model; suits NTU's clean backgrounds but is the weak link in gait accuracy.
2. **Recommended (YOLO-seg masks).** Run `extract_silhouette_masks.py` first, then
   pass `--mask-root` to the eval. Authoritative silhouettes, and the right way to
   report gait numbers in the paper.

### Generating masks (GPU)

```
python gait_integration/extract_silhouette_masks.py \
    --methods original,face_blur,body_blur,body_pixel,deepprivacy2 \
    --out gait_integration/work/masks
```

Masks are written **per method** to `<out>/<method>/<stem>/<frame>.png`. This is
deliberate: they are extracted from each method's anonymized frames, so if
body_pixel destroys the silhouette the segmenter fails and that coverage loss is
measured (mirrors the pose-coverage finding), not bypassed. Seg weights
(`yolo11x-seg.pt`) auto-download on first use; place a local copy under the
project or `models/` to avoid the download.

### Eval with segmentation masks

```
python gait_integration/eval_gait_identity.py \
    --methods original,face_blur,body_blur,body_pixel,deepprivacy2 \
    --mask-root gait_integration/work/masks --max-videos 0
```

The eval uses `<mask-root>/original` for the (clean) gallery and
`<mask-root>/<method>` for each probe automatically.

## Deviations from a CASIA-B benchmark (defend to reviewers)

- GaitBase here is the CASIA-B-trained checkpoint applied cross-dataset to NTU
  (no NTU gait training). This measures *transferable* gait identity leakage,
  which is the realistic adversary; note it as such.
- Gallery is always clean (original frames); probe is the anonymized method,
  matching the paper's threat model and the other adversaries.
