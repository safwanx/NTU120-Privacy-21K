# NTU120-Privacy-21K

**Do visual anonymizations actually protect identity in action-recognition video?**

NTU120-Privacy-21K is an evaluation protocol and pipeline that stress-tests
privacy-preserving anonymization methods against a battery of identity
adversaries, while measuring how much action-recognition utility survives. It is
built on NTU RGB+D 120.

The headline finding: anonymizations that remove the obvious cue (the face) can
still leak identity through other cues, body appearance, pose, and gait, so
privacy must be evaluated against multiple adversaries, not just one.

## What it measures

**Anonymization methods evaluated:** face blur, body blur, body pixelation, and
DeepPrivacy2.

**Identity leakage**, attacked from several independent cues:

| Cue | Adversary |
|-----|-----------|
| Face | ArcFace face recognition |
| Body appearance | OSNet person re-identification |
| Pose | Keypoint-based identity classifier |
| Gait | GaitBase (OpenGait) |
| Learned features | Privacy Beyond Pixels feature linkability |

**Utility:** action recognition with VideoMAEv2.

## Repository layout

| Path | What it is |
|------|------------|
| `pipeline/` | The evaluation pipeline: numbered stages (frame extraction, detection, anonymization, and each adversary/utility evaluation) plus shared modules. |
| `gait_integration/` | GaitBase (OpenGait) gait-identity adversary. |
| `pbp_integration/` | Privacy Beyond Pixels feature-linkability adversary. |
| `pilot_pack/` | Protocol data: `metadata.csv` and evaluation `splits/`. |
| `scripts/` | Convenience launchers. |
| `tools/` | Repo utilities. |

## Installation

```bash
# Install a CUDA-matched torch / torchvision first (see pytorch.org), then:
pip install -r requirements.txt
```

## Data and model checkpoints

Source video and model weights are **not** bundled (they are large and
separately licensed). You need:

- **NTU RGB+D 120** RGB video, obtained from the official dataset release.
- **Adversary checkpoints:** ArcFace, OSNet re-ID, GaitBase, DeepPrivacy2,
  downloaded from their respective projects.

Place them where the pipeline expects, or point to them with environment
variables (`ACCV_ROOT`, `PILOT_ROOT`, `MODELS_DIR`, `REPOS_DIR`).

## Quickstart

The pipeline runs as ordered stages. See [`pipeline/README.md`](pipeline/README.md)
for the full stage-by-stage commands. A typical run:

```bash
cd pipeline
python 01_validate_metadata.py
python 02_extract_frames.py
python 03_detect_people_faces_parallel.py
python 04_anonymize_baselines_parallel.py --methods face_blur,body_blur
python 11_run_remaining_after_04.py     # runs the identity + utility evaluations
```

Each stage is resumable, and later evaluation stages skip anonymization methods
that have not been produced yet.

## Citation

If you use this protocol or code, please cite the accompanying paper. *(Citation
details to be added.)*
