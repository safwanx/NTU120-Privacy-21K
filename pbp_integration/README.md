# Privacy Beyond Pixels (PBP) integration

Wires the ICLR 2026 latent anonymizer **Privacy Beyond Pixels** into our
NTU120-Privacy-21K protocol as a learned-SOTA baseline (remaining.md item 2).

PBP anonymizes VideoMAE **features**, not pixels, so the pixel-space attackers
(ArcFace / OSNet / pose / gait) do **not** apply. The matched attack is in
feature space: clean gallery descriptors vs anonymized probe descriptors, cosine
retrieval, same Rank-1/5/mAP + coverage metrics as the re-ID adversary. Report
PBP as a separate latent-paradigm row/table.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `pbp_paths.py` | CPU | Path glue + imports PBP's AAM without its tridet/mgfn deps. |
| `export_protocol_features_to_pbp.py` | CPU | Our cached VideoMAE features -> PBP HDF5 + label JSON. |
| `extract_videomae_frame_features.py` | **GPU** | Pool of clip features per video for PBP's SSL privacy loss. |
| `train_pbp_aam_ntu.py` | CPU/GPU | Faithful PBP AAM objective (recon + AR utility − NTXent), NTU-only. |
| `eval_pbp_feature_linkability.py` | CPU | Feature-space identity attack: clean vs AAM-anonymized probes. |

All artifacts land under `pbp_integration/work/` (override with `PBP_WORK`).

## Status (validated 2026-06-13)

- Repo cloned: `repos/privacy_methods/PrivacyBeyondPixels`.
- AAM build + forward + PBP's NTXent loss: **verified on CPU**.
- Trainer loop (fb on and off): **verified end-to-end on synthetic features, CPU**.
- Not yet run on real data: the cached VideoMAE features live on the SLURM box,
  not this machine. Everything below is ready to run once features are present.

## Data locations (2026-06-13)

- **Code + repos + PBP work dir:** `C:\path\to\ACCV` (this tree).
- **Protocol data:** `C:\path\to\ACCV\pilot_pack` (metadata.csv, splits, all 21,600 rgb videos,
  skeletons). `pbp_paths.py` auto-points `PILOT_ROOT` here.
- **Model weights:** `C:\path\to\ACCV\models` (deepprivacy2, gait_opengait, face_arcface,
  person_reid_osnet) and `D:\...\models`.
- **VideoMAE features: NOT present on any local drive.** `C:\path\to\ACCV\pilot_outputs` is an
  empty output scaffold (0 frames, empty aggregates). Features were computed on SLURM
  and only result CSVs returned. They must be regenerated: frames (CPU,
  pipeline/02) then VideoMAE forward (GPU, step 13) before any PBP step that
  needs features.
- **Validated against real data:** `pipeline_common` loads the real protocol (21,600
  videos, all splits correct), and the exporter wrote real label JSONs for
  train/test_c2/test_c3 (10,800 / 3,600 / 3,600).

## Pipeline order

PBP's privacy loss needs a *pool* of clip features per video. Our step-13
extractor mean-pools to one vector, which is enough for the clip-level file and
the feature-space attack, but not for the SSL privacy term.

1. **Export clip-level features + labels (CPU).** Needs the step-13 aggregates
   (`features/action_videomae_v2/aggregates/original__<split>.npz`) present.
   ```
   python pbp_integration/export_protocol_features_to_pbp.py \
       --splits train_original_r1_allcams,test_original_c2r2
   ```

2. **Extract the frame bank (GPU).** Only needed for the faithful privacy loss.
   ```
   python pbp_integration/extract_videomae_frame_features.py \
       --split train_original_r1_allcams --pool 10 \
       --out pbp_integration/work/frame_features/train
   ```
   Then re-run the exporter with `--frame-feat-dir pbp_integration/work/frame_features/train`
   to write the `..._fb10.h5` file.

3. **Train the AAM.** Faithful objective; CPU works (AAM is a 2-layer MLP).
   ```
   python pbp_integration/train_pbp_aam_ntu.py \
       --train-split train_original_r1_allcams --epochs 100
   ```
   Without the frame bank, run a recon+utility ablation with `--fb-weight 0`
   (not the real PBP, only a fallback).

4. **Run the feature-space attack (CPU).** Needs the per-video VideoMAE `.npy`
   present (for the privacy gallery/probe splits).
   ```
   python pbp_integration/eval_pbp_feature_linkability.py \
       --aam pbp_integration/work/saved_models/pbp_aam_ntu.pth
   ```
   Writes `features/feature_linkability_results.csv` with `original` (clean,
   leakage upper bound) and `pbp_aam` (anonymized) rows per protocol.

## Note on action utility

The attack script reports the privacy side. For the utility side, apply the
trained AAM to the C2/C3 test features and run the existing action head
(steps 14/15) on the transformed features; the AAM checkpoint stores the AR head
used during training as a reference.

## Deviations from upstream PBP (be able to defend these to reviewers)

- **NTU-only, no TAD/AD co-training.** PBP's `multitask_train_fa.py` hard-loads
  THUMOS + UCF-Crime even at zero loss weight, so we reimplement only the AR +
  privacy objective on NTU. Same losses, same AAM module, same NTXent.
- **MLP AAM default.** Our descriptor is one clip vector per video; the MLP AAM
  is the unambiguous per-vector transform. PBP's transformer AAM expects a
  sequence; available via `--arch transformer` if a sequence is exported.
- **Frame bank = temporal-window pool.** We approximate per-frame features by
  sampling `--pool` distinct 16-frame windows per video.
