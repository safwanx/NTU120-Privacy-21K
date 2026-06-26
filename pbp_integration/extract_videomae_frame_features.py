#!/usr/bin/env python3
"""GPU step: extract a POOL of VideoMAE clip features per video for PBP's privacy loss.

PBP's SSL privacy objective ("fb" frame bank) needs several feature vectors per
video, drawn from different temporal windows, so it can push two views of the
same video apart in the anonymized space. Our main extractor
(pipeline/13_extract_videomae_features.py) mean-pools to ONE vector
per video, which destroys this. This script samples `--pool` distinct 16-frame
windows per video and saves [pool, 768] to <out>/<stem>.npy.

Feed the output dir to export_protocol_features_to_pbp.py --frame-feat-dir.

Requires a GPU realistically (it is the same VideoMAE forward as step 13, just
`pool` times more clips). Run on the SLURM box:

    python pbp_integration/extract_videomae_frame_features.py \
        --split train_original_r1_allcams --pool 10 \
        --out pbp_integration/work/frame_features/train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

import pbp_paths as P
from action_videomae_common import load_action_split, frame_dir_for_method  # noqa: E402
from pipeline_common import DEVICE  # noqa: E402

DEFAULT_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"


def load_model(model_id: str):
    from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    proc = VideoMAEImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, config=cfg, trust_remote_code=True)
    model.eval().to(DEVICE)
    return proc, model


def sample_window(frame_paths, num_frames, offset):
    """Pick num_frames around a temporal offset in [0,1] of the clip."""
    n = len(frame_paths)
    center = offset * (n - 1)
    half = num_frames / 2
    idx = np.clip(np.round(np.linspace(center - half, center + half, num_frames)), 0, n - 1).astype(int)
    frames = []
    for i in idx:
        img = cv2.imread(str(frame_paths[i]))
        if img is None:
            return None
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


@torch.no_grad()
def extract(proc, model, videos):
    inputs = proc(videos, return_tensors="pt")
    pv = inputs["pixel_values"]
    if pv.shape[1] != 3 and pv.shape[2] == 3:
        pv = pv.permute(0, 2, 1, 3, 4).contiguous()
    pv = pv.to(DEVICE)
    out = model.extract_features(pv) if hasattr(model, "extract_features") else model(pixel_values=pv)
    feats = out if isinstance(out, torch.Tensor) else getattr(out, "last_hidden_state", out[0])
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    return feats.float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pool", type=int, default=P.FB_FRAME_POOL)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.model_id)
    offsets = np.linspace(0.0, 1.0, args.pool)

    rows = load_action_split(args.split)
    for r in tqdm(rows, desc=f"frame-bank {args.split}"):
        stem = str(r["stem"])
        dst = out_dir / f"{stem}.npy"
        if dst.exists():
            continue
        frame_paths = sorted(frame_dir_for_method("original", stem).glob("*.jpg"))
        if not frame_paths:
            continue
        windows = [w for off in offsets if (w := sample_window(frame_paths, args.num_frames, off))]
        if not windows:
            continue
        feats = extract(proc, model, windows)  # [pool, 768]
        np.save(dst, feats.astype(np.float16))
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
