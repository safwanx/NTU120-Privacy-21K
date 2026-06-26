#!/usr/bin/env python3
"""Extract YOLO pose keypoint features from original/anonymized RGB frames.

Outputs are compact per-video NPZ files. After these features are extracted
and evaluated, the large anonymized frame folder for that method can be deleted.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from pipeline_common import DEVICE, RANDOM_SEED, append_jsonl
from rgb_pose_common import (
    COCO_KEYPOINTS,
    RGB_POSE_LOG_DIR,
    extraction_targets,
    feature_status,
    filename_to_stem,
    frame_dir_for_method,
    method_video_ready,
    pose_feature_path,
    resolve_methods,
    resolve_protocols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="auto")
    parser.add_argument("--protocols", default="auto")
    parser.add_argument("--pose-model", default="yolo11m-pose.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-frames", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf", type=float, default=0.15)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per method; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-anonymized-gallery",
        action="store_true",
        help="Also extract anonymized gallery features. Not needed for the default original-gallery adversary.",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    if importlib.util.find_spec("ultralytics") is None:
        raise SystemExit("Missing package: ultralytics\nInstall in the active venv:\npython -m pip install ultralytics")


def sample_frame_paths(frame_dir: Path, sample_frames: int) -> list[Path]:
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        return []
    indices = np.linspace(0, len(paths) - 1, num=sample_frames)
    return [paths[int(round(index))] for index in indices]


def keypoints_for_result(result, image_shape: tuple[int, int], keypoint_conf: float) -> tuple[np.ndarray, float] | None:
    if result.keypoints is None:
        return None

    xy = result.keypoints.xy
    if xy is None or len(xy) == 0:
        return None

    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) > 0:
        best_idx = int(torch.argmax(result.boxes.conf).detach().cpu())
        box = result.boxes.xyxy[best_idx].detach().cpu().numpy().astype(np.float32)
    else:
        best_idx = 0
        height, width = image_shape
        box = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)

    xy_np = xy[best_idx].detach().cpu().numpy().astype(np.float32)
    if getattr(result.keypoints, "conf", None) is not None and result.keypoints.conf is not None:
        conf_np = result.keypoints.conf[best_idx].detach().cpu().numpy().astype(np.float32)
    else:
        conf_np = np.ones((xy_np.shape[0],), dtype=np.float32)

    x1, y1, x2, y2 = box.tolist()
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pose = np.zeros((COCO_KEYPOINTS, 3), dtype=np.float32)
    count = min(COCO_KEYPOINTS, xy_np.shape[0])
    pose[:count, 0] = (xy_np[:count, 0] - x1) / width
    pose[:count, 1] = (xy_np[:count, 1] - y1) / height
    pose[:count, 2] = conf_np[:count]

    low_conf = pose[:, 2] < keypoint_conf
    pose[low_conf, :2] = 0.0
    pose[low_conf, 2] = 0.0
    pose[:, :2] = np.clip(pose[:, :2], -0.5, 1.5)

    valid_conf = pose[:, 2][pose[:, 2] > 0]
    mean_conf = float(valid_conf.mean()) if len(valid_conf) else 0.0
    return pose, mean_conf


def feature_from_sequence(seq: np.ndarray) -> np.ndarray:
    # seq: T,K,3 with fixed T. Coordinates are bbox-normalized; confidence is
    # retained so the classifier can learn when anonymization destroys pose.
    flat = seq.reshape(-1)
    mean = seq.mean(axis=0).reshape(-1)
    std = seq.std(axis=0).reshape(-1)
    if len(seq) > 1:
        diff = np.diff(seq[:, :, :2], axis=0)
        diff_mean = diff.mean(axis=0).reshape(-1)
        diff_std = diff.std(axis=0).reshape(-1)
    else:
        diff_mean = np.zeros((COCO_KEYPOINTS * 2,), dtype=np.float32)
        diff_std = np.zeros((COCO_KEYPOINTS * 2,), dtype=np.float32)
    return np.concatenate([flat, mean, std, diff_mean, diff_std]).astype(np.float32)


def extract_video_feature(model, method: str, stem: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]] | None:
    frame_paths = sample_frame_paths(frame_dir_for_method(method, stem), args.sample_frames)
    if not frame_paths:
        return None

    sequence = []
    valid_frames = 0
    keypoint_confs = []

    for batch_start in range(0, len(frame_paths), args.batch_size):
        batch_paths = frame_paths[batch_start : batch_start + args.batch_size]
        images = []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                images.append(None)
            else:
                images.append(image)

        valid_images = [image for image in images if image is not None]
        if valid_images:
            predictions = model.predict(valid_images, conf=args.conf, verbose=False, device=DEVICE)
        else:
            predictions = []

        pred_iter = iter(predictions)
        for image in images:
            if image is None:
                sequence.append(np.zeros((COCO_KEYPOINTS, 3), dtype=np.float32))
                continue

            result = next(pred_iter)
            pose_result = keypoints_for_result(result, image.shape[:2], args.keypoint_conf)
            if pose_result is None:
                sequence.append(np.zeros((COCO_KEYPOINTS, 3), dtype=np.float32))
                continue

            pose, mean_conf = pose_result
            sequence.append(pose)
            valid_frames += 1
            keypoint_confs.append(mean_conf)

    if valid_frames < args.min_valid_frames:
        return None

    seq = np.stack(sequence).astype(np.float32, copy=False)
    feature = feature_from_sequence(seq)
    stats = {
        "sampled_frames": float(len(sequence)),
        "valid_pose_frames": float(valid_frames),
        "pose_frame_coverage": valid_frames / max(len(sequence), 1),
        "mean_keypoint_conf": float(np.mean(keypoint_confs)) if keypoint_confs else 0.0,
    }
    return feature, stats


def save_pose_feature(method: str, stem: str, feature: np.ndarray, stats: dict[str, float], args: argparse.Namespace) -> None:
    path = pose_feature_path(method, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature=feature.astype(np.float16),
        sampled_frames=np.asarray(stats["sampled_frames"], dtype=np.float32),
        valid_pose_frames=np.asarray(stats["valid_pose_frames"], dtype=np.float32),
        pose_frame_coverage=np.asarray(stats["pose_frame_coverage"], dtype=np.float32),
        mean_keypoint_conf=np.asarray(stats["mean_keypoint_conf"], dtype=np.float32),
        sample_frames=np.asarray(args.sample_frames, dtype=np.int32),
        feature_dim=np.asarray(feature.shape[0], dtype=np.int32),
        model=np.asarray(args.pose_model),
    )


def fast_method_video_ready(method: str, stem: str) -> bool:
    """Use a cheap readiness check for anonymized frame folders.

    The full anonymization_complete check counts every expected frame in each
    folder, which is very slow when selecting thousands of pose targets. Stage 04
    already owns exact completeness; here we only need to avoid empty/missing
    folders before extraction.
    """
    if method == "original":
        return method_video_ready(method, stem)

    frame_dir = frame_dir_for_method(method, stem)
    return frame_dir.is_dir() and next(frame_dir.glob("*.jpg"), None) is not None


def pending_stems(method: str, protocols: list[str], args: argparse.Namespace) -> list[str]:
    filenames = extraction_targets(method, protocols, include_anonymized_gallery=args.include_anonymized_gallery)
    stems = []
    for filename in filenames:
        stem = filename_to_stem(filename)
        if not fast_method_video_ready(method, stem):
            continue
        if pose_feature_path(method, stem).exists() and not args.overwrite:
            continue
        stems.append(stem)
    stems = sorted(dict.fromkeys(stems))
    if args.limit > 0:
        stems = stems[: args.limit]
    return stems


def main() -> int:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    check_dependencies()

    from ultralytics import YOLO

    protocols = resolve_protocols(args.protocols)
    methods = resolve_methods(args.methods, include_original=True)

    print(f"Pose model: {args.pose_model}")
    print(f"Methods   : {methods}")
    print(f"Protocols : {protocols}")
    model = YOLO(args.pose_model)
    model.to(DEVICE)

    for method in methods:
        print(f"{method}: scanning pending pose targets...", flush=True)
        stems = pending_stems(method, protocols, args)
        print(f"{method}: pending pose features={len(stems)}", flush=True)
        start_time = time.time()
        for stem in tqdm(stems, desc=f"pose {method}"):
            try:
                result = extract_video_feature(model, method, stem, args)
                if result is None:
                    append_jsonl(
                        RGB_POSE_LOG_DIR / "rgb_pose_extraction_errors.jsonl",
                        {"method": method, "stem": stem, "error": "insufficient_valid_pose_frames"},
                    )
                    continue
                feature, stats = result
                save_pose_feature(method, stem, feature, stats, args)
            except Exception as exc:
                append_jsonl(
                    RGB_POSE_LOG_DIR / "rgb_pose_extraction_errors.jsonl",
                    {"method": method, "stem": stem, "error": str(exc)},
                )
        elapsed = time.time() - start_time
        print(f"{method}: complete in {elapsed / 60:.1f} min")

    status = feature_status(methods, protocols)
    status_path = RGB_POSE_LOG_DIR / "rgb_pose_feature_status.csv"
    status.to_csv(status_path, index=False)
    print(status.to_string(index=False))
    print(f"Status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
