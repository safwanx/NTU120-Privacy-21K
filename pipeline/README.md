# Split Pilot Pipeline Scripts

Run these from this folder or from the project root with the same Python environment used for the notebook.
The scripts share configuration through `pipeline_common.py` and environment variables.

Recommended local Windows defaults are discovered automatically if `C:\path\to\ACCV\pilot_pack` exists. You can override paths explicitly:

```powershell
$env:ACCV_ROOT = "C:\path\to\ACCV"
$env:PILOT_ROOT = "C:\path\to\ACCV\pilot_pack"
$env:PILOT_OUTPUT_ROOT = "C:\path\to\ACCV\pilot_outputs"
$env:REPOS_DIR = "C:\path\to\ACCV\repos"
$env:MODELS_DIR = "C:\path\to\ACCV\models"
$env:CUDA_VISIBLE_DEVICES = "0"
```

Run order:

1. `python 01_validate_metadata.py`
2. `python 02_extract_frames.py`
3. `python 03_detect_people_faces_parallel.py --gpus 0,1 --workers-per-gpu 2`
4. `python 04_anonymize_baselines_parallel.py --workers 8 --methods face_blur,body_blur`
5. `python 11_run_remaining_after_04.py`

Before step 5, install the extra dependency imported by `torchreid`:

```powershell
python -m pip install gdown
```

Each stage is resumable where the original notebook had completion checks.

For dual-GPU detection, replace step 3 with:

```powershell
python 03_detect_people_faces_parallel.py --gpus 0,1
```

For a small smoke test:

```powershell
python 03_detect_people_faces_parallel.py --gpus 0,1 --limit 20
```

For parallel simple anonymization, replace step 4 with:

```powershell
python 04_anonymize_baselines_parallel.py --workers 8
```

For a small smoke test:

```powershell
python 04_anonymize_baselines_parallel.py --workers 4 --limit 40
```

If only selected anonymization methods are complete, later evaluation scripts
will skip incomplete method folders automatically.

## VideoMAEv2 Action Utility

The old `09_eval_action_utility.py` is only a lightweight smoke test. For the
paper-grade RGB action utility result, use the VideoMAEv2 feature pipeline:

```powershell
python -m pip install transformers timm safetensors accelerate
```

First run will download the Hugging Face model `OpenGVLab/VideoMAEv2-Base`.
That model uses custom Hugging Face remote code, so `13_extract_videomae_features.py`
loads it with `trust_remote_code=True`.

Recommended path while storage is constrained:

```powershell
$root = "C:\path\to\ACCV"
$env:ACCV_ROOT = $root
$env:PILOT_ROOT = "$root\pilot_pack"
$env:PILOT_OUTPUT_ROOT = "$root\pilot_outputs"
$env:CUDA_VISIBLE_DEVICES = "0"

cd "$root\pipeline"

python 12_prepare_action_videomae_splits.py
python 13_extract_videomae_features.py --methods original,body_pixel --batch-size 4 --precision auto
python 14_train_action_videomae_head.py --epochs 80
python 15_eval_action_videomae_head.py --methods original,body_pixel
```

For a quick smoke test:

```powershell
python 13_extract_videomae_features.py --methods original,body_pixel --limit 20 --batch-size 2
python 14_train_action_videomae_head.py --epochs 3
python 15_eval_action_videomae_head.py --methods original,body_pixel
```

Feature extraction is resumable per video. Once a method's VideoMAEv2 features
are extracted and evaluation CSVs are archived, the large anonymized frame folder
for that method can be deleted and regenerated later if needed.

Full wrapper:

```powershell
python 16_run_action_videomae_pipeline.py --methods original,body_pixel --batch-size 4 --epochs 80
```

## RGB-Extracted Pose Identity

`07_eval_pose_identity.py` uses native NTU skeleton files, so it does not see
the anonymized RGB output. For the anonymization-aware pose identity result,
run YOLO pose on original/anonymized frame folders and then train/evaluate the
compact keypoint features:

```powershell
python -m pip install ultralytics joblib
```

The default pose model is `yolo11m-pose.pt`. First run will download it through
Ultralytics if it is not already cached. If that model is unavailable in your
installed Ultralytics version, use `--pose-model yolov8m-pose.pt`.

Recommended run for the currently available method:

```powershell
$root = "C:\path\to\ACCV"
$env:ACCV_ROOT = $root
$env:PILOT_ROOT = "$root\pilot_pack"
$env:PILOT_OUTPUT_ROOT = "$root\pilot_outputs"
$env:CUDA_VISIBLE_DEVICES = "0"

cd "$root\pipeline"

python 17_extract_rgb_pose_keypoints.py --methods original,body_pixel --sample-frames 24 --batch-size 32
python 18_train_rgb_pose_identity.py
python 19_eval_rgb_pose_identity.py --methods original,body_pixel
```

Smoke test:

```powershell
python 17_extract_rgb_pose_keypoints.py --methods original,body_pixel --limit 20 --sample-frames 8 --batch-size 16
python 18_train_rgb_pose_identity.py
python 19_eval_rgb_pose_identity.py --methods original,body_pixel
```

Full wrapper:

```powershell
python 20_run_rgb_pose_identity_pipeline.py --methods original,body_pixel --sample-frames 24 --batch-size 32
```

After `17`, `18`, and `19` complete for a method, the pose result no longer
needs that method's anonymized JPEG frames. Keep:

- `pilot_outputs\features\rgb_pose_identity\`
- `pilot_outputs\features\action_videomae_v2\`
- archived CSVs/plots

Then you can delete only the large method folder under
`pilot_outputs\anonymized\`.
