$ErrorActionPreference = "Stop"

$root = "C:\path\to\ACCV"
$logDir = Join-Path $root "job_logs"
$stepsDir = Join-Path $root "pipeline"
$python = Join-Path $root ".venv\Scripts\python.exe"
$stage17 = Join-Path $stepsDir "17_extract_rgb_pose_keypoints.py"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:ACCV_ROOT = $root
$env:PILOT_ROOT = Join-Path $root "pilot_pack"
$env:PILOT_OUTPUT_ROOT = Join-Path $root "pilot_outputs"
$env:REPOS_DIR = Join-Path $root "repos"
$env:MODELS_DIR = Join-Path $root "models"

Set-Location $stepsDir

$log17 = Join-Path $logDir "body_blur_17_pose_extract.log"
$log18 = Join-Path $logDir "body_blur_18_pose_train.log"
$log19 = Join-Path $logDir "body_blur_19_pose_eval.log"
$preflightLog = Join-Path $logDir "body_blur_pose_preflight.log"

Remove-Item $log17, $log18, $log19, $preflightLog -Force -ErrorAction SilentlyContinue

"=== preflight ===" | Tee-Object -FilePath $preflightLog
"root=$root" | Tee-Object -FilePath $preflightLog -Append
"python=$python" | Tee-Object -FilePath $preflightLog -Append
"PILOT_ROOT=$env:PILOT_ROOT" | Tee-Object -FilePath $preflightLog -Append
"PILOT_OUTPUT_ROOT=$env:PILOT_OUTPUT_ROOT" | Tee-Object -FilePath $preflightLog -Append

if (!(Test-Path $python)) {
  throw "Venv python not found: $python"
}
if (!(Test-Path $stage17)) {
  throw "Stage 17 not found: $stage17"
}

$stage17Text = Get-Content $stage17 -Raw
if ($stage17Text -notmatch "fast_method_video_ready") {
  throw "Stage 17 is not patched. Copy the patched 17_extract_rgb_pose_keypoints.py first."
}
if ($stage17Text -notmatch "scanning pending pose targets") {
  throw "Stage 17 patch is incomplete: missing scanning log line."
}

$frameDirs = (Get-ChildItem (Join-Path $env:PILOT_OUTPUT_ROOT "frames") -Directory -ErrorAction Stop).Count
$manifestFiles = (Get-ChildItem (Join-Path $env:PILOT_OUTPUT_ROOT "frames\_manifests") -File -Filter "*.json" -ErrorAction Stop).Count
$bodyBlurDirs = (Get-ChildItem (Join-Path $env:PILOT_OUTPUT_ROOT "anonymized\body_blur") -Directory -ErrorAction Stop).Count

"frame_dirs=$frameDirs" | Tee-Object -FilePath $preflightLog -Append
"manifest_files=$manifestFiles" | Tee-Object -FilePath $preflightLog -Append
"body_blur_dirs=$bodyBlurDirs" | Tee-Object -FilePath $preflightLog -Append

if ($manifestFiles -ne 21600) {
  throw "Expected 21600 frame manifests under PILOT_OUTPUT_ROOT, found $manifestFiles. Wrong PILOT_OUTPUT_ROOT?"
}
if ($bodyBlurDirs -ne 21600) {
  throw "Expected 21600 body_blur frame dirs, found $bodyBlurDirs. Stage 04 incomplete or wrong output root."
}

& $python -u - <<'PY' 2>&1 | Tee-Object -FilePath $preflightLog -Append
import os
from pathlib import Path
print("python_exe_ok")
print("ACCV_ROOT", os.environ.get("ACCV_ROOT"))
print("PILOT_ROOT", os.environ.get("PILOT_ROOT"))
print("PILOT_OUTPUT_ROOT", os.environ.get("PILOT_OUTPUT_ROOT"))
from pipeline_common import ROOT, OUTPUT_ROOT, FRAMES_DIR, ANON_DIR
print("pipeline_ROOT", ROOT)
print("pipeline_OUTPUT_ROOT", OUTPUT_ROOT)
print("frames_exists", FRAMES_DIR.exists(), FRAMES_DIR)
print("body_blur_exists", (ANON_DIR / "body_blur").exists(), ANON_DIR / "body_blur")
if Path(os.environ["PILOT_OUTPUT_ROOT"]).resolve() != OUTPUT_ROOT.resolve():
    raise SystemExit("pipeline_common did not pick up PILOT_OUTPUT_ROOT")
PY

"=== 17: RGB pose extraction for body_blur ===" | Tee-Object -FilePath $log17
& $python -u $stage17 --methods body_blur --sample-frames 32 --batch-size 32 --min-valid-frames 2 --keypoint-conf 0.05 2>&1 | Tee-Object -FilePath $log17 -Append
if ($LASTEXITCODE -ne 0) { throw "Stage 17 failed with exit code $LASTEXITCODE" }

"=== 18: train RGB pose identity head ===" | Tee-Object -FilePath $log18
& $python -u (Join-Path $stepsDir "18_train_rgb_pose_identity.py") 2>&1 | Tee-Object -FilePath $log18 -Append
if ($LASTEXITCODE -ne 0) { throw "Stage 18 failed with exit code $LASTEXITCODE" }

"=== 19: eval RGB pose identity body_blur ===" | Tee-Object -FilePath $log19
& $python -u (Join-Path $stepsDir "19_eval_rgb_pose_identity.py") --methods original,body_blur 2>&1 | Tee-Object -FilePath $log19 -Append
if ($LASTEXITCODE -ne 0) { throw "Stage 19 failed with exit code $LASTEXITCODE" }

"=== body_blur 17-19 complete ===" | Tee-Object -FilePath (Join-Path $logDir "body_blur_pose_done.log")
