param(
  [string]$Root = "C:\path\to\ACCV",
  [string]$Gpu = "0",
  [int]$Limit = 0,
  [switch]$Smoke,
  [switch]$OverwriteAnonymized,
  [switch]$SkipAnonymize,
  [switch]$Skip11,
  [switch]$Skip16,
  [switch]$Skip20,
  [switch]$NoArchive
)

$ErrorActionPreference = "Stop"

if ($Smoke -and $Limit -le 0) {
  $Limit = 20
}
if ($Limit -gt 0 -and -not $Smoke -and -not ($Skip11 -and $Skip16 -and $Skip20)) {
  throw "Limit is only safe for smoke/anonymization-only runs. Use -Smoke, or skip 11/16/20."
}

$logDir = Join-Path $Root "job_logs"
$stepsDir = Join-Path $Root "pipeline"
$python = Join-Path $Root ".venv\Scripts\python.exe"
$method = "deepprivacy2"

$stage04 = Join-Path $stepsDir "04_deepprivacy2_probe_only.py"
$stage11 = Join-Path $stepsDir "11_run_remaining_after_04.py"
$stage16 = Join-Path $stepsDir "16_run_action_videomae_pipeline.py"
$stage20 = Join-Path $stepsDir "20_run_rgb_pose_identity_pipeline.py"
$stage17 = Join-Path $stepsDir "17_extract_rgb_pose_keypoints.py"
$pipelineCommon = Join-Path $stepsDir "pipeline_common.py"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:ACCV_ROOT = $Root
$env:PILOT_ROOT = Join-Path $Root "pilot_pack"
$env:PILOT_OUTPUT_ROOT = Join-Path $Root "pilot_outputs"
$env:REPOS_DIR = Join-Path $Root "repos"
$env:MODELS_DIR = Join-Path $Root "models"
$env:EVAL_ANON_METHODS = $method
$env:CUDA_VISIBLE_DEVICES = $Gpu

Set-Location $stepsDir

$preflightLog = Join-Path $logDir "deepprivacy2_probe_preflight.log"
$anonLog = Join-Path $logDir "deepprivacy2_probe_anonymize.log"
$eval11Log = Join-Path $logDir "deepprivacy2_11_eval.log"
$action16Log = Join-Path $logDir "deepprivacy2_16_action.log"
$pose20Log = Join-Path $logDir "deepprivacy2_20_pose.log"
$archiveLog = Join-Path $logDir "deepprivacy2_archive.log"

Remove-Item $preflightLog, $anonLog, $eval11Log, $action16Log, $pose20Log, $archiveLog -Force -ErrorAction SilentlyContinue

function Log-Line {
  param([string]$Message, [string]$Path = $preflightLog)
  $Message | Tee-Object -FilePath $Path -Append
}

function Require-Path {
  param([string]$Path, [string]$Label)
  if (!(Test-Path $Path)) {
    throw "$Label not found: $Path"
  }
}

function Run-PythonStage {
  param(
    [string]$Name,
    [string]$ScriptPath,
    [string[]]$StageArgs,
    [string]$LogPath
  )

  "=" * 80 | Tee-Object -FilePath $LogPath
  "Running $Name" | Tee-Object -FilePath $LogPath -Append
  "$python -u $ScriptPath $(($StageArgs) -join ' ')" | Tee-Object -FilePath $LogPath -Append
  "=" * 80 | Tee-Object -FilePath $LogPath -Append

  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $python -u $ScriptPath @StageArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }

  if ($exitCode -ne 0) {
    throw "$Name failed with exit code $exitCode. See $LogPath"
  }
}

function Copy-IfExists {
  param([string]$Source, [string]$Destination)
  if (Test-Path $Source) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item $Source $Destination -Force
  }
}

"=== deepprivacy2 probe pipeline preflight ===" | Tee-Object -FilePath $preflightLog
Log-Line "root=$Root"
Log-Line "python=$python"
Log-Line "stepsDir=$stepsDir"
Log-Line "PILOT_ROOT=$env:PILOT_ROOT"
Log-Line "PILOT_OUTPUT_ROOT=$env:PILOT_OUTPUT_ROOT"
Log-Line "EVAL_ANON_METHODS=$env:EVAL_ANON_METHODS"
Log-Line "CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
Log-Line "Smoke=$Smoke Limit=$Limit"

Require-Path $python "Venv python"
Require-Path $stepsDir "Pipeline steps directory"
Require-Path $stage04 "DeepPrivacy2 probe stage"
Require-Path $stage11 "Stage 11"
Require-Path $stage16 "Stage 16"
Require-Path $stage20 "Stage 20"
Require-Path $pipelineCommon "pipeline_common.py"
Require-Path (Join-Path $env:REPOS_DIR "privacy_methods\deep_privacy2\configs\anonymizers\FB_cse.py") "DeepPrivacy2 FB_cse config"

$pipelineCommonText = Get-Content $pipelineCommon -Raw
if ($pipelineCommonText -notmatch "EVAL_ANON_METHODS") {
  throw "pipeline_common.py is not patched for EVAL_ANON_METHODS."
}

$stage17Text = Get-Content $stage17 -Raw
if ($stage17Text -notmatch "fast_method_video_ready") {
  throw "Stage 17 is not patched. Copy the patched 17_extract_rgb_pose_keypoints.py first."
}
if ($stage17Text -notmatch "scanning pending pose targets") {
  throw "Stage 17 patch is incomplete: missing scanning log line."
}

$frameManifestDir = Join-Path $env:PILOT_OUTPUT_ROOT "frames\_manifests"
$detectionDir = Join-Path $env:PILOT_OUTPUT_ROOT "detections"
Require-Path $frameManifestDir "Frame manifest directory"
Require-Path $detectionDir "Detection directory"

$manifestFiles = (Get-ChildItem $frameManifestDir -File -Filter "*.json" -ErrorAction Stop).Count
$detectionFiles = (Get-ChildItem $detectionDir -File -Filter "*.json" -ErrorAction Stop).Count
Log-Line "manifest_files=$manifestFiles"
Log-Line "detection_files=$detectionFiles"

if ($manifestFiles -ne 21600) {
  throw "Expected 21600 frame manifests under PILOT_OUTPUT_ROOT, found $manifestFiles. Wrong PILOT_OUTPUT_ROOT?"
}
if ($detectionFiles -ne 21600) {
  throw "Expected 21600 detection JSON files under PILOT_OUTPUT_ROOT, found $detectionFiles. Run/fix stage 03 first."
}

$preflightPython = @'
import importlib.util
import os
from pathlib import Path
print("python_exe_ok")
print("ACCV_ROOT", os.environ.get("ACCV_ROOT"))
print("PILOT_ROOT", os.environ.get("PILOT_ROOT"))
print("PILOT_OUTPUT_ROOT", os.environ.get("PILOT_OUTPUT_ROOT"))
print("EVAL_ANON_METHODS", os.environ.get("EVAL_ANON_METHODS"))
missing = []
for module_name in ["tops.config"]:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        missing.append(module_name)
if missing:
    raise SystemExit(
        "Missing DeepPrivacy2 dependency module(s): "
        + ", ".join(missing)
        + "\nInstall with:\n"
        + "python -m pip install --upgrade --force-reinstall \"tops @ git+https://github.com/hukkelas/torch_ops.git\""
    )
from pipeline_common import ROOT, OUTPUT_ROOT, FRAMES_DIR, ANON_DIR, REPOS
print("pipeline_ROOT", ROOT)
print("pipeline_OUTPUT_ROOT", OUTPUT_ROOT)
print("frames_exists", FRAMES_DIR.exists(), FRAMES_DIR)
print("deepprivacy2_output", ANON_DIR / "deepprivacy2")
print("dp2_repo_exists", (REPOS / "privacy_methods" / "deep_privacy2").exists())
if Path(os.environ["PILOT_OUTPUT_ROOT"]).resolve() != OUTPUT_ROOT.resolve():
    raise SystemExit("pipeline_common did not pick up PILOT_OUTPUT_ROOT")
'@
$preflightPython | & $python -u - 2>&1 | Tee-Object -FilePath $preflightLog -Append
if ($LASTEXITCODE -ne 0) {
  throw "Python preflight failed. See $preflightLog"
}

$dryArgs = @("--dry-run")
if ($Limit -gt 0) {
  $dryArgs += @("--limit", "$Limit")
}
Run-PythonStage -Name "04_deepprivacy2_probe_only.py dry-run" -ScriptPath $stage04 -StageArgs $dryArgs -LogPath $preflightLog

if (-not $SkipAnonymize) {
  $anonArgs = @()
  if ($Limit -gt 0) {
    $anonArgs += @("--limit", "$Limit")
  }
  if ($OverwriteAnonymized) {
    $anonArgs += "--overwrite"
  }
  Run-PythonStage -Name "04_deepprivacy2_probe_only.py" -ScriptPath $stage04 -StageArgs $anonArgs -LogPath $anonLog
}

if ($Smoke) {
  "Smoke run complete. Not running 11/16/20." | Tee-Object -FilePath (Join-Path $logDir "deepprivacy2_smoke_done.log")
  exit 0
}

if (-not $Skip11) {
  Run-PythonStage -Name "11_run_remaining_after_04.py" -ScriptPath $stage11 -StageArgs @("--identity-mode", "parallel", "--identity-gpus", "0,1") -LogPath $eval11Log
}

if (-not $Skip16) {
  Run-PythonStage -Name "16_run_action_videomae_pipeline.py" -ScriptPath $stage16 -StageArgs @("--methods", "original,deepprivacy2", "--batch-size", "4", "--epochs", "80") -LogPath $action16Log
}

if (-not $Skip20) {
  Run-PythonStage -Name "20_run_rgb_pose_identity_pipeline.py" -ScriptPath $stage20 -StageArgs @("--methods", "original,deepprivacy2", "--sample-frames", "32", "--batch-size", "32", "--min-valid-frames", "2", "--keypoint-conf", "0.05") -LogPath $pose20Log
}

if (-not $NoArchive) {
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $archiveRoot = Join-Path $Root "result_archives\deepprivacy2_$timestamp"
  New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

  "archiveRoot=$archiveRoot" | Tee-Object -FilePath $archiveLog

  $features = Join-Path $env:PILOT_OUTPUT_ROOT "features"
  foreach ($name in @(
    "face_recognition_results.csv",
    "person_reid_results.csv",
    "silhouette_proxy_results.csv",
    "pose_identity_results.csv",
    "action_utility_proxy_results.csv",
    "combined_privacy_attack_results.csv",
    "deepprivacy2_probe_status.csv"
  )) {
    Copy-IfExists (Join-Path $features $name) (Join-Path $archiveRoot $name)
  }

  foreach ($name in @(
    "action_videomae_v2_results.csv",
    "action_videomae_v2_per_class.csv",
    "action_videomae_v2_train_history.csv",
    "action_videomae_v2_results.png",
    "action_split_summary.csv"
  )) {
    Copy-IfExists (Join-Path $features "action_videomae_v2\$name") (Join-Path $archiveRoot "action_videomae_v2\$name")
  }

  foreach ($name in @(
    "rgb_pose_identity_results.csv",
    "rgb_pose_identity_train_summary.csv",
    "rgb_pose_identity_results.png"
  )) {
    Copy-IfExists (Join-Path $features "rgb_pose_identity\$name") (Join-Path $archiveRoot "rgb_pose_identity\$name")
  }

  foreach ($log in @($preflightLog, $anonLog, $eval11Log, $action16Log, $pose20Log)) {
    Copy-IfExists $log (Join-Path $archiveRoot ("logs\" + (Split-Path $log -Leaf)))
  }

  "archive complete" | Tee-Object -FilePath $archiveLog -Append
}

"=== deepprivacy2 probe pipeline complete ===" | Tee-Object -FilePath (Join-Path $logDir "deepprivacy2_probe_done.log")



