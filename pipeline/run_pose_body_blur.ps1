$root = "C:\path\to\ACCV"
$steps = "$root\pipeline"
$logs = "$root\job_logs"
$masterLog = "$logs\pose_pipeline.log"

if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Force -Path $logs | Out-Null }

# If not already the detached worker, relaunch as one and exit.
if ($env:_POSE_WORKER -ne "1") {
    $env:_POSE_WORKER = "1"
    $proc = Start-Process powershell.exe `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" `
        -WindowStyle Hidden -PassThru
    Write-Host "Launched detached (PID $($proc.Id)). Safe to close SSH."
    Write-Host ""
    Write-Host "  Tail: Get-Content `"$masterLog`" -Wait"
    Write-Host "  Last: Get-Content `"$masterLog`" -Tail 40"
    Write-Host "  Alive: Get-Process -Id $($proc.Id) -ErrorAction SilentlyContinue"
    exit 0
}

# ---- Worker (runs inside the detached process) ----

cd $root; . .\.venv\Scripts\Activate.ps1; cd $steps
$env:PYTHONUNBUFFERED = "1"
$env:ACCV_ROOT = $root
$env:PILOT_ROOT = "$root\pilot_pack"
$env:PILOT_OUTPUT_ROOT = "$root\pilot_outputs"

function Log([string]$msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $masterLog -Value $line -Encoding utf8
}

function Run-Stage([string]$label, [string]$script, [string]$logFile) {
    Log "START $label"
    cmd /c "python $script > `"$logFile`" 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 0) { Log "FAIL $label (exit $code)"; return $false }
    Log "DONE  $label"
    return $true
}

Log "======== POSE PIPELINE STARTED ========"

$ok = Run-Stage "17_extract" `
    "17_extract_rgb_pose_keypoints.py --methods body_blur --sample-frames 32 --batch-size 32 --min-valid-frames 2 --keypoint-conf 0.05" `
    "$logs\body_blur_17_pose_extract.log"

if ($ok) {
    $ok = Run-Stage "18_train" "18_train_rgb_pose_identity.py" "$logs\body_blur_18_pose_train.log"
}
if ($ok) {
    $ok = Run-Stage "19_eval" "19_eval_rgb_pose_identity.py --methods original,body_blur" "$logs\body_blur_19_pose_eval.log"
}

if ($ok) { Log "======== PIPELINE COMPLETE ========" }
else      { Log "======== PIPELINE FAILED - check stage logs ========" }
