"""Path/config glue for the OpenGait (GaitBase) identity adversary.

Mirrors pbp_integration/pbp_paths.py: code + repo on this drive, protocol DATA
(pilot_pack: metadata, splits, frames, detections) resolved via PILOT_ROOT,
which we auto-point at the real pilot_pack if the caller did not set it.

No torch import here; safe to import anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ACCV_ROOT", Path(__file__).resolve().parents[1]))
PIPELINE_STEPS_DIR = PROJECT_ROOT / "pipeline"
OPENGAIT_REPO = PROJECT_ROOT / "repos" / "adversaries" / "OpenGait"
OPENGAIT_PKG = OPENGAIT_REPO / "opengait"

# Point the pipeline at the real pilot_pack if the caller did not.
if not os.environ.get("PILOT_ROOT"):
    for _cand in (Path(r"C:/path/to/ACCV/pilot_pack"), PROJECT_ROOT / "pilot_pack", PROJECT_ROOT / "pilot"):
        if (_cand / "metadata.csv").exists():
            os.environ["PILOT_ROOT"] = str(_cand)
            break

if str(PIPELINE_STEPS_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_STEPS_DIR))


def find_gaitbase_checkpoint() -> Path:
    """Locate the GaitBase (CASIA-B, DA) checkpoint across known model roots."""
    env = os.environ.get("GAITBASE_CKPT")
    if env and Path(env).exists():
        return Path(env)
    rel = ("gait_opengait/opengait_OpenGait_checkpoints/CASIA-B/Baseline/"
           "GaitBase_DA/checkpoints/GaitBase_DA-60000.pt")
    for root in (PROJECT_ROOT / "models", Path(r"C:/path/to/ACCV/models"), Path(r"C:/path/to/ACCV/models")):
        cand = root / rel
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "GaitBase checkpoint not found. Set GAITBASE_CKPT or place it under models/"
        + rel
    )


GAIT_WORK = Path(os.environ.get("GAIT_WORK", PROJECT_ROOT / "gait_integration" / "work"))
GAIT_SIL_DIR = GAIT_WORK / "silhouettes"     # optional cached silhouette sequences
GAIT_RESULTS_DIR = GAIT_WORK / "results"
for _d in (GAIT_SIL_DIR, GAIT_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# GaitBase silhouette geometry (OpenGait standard).
SIL_H, SIL_W = 64, 44
