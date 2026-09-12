from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTOSEG_SCRIPT = PACKAGE_ROOT / "autoseg" / "scripts" / "run_thyroid_segmentation_pipeline.ps1"
DEFAULT_AUTOSEG_MODEL = (
    PACKAGE_ROOT
    / "autoseg_model"
    / "Dataset1102_ThyroidSegmentation"
    / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
)


@dataclass(frozen=True)
class AutoSegResult:
    stage_dir: Path
    mask_path: Path
    geometry_json: Path
    geometry_summary: dict
    stdout: str


def _case_id_from_dir(case_dir: Path) -> str:
    return case_dir.name.replace(" ", "_")


def run_thyroid_autoseg(
    case_dir: str | Path,
    output_dir: str | Path,
    case_id: str | None = None,
    script_path: str | Path | None = None,
    model_folder: str | Path | None = None,
) -> AutoSegResult:
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    case_id = case_id or _case_id_from_dir(case_dir)
    script = Path(script_path) if script_path else DEFAULT_AUTOSEG_SCRIPT
    model = Path(model_folder) if model_folder else DEFAULT_AUTOSEG_MODEL

    if not script.exists():
        raise FileNotFoundError(f"Auto-segmentation pipeline script not found: {script}")
    if not model.exists():
        raise FileNotFoundError(f"Auto-segmentation model folder not found: {model}")
    if not (model / "fold_0" / "checkpoint_best.pth").exists():
        raise FileNotFoundError(f"checkpoint_best.pth not found under: {model / 'fold_0'}")

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-PatientDir",
        str(case_dir),
        "-OutputDir",
        str(output_dir),
        "-CaseId",
        case_id,
        "-ModelFolder",
        str(model),
        "-Checkpoint",
        "checkpoint_best.pth",
        "-Device",
        "cuda",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "Auto-segmentation pipeline failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    stage_dir = output_dir / case_id
    mask_path = stage_dir / "recovered_mha" / "thyroid_mask_autoseg.mha"
    geometry_json = stage_dir / "reports" / "geometry_summary.json"
    if not mask_path.exists():
        raise FileNotFoundError(f"Auto-segmentation mask was not produced: {mask_path}")
    geometry_summary = {}
    if geometry_json.exists():
        geometry_summary = json.loads(geometry_json.read_text(encoding="utf-8"))
    return AutoSegResult(stage_dir, mask_path, geometry_json, geometry_summary, completed.stdout)
