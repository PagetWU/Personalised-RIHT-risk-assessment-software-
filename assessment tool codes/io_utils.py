from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


IMAGE_EXTENSIONS = (".mha", ".nii", ".nii.gz")
TARGET_TOKENS = ("GTV", "CTV", "PTV")


@dataclass(frozen=True)
class CaseFiles:
    case_dir: Path
    ct_path: Path
    dose_path: Path
    thyroid_mask_path: Path | None
    target_mask_paths: tuple[Path, ...]


def has_image_extension(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in IMAGE_EXTENSIONS)


def image_files(case_dir: Path) -> list[Path]:
    return sorted(path for path in case_dir.rglob("*") if path.is_file() and has_image_extension(path))


def score_candidate(path: Path, tokens: tuple[str, ...]) -> int:
    name = path.name.lower()
    score = 0
    for index, token in enumerate(tokens):
        if token.lower() in name:
            score += 100 - index
    return score


def pick_required(files: list[Path], tokens: tuple[str, ...], label: str) -> Path:
    ranked = sorted(((score_candidate(path, tokens), path) for path in files), key=lambda item: (-item[0], len(str(item[1]))))
    if not ranked or ranked[0][0] <= 0:
        raise FileNotFoundError(f"Cannot find {label} image in case folder.")
    return ranked[0][1]


def pick_optional(files: list[Path], tokens: tuple[str, ...]) -> Path | None:
    ranked = sorted(((score_candidate(path, tokens), path) for path in files), key=lambda item: (-item[0], len(str(item[1]))))
    if not ranked or ranked[0][0] <= 0:
        return None
    return ranked[0][1]


def discover_case_files(case_dir: str | Path, thyroid_mask_path: str | Path | None = None, require_thyroid: bool = True) -> CaseFiles:
    root = Path(case_dir)
    if not root.exists():
        raise FileNotFoundError(f"Case folder does not exist: {root}")
    files = image_files(root)
    ct = pick_required(files, ("CT", "image", "img"), "CT")
    dose = pick_required(files, ("RTdose", "DOSE", "dose"), "dose")
    if thyroid_mask_path:
        thyroid = Path(thyroid_mask_path)
        if not thyroid.exists():
            raise FileNotFoundError(f"Thyroid mask override does not exist: {thyroid}")
    elif require_thyroid:
        thyroid = pick_required(files, ("Thyroid_mask", "thyroid", "thy"), "thyroid mask")
    else:
        thyroid = pick_optional(files, ("Thyroid_mask", "thyroid_mask", "thyroid", "thy"))
    used = {ct.resolve(), dose.resolve()}
    if thyroid is not None:
        used.add(thyroid.resolve())
    targets = []
    for path in files:
        if path.resolve() in used:
            continue
        upper_name = path.name.upper()
        if any(token in upper_name for token in TARGET_TOKENS):
            targets.append(path)
    return CaseFiles(root, ct, dose, thyroid, tuple(sorted(targets)))


def read_image(path: str | Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def image_from_array(values: np.ndarray, reference: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(values.astype(np.float32))
    out.CopyInformation(reference)
    return out


def binary_mask(mask: sitk.Image) -> sitk.Image:
    values = array(mask)
    out = sitk.GetImageFromArray((values > 0).astype(np.uint8))
    out.CopyInformation(mask)
    return out


def resample_image(image: sitk.Image, reference: sitk.Image, interpolator: int, default_value: float = 0) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def resample_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return binary_mask(resample_image(binary_mask(mask), reference, sitk.sitkNearestNeighbor, 0))


def dose_scale_to_cgy(raw_dose: sitk.Image, provided_scale: float | None = None) -> float:
    if provided_scale is not None and np.isfinite(provided_scale):
        return float(provided_scale)
    values = sitk.GetArrayViewFromImage(raw_dose)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    # Existing project convention: dose maps with max <=120 are Gy-like and need Gy -> cGy.
    return 100.0 if float(np.nanmax(finite)) <= 120.0 else 1.0


def spacing_zyx(image: sitk.Image) -> tuple[float, float, float]:
    x, y, z = image.GetSpacing()
    return (float(z), float(y), float(x))


def voxel_volume_cc(image: sitk.Image) -> float:
    return float(np.prod(image.GetSpacing()) / 1000.0)
