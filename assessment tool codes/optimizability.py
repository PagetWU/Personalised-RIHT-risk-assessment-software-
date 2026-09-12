from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from .io_utils import array, read_image, resample_mask, spacing_zyx, voxel_volume_cc


HOTSPOT_THRESHOLD_CGY = 4000.0
SHELLS_MM = [(0, 3), (3, 6), (6, 10), (10, 20)]


def min_distance_mm(source: np.ndarray, target: np.ndarray, sampling: tuple[float, float, float]) -> float:
    if not source.any() or not target.any():
        return np.nan
    dist = distance_transform_edt(~target.astype(bool), sampling=sampling)
    return float(np.min(dist[source.astype(bool)]))


def target_union(target_paths: tuple[Path, ...], reference: sitk.Image) -> tuple[np.ndarray, list[str]]:
    union = np.zeros(array(reference).shape, dtype=bool)
    names = []
    for path in target_paths:
        try:
            mask = resample_mask(read_image(path), reference)
            values = array(mask) > 0
            if values.any():
                union |= values
                names.append(path.name)
        except Exception:
            continue
    return union, names


def shell_stats(dose_cgy: np.ndarray, thyroid: np.ndarray, sampling: tuple[float, float, float]) -> dict[str, Any]:
    dist = distance_transform_edt(~thyroid.astype(bool), sampling=sampling)
    outside = ~thyroid.astype(bool)
    out: dict[str, Any] = {}
    for low, high in SHELLS_MM:
        region = outside & (dist > low) & (dist <= high)
        key = f"shell_{low}_{high}mm"
        if region.any():
            values = dose_cgy[region]
            out[f"{key}_mean_cgy"] = float(np.nanmean(values))
            out[f"{key}_max_cgy"] = float(np.nanmax(values))
            out[f"{key}_voxels"] = int(region.sum())
        else:
            out[f"{key}_mean_cgy"] = np.nan
            out[f"{key}_max_cgy"] = np.nan
            out[f"{key}_voxels"] = 0
    return out


def classify(row: dict[str, Any]) -> str:
    if not row["target_available"]:
        return "indeterminate_missing_target"
    overlap = row["thyroid_target_overlap_cc"]
    hot_dist = row["hotspot_to_target_min_distance_mm"]
    if np.isfinite(overlap) and overlap > 0:
        return "non_optimizable_target_adjacent"
    if np.isfinite(hot_dist) and hot_dist <= 3:
        return "non_optimizable_target_adjacent"
    if np.isfinite(hot_dist) and hot_dist <= 10:
        return "limited_optimizable_near_target"
    near = row.get("shell_0_3mm_mean_cgy", np.nan)
    far = row.get("shell_10_20mm_mean_cgy", np.nan)
    if np.isfinite(hot_dist) and hot_dist > 10 and np.isfinite(near) and np.isfinite(far) and near >= far:
        return "potentially_optimizable_dose_tail"
    return "limited_optimizable_near_target"


def optimizability_metrics(
    dose_cgy_image: sitk.Image,
    thyroid_on_dose: sitk.Image,
    target_paths: tuple[Path, ...],
    hotspot_threshold_cgy: float = HOTSPOT_THRESHOLD_CGY,
) -> pd.DataFrame:
    dose = array(dose_cgy_image).astype(np.float32)
    thyroid = array(thyroid_on_dose) > 0
    target, names = target_union(target_paths, dose_cgy_image)
    sampling = spacing_zyx(dose_cgy_image)
    voxel_cc = voxel_volume_cc(dose_cgy_image)
    values = dose[thyroid]
    hotspot_threshold_cgy = float(hotspot_threshold_cgy)
    hotspot = thyroid & (dose >= hotspot_threshold_cgy)
    row: dict[str, Any] = {
        "target_available": bool(target.any()),
        "target_mask_count": len(names),
        "target_mask_names": ";".join(names[:60]),
        "hotspot_threshold_cgy": hotspot_threshold_cgy,
        "thyroid_volume_cc": float(thyroid.sum() * voxel_cc),
        "thyroid_dmean_gy": float(np.nanmean(values) / 100.0) if values.size else np.nan,
        "thyroid_dmax_gy": float(np.nanmax(values) / 100.0) if values.size else np.nan,
        "hotspot_volume_cc": float(hotspot.sum() * voxel_cc),
        "hotspot_fraction_of_thyroid_pct": float(hotspot.sum() / thyroid.sum() * 100.0) if thyroid.sum() else np.nan,
    }
    if target.any():
        overlap = thyroid & target
        row["thyroid_to_target_min_distance_mm"] = min_distance_mm(thyroid, target, sampling)
        row["hotspot_to_target_min_distance_mm"] = min_distance_mm(hotspot, target, sampling)
        row["thyroid_target_overlap_cc"] = float(overlap.sum() * voxel_cc)
        row["thyroid_target_overlap_pct_of_thyroid"] = float(overlap.sum() / thyroid.sum() * 100.0) if thyroid.sum() else np.nan
    else:
        row["thyroid_to_target_min_distance_mm"] = np.nan
        row["hotspot_to_target_min_distance_mm"] = np.nan
        row["thyroid_target_overlap_cc"] = np.nan
        row["thyroid_target_overlap_pct_of_thyroid"] = np.nan
    row.update(shell_stats(dose, thyroid, sampling))
    row["optimizable_class"] = classify(row)
    return pd.DataFrame([row])
