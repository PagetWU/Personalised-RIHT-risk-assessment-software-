from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .features import DM_FEATURE, extract_dosiomics, compute_dvh
from .io_utils import array, image_from_array
from .model import CoxRIHTModel, TIME_LABELS


@dataclass(frozen=True)
class Scenario:
    name: str
    display: str
    kind: str
    factor: float | None = None
    threshold_cgy: float | None = None
    tau_cgy: float | None = None
    max_reduction: float | None = None
    sigma_mm: float = 0.0


def build_scenarios(hotspot_threshold_cgy: float = 4000.0) -> list[Scenario]:
    threshold_cgy = float(hotspot_threshold_cgy)
    threshold_gy = threshold_cgy / 100.0
    return [
        Scenario("whole_x090", "Whole dose map x0.90", "global", factor=0.90),
        Scenario("whole_x080", "Whole dose map x0.80", "global", factor=0.80),
        Scenario(
            f"hard_gt{threshold_cgy:.0f}_x080",
            f"Thyroid voxels >{threshold_gy:g} Gy x0.80",
            "threshold",
            factor=0.80,
            threshold_cgy=threshold_cgy,
        ),
        Scenario(
            f"soft{threshold_cgy:.0f}_tau200_cap080_sigma3mm",
            f"Soft >{threshold_gy:g} Gy cap x0.80, tau=2 Gy, Gaussian 3 mm",
            "soft_cap",
            threshold_cgy=threshold_cgy,
            tau_cgy=200.0,
            max_reduction=0.20,
            sigma_mm=3.0,
        ),
    ]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def smooth_like_image(values: np.ndarray, reference: sitk.Image, sigma_mm: float) -> np.ndarray:
    if sigma_mm <= 0:
        return values
    img = image_from_array(values.astype(np.float32), reference)
    smoothed = sitk.SmoothingRecursiveGaussian(img, float(sigma_mm))
    return array(smoothed).astype(np.float32)


def apply_scenario(dose_cgy: np.ndarray, thyroid_mask: np.ndarray, scenario: Scenario, reference: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    roi = thyroid_mask > 0
    if scenario.kind == "global":
        factor = np.ones_like(dose_cgy, dtype=np.float32)
        factor[:] = float(scenario.factor)
        return dose_cgy * factor, factor
    if scenario.kind == "threshold":
        factor = np.ones_like(dose_cgy, dtype=np.float32)
        hit = roi & (dose_cgy > float(scenario.threshold_cgy))
        factor[hit] = float(scenario.factor)
        return dose_cgy * factor, factor
    if scenario.kind == "soft_cap":
        reduction = np.zeros_like(dose_cgy, dtype=np.float32)
        reduction[roi] = float(scenario.max_reduction) * sigmoid((dose_cgy[roi] - float(scenario.threshold_cgy)) / float(scenario.tau_cgy))
        reduction = smooth_like_image(reduction, reference, scenario.sigma_mm)
        reduction = np.clip(reduction, 0.0, float(scenario.max_reduction))
        reduction[~roi] = 0.0
        factor = 1.0 - reduction
        return dose_cgy * factor, factor
    raise ValueError(f"Unsupported scenario kind: {scenario.kind}")


def counterfactual_audit(
    model: CoxRIHTModel,
    baseline_features: dict[str, float],
    dose_cgy_image: sitk.Image,
    thyroid_on_dose: sitk.Image,
    hotspot_threshold_cgy: float = 4000.0,
) -> pd.DataFrame:
    base_pred = model.predict(baseline_features)
    dose = array(dose_cgy_image).astype(np.float32)
    mask = array(thyroid_on_dose) > 0
    rows: list[dict[str, Any]] = []
    for scenario in build_scenarios(hotspot_threshold_cgy):
        new_dose, factors = apply_scenario(dose, mask, scenario, dose_cgy_image)
        new_image = image_from_array(new_dose, dose_cgy_image)
        new_features = dict(baseline_features)
        new_features.update(extract_dosiomics(new_image, thyroid_on_dose))
        updated_dvh = compute_dvh(new_image, thyroid_on_dose)
        new_features[DM_FEATURE] = updated_dvh[DM_FEATURE]
        new_pred = model.predict(new_features)
        roi_factors = factors[mask]
        row = {
            "scenario": scenario.name,
            "scenario_display": scenario.display,
            "baseline_lp": base_pred.linear_predictor,
            "counterfactual_lp": new_pred.linear_predictor,
            "delta_lp": new_pred.linear_predictor - base_pred.linear_predictor,
            "baseline_partial_hazard": base_pred.partial_hazard,
            "counterfactual_partial_hazard": new_pred.partial_hazard,
            "partial_hazard_relative_change_pct": 100.0 * (new_pred.partial_hazard / base_pred.partial_hazard - 1.0),
            "baseline_dmean_gy": baseline_features[DM_FEATURE],
            "counterfactual_dmean_gy": new_features[DM_FEATURE],
            "dmean_percent_change": 100.0 * (new_features[DM_FEATURE] - baseline_features[DM_FEATURE]) / baseline_features[DM_FEATURE],
            "median_voxel_factor_in_thyroid": float(np.median(roi_factors)) if roi_factors.size else np.nan,
            "q05_voxel_factor_in_thyroid": float(np.quantile(roi_factors, 0.05)) if roi_factors.size else np.nan,
            "q95_voxel_factor_in_thyroid": float(np.quantile(roi_factors, 0.95)) if roi_factors.size else np.nan,
        }
        for horizon in TIME_LABELS:
            base_risk = float(base_pred.risk_curve.loc[base_pred.risk_curve["horizon"].eq(horizon), "risk"].iloc[0])
            new_risk = float(new_pred.risk_curve.loc[new_pred.risk_curve["horizon"].eq(horizon), "risk"].iloc[0])
            row[f"baseline_risk_{horizon}"] = base_risk
            row[f"counterfactual_risk_{horizon}"] = new_risk
            row[f"delta_risk_{horizon}"] = new_risk - base_risk
        for feature, old_value in baseline_features.items():
            if feature.startswith("dosiomics_"):
                new_value = new_features.get(feature, np.nan)
                row[f"{feature}__baseline"] = old_value
                row[f"{feature}__counterfactual"] = new_value
                row[f"{feature}__percent_change"] = 100.0 * (new_value - old_value) / old_value if old_value else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
