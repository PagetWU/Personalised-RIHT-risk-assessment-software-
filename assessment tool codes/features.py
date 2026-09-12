from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

from .io_utils import array, dose_scale_to_cgy, image_from_array, read_image, resample_mask, voxel_volume_cc


RADIOMICS_FEATURES = {
    "radiomics_log-sigma-3-mm-3D_glrlm_GrayLevelNonUniformity": "log-sigma-3-mm-3D_glrlm_GrayLevelNonUniformity",
    "radiomics_original_gldm_GrayLevelNonUniformity": "original_gldm_GrayLevelNonUniformity",
    "radiomics_wavelet-LHL_ngtdm_Strength": "wavelet-LHL_ngtdm_Strength",
}

DOSIOMICS_FEATURES = {
    "dosiomics_original_shape_MeshVolume": "original_shape_MeshVolume",
    "dosiomics_wavelet-HLH_gldm_GrayLevelNonUniformity": "wavelet-HLH_gldm_GrayLevelNonUniformity",
    "dosiomics_wavelet-LLH_gldm_GrayLevelNonUniformity": "wavelet-LLH_gldm_GrayLevelNonUniformity",
    "dosiomics_wavelet-HLL_ngtdm_Coarseness": "wavelet-HLL_ngtdm_Coarseness",
}

DM_FEATURE = "Thyroid Dmean (Gy)"
VOLUME_FEATURE = "Thyroid volume (cc)"


@dataclass
class ExtractedImages:
    ct_image: sitk.Image
    dose_cgy_image: sitk.Image
    thyroid_on_ct: sitk.Image
    thyroid_on_dose: sitk.Image
    dose_scale_to_cgy: float


def make_radiomics_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        "binCount": 32,
        "interpolator": "sitkLinear",
        "resampledPixelSpacing": None,
        "weightingNorm": None,
        "voxelArrayShift": 0,
        "additionalInfo": False,
        "correctMask": False,
        "minimumROIDimensions": 3,
        "geometryTolerance": 1e-3,
        "label": 1,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1, 2, 3, 4, 5]})
    extractor.enableImageTypeByName("Wavelet")
    extractor.disableAllFeatures()
    for feature_class in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def make_dosiomics_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        "binCount": 32,
        "interpolator": "sitkLinear",
        "resampledPixelSpacing": None,
        "weightingNorm": None,
        "voxelArrayShift": 0,
        "additionalInfo": False,
        "correctMask": False,
        "minimumROIDimensions": 3,
        "geometryTolerance": 1e-3,
        "label": 1,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape=["MeshVolume"], gldm=["GrayLevelNonUniformity"], ngtdm=["Coarseness"])
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    return extractor


def prepare_images(ct_path, dose_path, thyroid_mask_path, dose_scale: float | None = None) -> ExtractedImages:
    ct = read_image(ct_path)
    raw_dose = read_image(dose_path)
    scale = dose_scale_to_cgy(raw_dose, dose_scale)
    dose_cgy = image_from_array(array(raw_dose).astype(np.float32) * scale, raw_dose)
    thyroid_raw = read_image(thyroid_mask_path)
    thyroid_ct = resample_mask(thyroid_raw, ct)
    thyroid_dose = resample_mask(thyroid_raw, dose_cgy)
    return ExtractedImages(ct, dose_cgy, thyroid_ct, thyroid_dose, scale)


def extract_named(result: dict, mapping: dict[str, str]) -> dict[str, float]:
    out = {}
    for model_name, pyrad_name in mapping.items():
        out[model_name] = float(result[pyrad_name]) if pyrad_name in result else np.nan
    return out


def extract_radiomics(ct_image: sitk.Image, thyroid_mask: sitk.Image) -> dict[str, float]:
    result = make_radiomics_extractor().execute(ct_image, thyroid_mask)
    return extract_named(result, RADIOMICS_FEATURES)


def extract_dosiomics(dose_cgy_image: sitk.Image, thyroid_mask: sitk.Image) -> dict[str, float]:
    result = make_dosiomics_extractor().execute(dose_cgy_image, thyroid_mask)
    return extract_named(result, DOSIOMICS_FEATURES)


def compute_dvh(
    dose_cgy_image: sitk.Image,
    thyroid_mask_on_dose: sitk.Image,
    volume_reference_image: sitk.Image | None = None,
    thyroid_mask_for_volume: sitk.Image | None = None,
) -> dict[str, float]:
    dose = array(dose_cgy_image).astype(np.float32)
    dose_mask = array(thyroid_mask_on_dose) > 0
    values = dose[dose_mask]
    volume_image = volume_reference_image if volume_reference_image is not None else dose_cgy_image
    volume_mask_image = thyroid_mask_for_volume if thyroid_mask_for_volume is not None else thyroid_mask_on_dose
    volume_mask = array(volume_mask_image) > 0
    return {
        DM_FEATURE: float(np.nanmean(values) / 100.0) if values.size else np.nan,
        VOLUME_FEATURE: float(volume_mask.sum() * voxel_volume_cc(volume_image)),
    }


def compute_full_dvh_metrics(
    dose_cgy_image: sitk.Image,
    thyroid_mask_on_dose: sitk.Image,
    volume_reference_image: sitk.Image | None = None,
    thyroid_mask_for_volume: sitk.Image | None = None,
) -> dict[str, float]:
    dose = array(dose_cgy_image).astype(np.float32)
    mask = array(thyroid_mask_on_dose) > 0
    values_cgy = dose[mask]
    base = compute_dvh(dose_cgy_image, thyroid_mask_on_dose, volume_reference_image, thyroid_mask_for_volume)
    out = dict(base)
    if not values_cgy.size:
        return out
    values_gy = values_cgy / 100.0
    out.update(
        {
            "Thyroid Dmax (Gy)": float(np.nanmax(values_gy)),
            "Thyroid Dmin (Gy)": float(np.nanmin(values_gy)),
            "Thyroid D2 (Gy)": float(np.nanpercentile(values_gy, 98)),
            "Thyroid D50 (Gy)": float(np.nanpercentile(values_gy, 50)),
            "Thyroid D95 (Gy)": float(np.nanpercentile(values_gy, 5)),
        }
    )
    for threshold in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70):
        out[f"Thyroid V{threshold} (%)"] = float(np.mean(values_gy >= threshold) * 100.0)
        out[f"Thyroid V{threshold} (cc)"] = float(out[VOLUME_FEATURE] * out[f"Thyroid V{threshold} (%)"] / 100.0)
        out[f"Thyroid VS{threshold} (cc)"] = float(out[VOLUME_FEATURE] - out[f"Thyroid V{threshold} (cc)"])
    out["Thyroid V30-60 (%)"] = float(max(0.0, out.get("Thyroid V30 (%)", np.nan) - out.get("Thyroid V60 (%)", np.nan)))
    out["Thyroid V30-60 (cc)"] = float(out[VOLUME_FEATURE] * out["Thyroid V30-60 (%)"] / 100.0)
    return out


def extract_model_features(images: ExtractedImages, age: float, n_stage: float) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(extract_radiomics(images.ct_image, images.thyroid_on_ct))
    features.update(extract_dosiomics(images.dose_cgy_image, images.thyroid_on_dose))
    features["Age"] = float(age)
    features["N-stage"] = float(n_stage)
    features.update(compute_dvh(images.dose_cgy_image, images.thyroid_on_dose, images.ct_image, images.thyroid_on_ct))
    return features
