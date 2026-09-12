# Personalised RIHT Risk Assessment Software

Research software accompanying the multicentre study:

**Development of a cumulative risk model and assessment tool for radiation-induced hypothyroidism: a multicentre study**

This repository provides a single-patient demonstration of cumulative radiation-induced hypothyroidism (RIHT) risk assessment after head-and-neck radiotherapy. It extracts thyroid CT radiomics, dose-map dosiomics, thyroid dose-volume metrics, and clinical variables; applies the fixed Cox model; and generates a patient-level HTML report.

The software is intended for research demonstration and model reproducibility. It is not a certified medical device, a treatment-planning system, or a substitute for clinician review.

## Main Functions

- Read a planning CT, 3D dose distribution, thyroid contour, and optional target contour.
- Extract the features required by the fixed multimodal Cox model.
- Estimate cumulative RIHT risk at multiple follow-up horizons.
- Display the temporal risk group and risk curve.
- Summarise thyroid DVH metrics and published NTCP-style comparators.
- Explore hypothetical thyroid dose-reduction scenarios without modifying the clinical treatment plan.
- Generate a self-contained HTML report and machine-readable output tables.

## Repository Structure

```text
assessment tool codes/
  Core Python package for input handling, feature extraction,
  Cox prediction, comparison models, dose audits, and reporting.

model_assets_parameters/
  Fixed Cox coefficients, scaling parameters, model specification,
  and QEH baseline cumulative hazard.

RIHT Software Demo Case/
  Cropped and de-identified example containing CT, dose,
  thyroid mask, one PTV mask, and example outputs.

autoseg/
  Optional thyroid auto-segmentation helper scripts.

autoseg_model/
  Expected local folder layout for the optional nnU-Net checkpoint.

RIHT_demo_launcher.exe
  Windows demonstration launcher.

requirements.txt
  Python dependencies.
```

## Installation

Python 3.10 or later is recommended.

```powershell
git clone https://github.com/PagetWU/Personalised-RIHT-risk-assessment-software.git
cd Personalised-RIHT-risk-assessment-software
python -m pip install -r requirements.txt
```

The principal dependencies are NumPy, pandas, SciPy, SimpleITK, Matplotlib, and PyRadiomics.

## Run the Included Demonstration Case

From the repository root:

```powershell
python -m "assessment tool codes.cli" predict `
  --case-dir ".\RIHT Software Demo Case" `
  --age 55 `
  --gender Unknown `
  --n-stage 2 `
  --asset-dir ".\model_assets_parameters" `
  --thyroid-mask-path ".\RIHT Software Demo Case\Thyroid_mask.mha" `
  --no-auto-segment-thyroid `
  --out-dir ".\RIHT Software Demo Case\output"
```

The age, gender, and N stage supplied above are synthetic demonstration inputs and are not attributes of the source patient. The included image data are cropped to a thyroid-centred lower-neck region and have had direct identifiers, image metadata, and the original physical origin removed.

The included example has an image-derived thyroid volume of approximately 20.02 cc and a mean thyroid dose of approximately 59.58 Gy.

After processing, open:

```text
RIHT Software Demo Case/output/RIHT_demo_report.html
```

## Run Another Case

A case folder should contain:

```text
CT.mha
RTdose.mha                 or another recognised dose filename
Thyroid_mask.mha           optional when auto-segmentation is configured
PTV_*.mha / CTV_*.mha      optional
GTV_*.mha                  optional
```

Supported image formats are `.mha`, `.nii`, and `.nii.gz`. CT, dose, and masks must share valid physical geometry; the software resamples dose and masks when required. Dose maps may be provided in Gy or cGy. The software automatically treats maps with a maximum value no greater than 120 as Gy-like and converts them to cGy; `--dose-scale-to-cgy` can override this behavior.

Example:

```powershell
python -m "assessment tool codes.cli" predict `
  --case-dir "D:\case001" `
  --age 58 `
  --gender Male `
  --n-stage 2 `
  --asset-dir ".\model_assets_parameters" `
  --out-dir "D:\riht_outputs\case001"
```

If a thyroid mask is stored elsewhere, provide it with `--thyroid-mask-path`. To require a provided thyroid mask and disable automatic segmentation, add `--no-auto-segment-thyroid`.

## Optional Thyroid Auto-Segmentation

Automatic thyroid segmentation is optional. It is used only when no thyroid mask is supplied or when `--force-auto-segment-thyroid` is specified.

The nnU-Net checkpoint is distributed separately because `checkpoint_best.pth` is too large for normal GitHub tracking. The model-asset package is available from [Google Drive](https://drive.google.com/file/d/11w32m51_bM2XIsvkfD604FJQ0GFpoJRB/view?usp=drive_link).

Place the downloaded files in the following layout:

```text
autoseg_model/
  Dataset1102_ThyroidSegmentation/
    nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
      dataset.json
      plans.json
      fold_0/
        checkpoint_best.pth
```

A local nnU-Net v2 environment is required. Generated contours must be reviewed by a qualified clinician before research interpretation.

## Windows Launcher

A prebuilt Windows launcher is included for demonstration. The Python command-line interface remains the canonical and most portable way to run the software. Because the package directory contains spaces, keep the module name in quotation marks as shown above.

The current launcher expects a compatible local Python environment. If its configured Python executable is unavailable, run the command-line example above or edit `PythonExe` in `RIHT_demo_launcher.cs` and rebuild:

```powershell
.\build_launcher.ps1
```

## Main Outputs

| File | Description |
|---|---|
| `RIHT_demo_report.html` | Patient-level report containing risk, DVH, comparison, and dose-audit panels. |
| `prediction_summary.json` | Input paths, model metadata, and principal predictions. |
| `patient_features.csv` | Extracted and scaled model features. |
| `dvh_metrics.csv` | Thyroid dose-volume metrics. |
| `risk_curve.csv` | Estimated cumulative RIHT risk by follow-up horizon. |
| `other_model_reproduction.csv` | Reproduced published NTCP-style estimates. |
| `target_adjacency_metrics.csv` | Thyroid-target proximity and dose-overlap summaries. |
| `dose_optimization_audit.csv` | Hypothetical thyroid dose-reduction scenarios. |
| `thyroid_zoom.png` | CT, thyroid, target, and dose visualisation used in the report. |

## Model

The packaged model uses three CT radiomics features, four dosiomics features, age, N stage, and thyroid mean dose. Model coefficients and preprocessing parameters are fixed in `model_assets_parameters/`; they should not be re-estimated for an individual case.

The output is a model-based estimate of cumulative RIHT risk. Counterfactual dose-reduction results are exploratory simulations rather than deliverable treatment plans.

## Data Protection

Do not upload identifiable patient images or protected health information to a public repository. Users are responsible for confirming that local use, data sharing, and publication comply with institutional approvals and applicable privacy requirements.

## Disclaimer

This software is provided for research, reproducibility, and demonstration only. It has not been cleared or approved for clinical diagnosis, treatment selection, or automated radiotherapy plan modification.
