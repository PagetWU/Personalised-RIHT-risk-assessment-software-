from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .autoseg import run_thyroid_autoseg
from .counterfactual import counterfactual_audit
from .features import extract_model_features, prepare_images, compute_full_dvh_metrics
from .io_utils import discover_case_files
from .model import CoxRIHTModel
from .ntcp_models import reproduce_ntcp_models
from .optimizability import optimizability_metrics
from .report import make_zoom_png, write_html_report


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = PACKAGE_ROOT / "model_assets_no_volume"


def image_format(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii.gz"):
        return "nii.gz"
    if path.suffix.lower() == ".nii":
        return "nii"
    if path.suffix.lower() == ".mha":
        return "mha"
    return path.suffix.lower().lstrip(".") or "unknown"


def dataframe_from_features(features: dict[str, float], scaled: dict[str, float]) -> pd.DataFrame:
    rows = []
    for feature, raw_value in features.items():
        if feature == "Thyroid volume (cc)":
            continue
        rows.append({"feature": feature, "raw_value": raw_value, "scaled_value": scaled.get(feature)})
    return pd.DataFrame(rows)


def run_predict(args: argparse.Namespace) -> None:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_files = discover_case_files(args.case_dir, thyroid_mask_path=args.thyroid_mask_path, require_thyroid=False)
    autoseg_result = None
    thyroid_mask_source = "provided"
    if args.force_auto_segment_thyroid or case_files.thyroid_mask_path is None:
        if args.no_auto_segment_thyroid:
            raise FileNotFoundError("No thyroid mask was found and automatic thyroid segmentation was disabled.")
        autoseg_dir = Path(args.autoseg_output_dir) if args.autoseg_output_dir else out_dir / "autoseg"
        autoseg_result = run_thyroid_autoseg(
            args.case_dir,
            autoseg_dir,
            case_id=args.case_id,
            script_path=args.autoseg_script,
            model_folder=args.autoseg_model_folder,
        )
        case_files = discover_case_files(args.case_dir, thyroid_mask_path=autoseg_result.mask_path, require_thyroid=True)
        thyroid_mask_source = "auto-segmented"
    images = prepare_images(case_files.ct_path, case_files.dose_path, case_files.thyroid_mask_path, args.dose_scale_to_cgy)
    features = extract_model_features(images, age=args.age, n_stage=args.n_stage)
    dvh_metrics = compute_full_dvh_metrics(images.dose_cgy_image, images.thyroid_on_dose, images.ct_image, images.thyroid_on_ct)
    model = CoxRIHTModel(args.asset_dir)
    prediction = model.predict(features)
    feature_df = dataframe_from_features(features, prediction.scaled_features)
    dvh_df = pd.DataFrame([{"metric": key, "value": value} for key, value in dvh_metrics.items()])
    ntcp_df = reproduce_ntcp_models(dvh_metrics, n_stage=args.n_stage)
    hotspot_threshold_cgy = float(args.hotspot_threshold_gy) * 100.0
    optim_df = optimizability_metrics(
        images.dose_cgy_image,
        images.thyroid_on_dose,
        case_files.target_mask_paths,
        hotspot_threshold_cgy=hotspot_threshold_cgy,
    )
    cf_df = counterfactual_audit(
        model,
        features,
        images.dose_cgy_image,
        images.thyroid_on_dose,
        hotspot_threshold_cgy=hotspot_threshold_cgy,
    )

    feature_df.to_csv(out_dir / "patient_features.csv", index=False, encoding="utf-8-sig")
    dvh_df.to_csv(out_dir / "dvh_metrics.csv", index=False, encoding="utf-8-sig")
    ntcp_df.to_csv(out_dir / "other_model_reproduction.csv", index=False, encoding="utf-8-sig")
    prediction.risk_curve.to_csv(out_dir / "risk_curve.csv", index=False, encoding="utf-8-sig")
    optim_df.to_csv(out_dir / "target_adjacency_metrics.csv", index=False, encoding="utf-8-sig")
    cf_df.to_csv(out_dir / "dose_optimization_audit.csv", index=False, encoding="utf-8-sig")

    zoom_path = out_dir / "thyroid_zoom.png"
    make_zoom_png(
        images.ct_image,
        images.dose_cgy_image,
        images.thyroid_on_dose,
        case_files.target_mask_paths,
        zoom_path,
        ct_window_level=args.ct_window_level,
        ct_window_width=args.ct_window_width,
        hotspot_threshold_cgy=hotspot_threshold_cgy,
    )

    risk_lookup = dict(zip(prediction.risk_curve["horizon"], prediction.risk_curve["risk"]))
    risk_group = {
        "persistent_low": "low-risk",
        "intermediate": "intermediate-risk",
        "late_accumulating": "high-risk",
        "early_accelerating": "high-risk",
    }.get(prediction.phenotype, "undetermined")
    summary = {
        "case_dir": str(case_files.case_dir),
        "ct_path": str(case_files.ct_path),
        "dose_path": str(case_files.dose_path),
        "thyroid_mask_path": str(case_files.thyroid_mask_path),
        "thyroid_mask_source": thyroid_mask_source,
        "target_mask_paths": [str(path) for path in case_files.target_mask_paths],
        "model_id": model.spec.get("model_id"),
        "model_label": model.spec.get("model_label"),
        "age": float(args.age),
        "gender": str(args.gender),
        "n_stage": float(args.n_stage),
        "auto_segmentation_used": bool(autoseg_result is not None),
        "input_image_format": {
            "ct": image_format(case_files.ct_path),
            "dose": image_format(case_files.dose_path),
            "thyroid_mask": image_format(case_files.thyroid_mask_path),
        },
        "ct_window_level": float(args.ct_window_level),
        "ct_window_width": float(args.ct_window_width),
        "hotspot_threshold_gy": float(args.hotspot_threshold_gy),
        "hotspot_threshold_cgy": hotspot_threshold_cgy,
        "dose_scale_to_cgy": images.dose_scale_to_cgy,
        "linear_predictor": prediction.linear_predictor,
        "partial_hazard": prediction.partial_hazard,
        "phenotype": prediction.phenotype,
        "risk_group": risk_group,
        "risk_1y": float(risk_lookup["1y"]),
        "risk_3y": float(risk_lookup["3y"]),
        "risk_7y": float(risk_lookup["7y"]),
        "risk_all": float(risk_lookup["all"]),
        "optimizable_class": str(optim_df["optimizable_class"].iloc[0]),
        "target_available": bool(optim_df["target_available"].iloc[0]),
        "outputs": {
            "patient_features": "patient_features.csv",
            "dvh_metrics": "dvh_metrics.csv",
            "other_model_reproduction": "other_model_reproduction.csv",
            "risk_curve": "risk_curve.csv",
            "target_adjacency_metrics": "target_adjacency_metrics.csv",
            "dose_optimization_audit": "dose_optimization_audit.csv",
            "thyroid_zoom": "thyroid_zoom.png",
            "html_report": "RIHT_demo_report.html",
        },
    }
    if autoseg_result is not None:
        summary["auto_segmentation"] = {
            "stage_dir": str(autoseg_result.stage_dir),
            "mask_path": str(autoseg_result.mask_path),
            "geometry_json": str(autoseg_result.geometry_json),
            "geometry_summary": autoseg_result.geometry_summary,
        }
    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(
        out_dir / "RIHT_demo_report.html",
        summary,
        feature_df,
        dvh_df,
        ntcp_df,
        prediction.risk_curve,
        optim_df,
        cf_df,
        zoom_path.name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RIHT single-patient demo toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Run single-patient RIHT prediction and report generation.")
    predict.add_argument("--case-dir", required=True, help="Folder containing CT, dose, thyroid mask, and optional target masks.")
    predict.add_argument("--age", required=True, type=float, help="Patient age.")
    predict.add_argument("--gender", default="Unknown", help="Patient gender for report display, e.g. Male or Female.")
    predict.add_argument("--n-stage", required=True, type=float, help="N-stage encoded as the training table value, usually 0-3.")
    predict.add_argument("--out-dir", required=True, help="Output report folder.")
    predict.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR), help="Model asset directory.")
    predict.add_argument("--thyroid-mask-path", default=None, help="Optional explicit thyroid mask path. If omitted and no mask is found, auto-segmentation is attempted.")
    predict.add_argument("--force-auto-segment-thyroid", action="store_true", help="Run automatic thyroid segmentation even if a thyroid mask already exists in the case folder.")
    predict.add_argument("--no-auto-segment-thyroid", action="store_true", help="Disable automatic thyroid segmentation when no thyroid mask is present.")
    predict.add_argument("--autoseg-output-dir", default=None, help="Optional auto-segmentation working/output directory. Defaults to OUT_DIR/autoseg.")
    predict.add_argument("--case-id", default=None, help="Optional case id for auto-segmentation outputs.")
    predict.add_argument("--autoseg-script", default=None, help="Optional path to run_thyroid_segmentation_pipeline.ps1.")
    predict.add_argument("--autoseg-model-folder", default=None, help="Optional path to the packaged Dataset1102 nnU-Net model folder.")
    predict.add_argument("--dose-scale-to-cgy", type=float, default=None, help="Optional dose scale override. If omitted, Gy-like maps are multiplied by 100.")
    predict.add_argument("--ct-window-level", type=float, default=50.0, help="CT display window level for the report montage. Default: 50.")
    predict.add_argument("--ct-window-width", type=float, default=400.0, help="CT display window width for the report montage. Default: 400.")
    predict.add_argument("--hotspot-threshold-gy", type=float, default=40.0, help="Dose threshold for the thyroid hotspot contour, optimizability audit, and threshold-based sparing scenarios. Default: 40 Gy.")
    predict.set_defaults(func=run_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
