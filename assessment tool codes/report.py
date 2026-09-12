from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .io_utils import array, read_image, resample_image, resample_mask
from .optimizability import HOTSPOT_THRESHOLD_CGY, target_union


DOSE_BLUE_RED_CMAP = LinearSegmentedColormap.from_list(
    "dose_blue_red",
    ["#08306b", "#08519c", "#ffffff", "#fb6a4a", "#a50f15"],
)


def _bbox(mask: np.ndarray, spacing_xyz: tuple[float, float, float], margin_mm: float = 25.0):
    coords = np.argwhere(mask)
    if coords.size == 0:
        z, y, x = mask.shape
        return slice(0, z), slice(0, y), slice(0, x)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    sx, sy, sz = spacing_xyz
    mx = int(np.ceil(margin_mm / sx))
    my = int(np.ceil(margin_mm / sy))
    mz = int(np.ceil(8.0 / sz))
    return (
        slice(max(0, zmin - mz), min(mask.shape[0], zmax + mz + 1)),
        slice(max(0, ymin - my), min(mask.shape[1], ymax + my + 1)),
        slice(max(0, xmin - mx), min(mask.shape[2], xmax + mx + 1)),
    )


def _slice_indices(thyroid: np.ndarray, hotspot: np.ndarray, target: np.ndarray) -> list[int]:
    score = thyroid.astype(int) + 5 * hotspot.astype(int) + 2 * (target & thyroid).astype(int)
    per_slice = score.sum(axis=(1, 2))
    z = int(np.argmax(per_slice)) if per_slice.max() > 0 else thyroid.shape[0] // 2
    return sorted(set([max(0, z - 2), z, min(thyroid.shape[0] - 1, z + 2)]))


def _contour(ax, mask: np.ndarray, color: str, label: str, lw: float = 1.5) -> None:
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)
        ax.plot([], [], color=color, linewidth=lw, label=label)


def _clean_image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _target_masks_by_type(target_paths: tuple[Path, ...], reference: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    shape = array(reference).shape
    ptv_ctv = np.zeros(shape, dtype=bool)
    gtv_other = np.zeros(shape, dtype=bool)
    for path in target_paths:
        try:
            values = array(resample_mask(read_image(path), reference)) > 0
        except Exception:
            continue
        upper_name = path.name.upper()
        if "PTV" in upper_name or "CTV" in upper_name:
            ptv_ctv |= values
        elif "GTV" in upper_name:
            gtv_other |= values
    return ptv_ctv, gtv_other


def make_zoom_png(
    ct_image: sitk.Image,
    dose_cgy_image: sitk.Image,
    thyroid_on_dose: sitk.Image,
    target_paths: tuple[Path, ...],
    out_path: Path,
    ct_window_level: float = 50.0,
    ct_window_width: float = 400.0,
    hotspot_threshold_cgy: float = HOTSPOT_THRESHOLD_CGY,
) -> None:
    ct_on_dose = resample_image(ct_image, dose_cgy_image, sitk.sitkLinear, -1000)
    ct = array(ct_on_dose).astype(np.float32)
    dose = array(dose_cgy_image).astype(np.float32)
    thyroid = array(thyroid_on_dose) > 0
    target, _names = target_union(target_paths, dose_cgy_image)
    ptv_ctv, gtv_other = _target_masks_by_type(target_paths, dose_cgy_image)
    hotspot_threshold_cgy = float(hotspot_threshold_cgy)
    hotspot_threshold_gy = hotspot_threshold_cgy / 100.0
    hotspot = thyroid & (dose >= hotspot_threshold_cgy)
    zsl, ysl, xsl = _bbox(thyroid | hotspot, dose_cgy_image.GetSpacing(), margin_mm=10.0)
    slices = _slice_indices(thyroid, hotspot, target)
    dose_vmax = 7000.0
    fig, axes = plt.subplots(2, len(slices), figsize=(4.0 * len(slices), 7.0), dpi=180)
    if len(slices) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    ct_level = float(ct_window_level)
    ct_width = float(ct_window_width)
    ct_vmin = ct_level - ct_width / 2.0
    ct_vmax = ct_level + ct_width / 2.0
    dose_vmax_gy = dose_vmax / 100.0
    dose_im = None
    for col, z in enumerate(slices):
        ax = axes[0, col]
        ct2d = ct[z, ysl, xsl]
        dose2d = dose[z, ysl, xsl]
        ax.imshow(ct2d, cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
        _contour(ax, gtv_other[z, ysl, xsl], "#ff3b30", "GTV")
        _contour(ax, ptv_ctv[z, ysl, xsl], "#22c55e", "PTV/CTV")
        _contour(ax, thyroid[z, ysl, xsl], "#ffd400", "thyroid", lw=1.8)
        _contour(ax, hotspot[z, ysl, xsl], "#ffffff", f"thyroid >={hotspot_threshold_gy:g}Gy", lw=1.2)
        ax.set_title(f"CT soft-tissue window z={z}", fontsize=8)
        _clean_image_axis(ax)
        ax2 = axes[1, col]
        dose_im = ax2.imshow(dose2d / 100.0, cmap=DOSE_BLUE_RED_CMAP, vmin=0, vmax=dose_vmax_gy)
        _contour(ax2, gtv_other[z, ysl, xsl], "#ff3b30", "GTV")
        _contour(ax2, ptv_ctv[z, ysl, xsl], "#22c55e", "PTV/CTV")
        _contour(ax2, thyroid[z, ysl, xsl], "#ffd400", "thyroid", lw=1.8)
        _contour(ax2, hotspot[z, ysl, xsl], "#ffffff", f"thyroid >={hotspot_threshold_gy:g}Gy", lw=1.2)
        ax2.set_title(f"Dose map z={z} (0-{dose_vmax / 100:.1f} Gy)", fontsize=8)
        _clean_image_axis(ax2)
    axes[0, 0].legend(loc="lower left", fontsize=6, frameon=True)
    fig.subplots_adjust(right=0.88, wspace=0.035, hspace=0.18)
    cax_ct = fig.add_axes([0.905, 0.57, 0.012, 0.29])
    ct_mappable = ScalarMappable(norm=Normalize(vmin=ct_vmin, vmax=ct_vmax), cmap="gray")
    cbar_ct = fig.colorbar(ct_mappable, cax=cax_ct)
    cbar_ct.set_label(f"CT (HU)\nWL {ct_level:g} / WW {ct_width:g}", fontsize=7)
    cbar_ct.set_ticks([ct_vmin, ct_level, ct_vmax])
    cbar_ct.ax.tick_params(labelsize=6)
    if dose_im is not None:
        cax_dose = fig.add_axes([0.905, 0.12, 0.012, 0.29])
        cbar = fig.colorbar(dose_im, cax=cax_dose)
        cbar.set_label(f"Dose (Gy)\n0-{dose_vmax_gy:.0f}", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    view = df.copy().head(max_rows)
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return f"<div class='table-wrap'>{view.to_html(index=False, escape=False, classes='data-table')}</div>"


def _metric_lookup(df: pd.DataFrame) -> dict[str, float]:
    if "metric" not in df.columns or "value" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            out[str(row["metric"])] = float(row["value"])
        except Exception:
            continue
    return out


def _fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    try:
        number = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}{suffix}"


def _risk_svg(risk_curve: pd.DataFrame) -> str:
    rows = risk_curve.copy()
    rows = rows[rows["horizon"].astype(str).ne("all")].copy()
    if rows.empty:
        return ""
    xs = rows["horizon"].astype(str).str.replace("y", "", regex=False).astype(float).to_numpy()
    ys = pd.to_numeric(rows["risk"], errors="coerce").fillna(0).to_numpy()
    width, height = 760, 250
    left, right, top, bottom = 54, 18, 18, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 1.0, 9.0
    y_min, y_max = 0.0, 1.0

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    points = " ".join(f"{px(float(x)):.1f},{py(float(y)):.1f}" for x, y in zip(xs, ys))
    circles = "".join(
        f"<circle cx='{px(float(x)):.1f}' cy='{py(float(y)):.1f}' r='4'><title>{int(x)}y risk {_fmt(y)}</title></circle>"
        for x, y in zip(xs, ys)
    )
    x_ticks = "".join(
        f"<line x1='{px(x):.1f}' y1='{top}' x2='{px(x):.1f}' y2='{top + plot_h}' class='grid'/>"
        f"<text x='{px(x):.1f}' y='{height - 16}' text-anchor='middle'>{int(x)}y</text>"
        for x in range(1, 10)
    )
    y_ticks = "".join(
        f"<line x1='{left}' y1='{py(y):.1f}' x2='{left + plot_w}' y2='{py(y):.1f}' class='grid'/>"
        f"<text x='{left - 10}' y='{py(y) + 4:.1f}' text-anchor='end'>{int(y * 100)}%</text>"
        for y in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return f"""
<svg class="risk-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Predicted cumulative RIHT risk curve">
  <style>
    .risk-svg text {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .risk-svg .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .risk-svg .axis {{ stroke: #374151; stroke-width: 1.2; }}
    .risk-svg polyline {{ fill: none; stroke: #2563eb; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .risk-svg circle {{ fill: #2563eb; stroke: white; stroke-width: 1.4; }}
  </style>
  {x_ticks}
  {y_ticks}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
  <polyline points="{points}"/>
  {circles}
</svg>
"""


def _ntcp_display_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    view = df.copy()
    if "value" in view.columns:
        view["value"] = view["value"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    cols = [c for c in ["model", "value", "risk_call", "paper", "formula", "note"] if c in view.columns]
    return _table(view[cols], 200)


def _risk_group(summary: dict) -> str:
    return str(summary.get("risk_group", "")).replace("_", "-")


def _input_format_text(summary: dict) -> str:
    formats = summary.get("input_image_format", {})
    if not isinstance(formats, dict) or not formats:
        return "Unknown"
    values = [formats.get(key) for key in ("ct", "dose", "thyroid_mask") if formats.get(key)]
    unique_values = list(dict.fromkeys(values))
    if len(unique_values) == 1:
        return str(unique_values[0])
    return " / ".join(str(value) for value in unique_values) if unique_values else "Unknown"


def _key_dvh_cards(dvh_metrics: pd.DataFrame, summary: dict | None = None) -> str:
    metrics = _metric_lookup(dvh_metrics)
    summary = summary or {}
    items = [
        ("Age", _fmt(summary.get("age"), 0, " y")),
        ("Gender", str(summary.get("gender", "Unknown"))),
        ("Dmean", _fmt(metrics.get("Thyroid Dmean (Gy)"), 2, " Gy")),
        ("Volume", _fmt(metrics.get("Thyroid volume (cc)"), 2, " cc")),
        ("Dmax", _fmt(metrics.get("Thyroid Dmax (Gy)"), 2, " Gy")),
        ("V30", _fmt(metrics.get("Thyroid V30 (%)"), 1, "%")),
        ("V40", _fmt(metrics.get("Thyroid V40 (%)"), 1, "%")),
        ("V50", _fmt(metrics.get("Thyroid V50 (%)"), 1, "%")),
    ]
    cards = "".join(f"<div class='mini-card'><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>" for label, value in items)
    return f"<div class='mini-grid'>{cards}</div>"


def _interpretation(summary: dict) -> str:
    phenotype = str(summary.get("phenotype", ""))
    optimizable = str(summary.get("optimizable_class", ""))
    pheno_text = {
        "persistent_low": "Predicted cumulative risk remains comparatively low across the observed horizon.",
        "intermediate": "Predicted risk is intermediate; interpretation should emphasize calibration and follow-up context.",
        "late_accumulating": "Predicted risk is low at 3 years but accumulates by 7 years, consistent with a delayed-risk pattern.",
        "early_accelerating": "Predicted risk is already high by 3 years and remains high at later horizons.",
    }.get(phenotype, "Temporal phenotype is unavailable.")
    opt_text = {
        "non_optimizable_target_adjacent": "Thyroid hotspot/thyroid mask is adjacent to or overlaps target masks, so dose sparing may be anatomically constrained.",
        "limited_optimizable_near_target": "Hotspot is near target masks; only limited dose-sparing room should be assumed.",
        "potentially_optimizable_dose_tail": "Pattern is compatible with a dose-tail component, so counterfactual dose-sparing may be worth review.",
        "indeterminate_missing_target": "Target masks are unavailable, so optimizability cannot be determined from anatomy.",
    }.get(optimizable, "Optimizability class is unavailable.")
    return f"<p>{html.escape(pheno_text)}</p><p>{html.escape(opt_text)}</p>"


def _display_optimizability_class(raw_class: object) -> str:
    raw = str(raw_class)
    if raw in {"limited_optimizable_near_target", "potentially_optimizable_dose_tail"}:
        return "optimizable"
    return "non-optimizable"


def _optimizability_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    row = df.iloc[0]
    cards = [
        ("Class", _display_optimizability_class(row.get("optimizable_class", ""))),
        ("Target masks", f"{int(row.get('target_mask_count', 0))}" if pd.notna(row.get("target_mask_count", np.nan)) else "0"),
        ("Overlap", _fmt(row.get("thyroid_target_overlap_cc"), 2, " cc")),
        ("Overlap %", _fmt(row.get("thyroid_target_overlap_pct_of_thyroid"), 1, "%")),
        ("Thyroid-target distance", _fmt(row.get("thyroid_to_target_min_distance_mm"), 1, " mm")),
        ("Hotspot-target distance", _fmt(row.get("hotspot_to_target_min_distance_mm"), 1, " mm")),
    ]
    html_cards = "".join(f"<div class='mini-card'><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>" for k, v in cards)
    return f"<div class='mini-grid'>{html_cards}</div>"


def _counterfactual_display(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    view = df.copy()
    for horizon in ("3y", "7y", "all"):
        delta_col = f"delta_risk_{horizon}"
        if delta_col in view.columns:
            view[f"risk_reduction_{horizon}_pp"] = view[delta_col].map(
                lambda x: np.nan if pd.isna(x) else max(0.0, -float(x) * 100.0)
            )
    keep = [
        "scenario_display",
        "baseline_risk_3y",
        "counterfactual_risk_3y",
        "delta_risk_3y",
        "risk_reduction_3y_pp",
        "baseline_risk_7y",
        "counterfactual_risk_7y",
        "delta_risk_7y",
        "risk_reduction_7y_pp",
        "baseline_risk_all",
        "counterfactual_risk_all",
        "delta_risk_all",
        "risk_reduction_all_pp",
        "dmean_percent_change",
    ]
    keep = [col for col in keep if col in view.columns]
    view = view[keep].copy()
    rename = {
        "scenario_display": "scenario",
        "baseline_risk_3y": "base 3y",
        "counterfactual_risk_3y": "cf 3y",
        "delta_risk_3y": "delta 3y",
        "risk_reduction_3y_pp": "3y reduction pp",
        "baseline_risk_7y": "base 7y",
        "counterfactual_risk_7y": "cf 7y",
        "delta_risk_7y": "delta 7y",
        "risk_reduction_7y_pp": "7y reduction pp",
        "baseline_risk_all": "base all",
        "counterfactual_risk_all": "cf all",
        "delta_risk_all": "delta all",
        "risk_reduction_all_pp": "all reduction pp",
        "dmean_percent_change": "Dmean change %",
    }
    return _table(view.rename(columns=rename), 100)


def _counterfactual_curve_svg(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    labels = [f"{i}y" for i in range(1, 10)]
    width, height = 820, 320
    left, right, top, bottom = 58, 190, 20, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "Baseline": "#111827",
        "Whole dose map x0.90": "#2563eb",
        "Whole dose map x0.80": "#16a34a",
    }

    def series_color(name: str, idx: int) -> str:
        if name in colors:
            return colors[name]
        if name.startswith("Thyroid voxels >"):
            return "#dc2626"
        if name.startswith("Soft >"):
            return "#9333ea"
        return ["#0f766e", "#b45309", "#475569", "#be185d"][idx % 4]

    def px(index: int) -> float:
        return left + index / 8.0 * plot_w

    def py(risk: float) -> float:
        return top + (1.0 - float(np.clip(risk, 0, 1))) * plot_h

    series: list[tuple[str, list[float]]] = []
    first = df.iloc[0]
    base = [float(first.get(f"baseline_risk_{label}", np.nan)) for label in labels]
    if all(np.isfinite(base)):
        series.append(("Baseline", base))
    for _, row in df.iterrows():
        values = [float(row.get(f"counterfactual_risk_{label}", np.nan)) for label in labels]
        if all(np.isfinite(values)):
            series.append((str(row.get("scenario_display", row.get("scenario", "scenario"))), values))

    grid = "".join(
        f"<line x1='{px(i):.1f}' y1='{top}' x2='{px(i):.1f}' y2='{top + plot_h}' class='grid'/>"
        f"<text x='{px(i):.1f}' y='{height - 16}' text-anchor='middle'>{i + 1}y</text>"
        for i in range(9)
    )
    grid += "".join(
        f"<line x1='{left}' y1='{py(y):.1f}' x2='{left + plot_w}' y2='{py(y):.1f}' class='grid'/>"
        f"<text x='{left - 10}' y='{py(y) + 4:.1f}' text-anchor='end'>{int(y * 100)}%</text>"
        for y in (0, 0.25, 0.5, 0.75, 1.0)
    )
    lines = []
    legend = []
    for idx, (name, values) in enumerate(series):
        color = series_color(name, idx)
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        lines.append(f"<polyline points='{pts}' style='stroke:{color}'><title>{html.escape(name)}</title></polyline>")
        y = top + 18 + idx * 22
        legend.append(
            f"<line x1='{left + plot_w + 22}' y1='{y}' x2='{left + plot_w + 44}' y2='{y}' style='stroke:{color};stroke-width:3'/>"
            f"<text x='{left + plot_w + 50}' y='{y + 4}'>{html.escape(name[:36])}</text>"
        )
    return f"""
<svg class="cf-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Counterfactual risk curves">
  <style>
    .cf-svg text {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .cf-svg .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .cf-svg .axis {{ stroke: #374151; stroke-width: 1.2; }}
    .cf-svg polyline {{ fill: none; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
  </style>
  {grid}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
  {''.join(lines)}
  {''.join(legend)}
</svg>
"""


def _risk_reduction_bar_svg(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    horizons = [("3y", "3y"), ("7y", "7y"), ("all", "all")]
    rows: list[tuple[str, list[float]]] = []
    for _, row in df.iterrows():
        values = []
        for key, _ in horizons:
            delta = float(row.get(f"delta_risk_{key}", np.nan))
            values.append(max(0.0, -delta * 100.0) if np.isfinite(delta) else np.nan)
        if any(np.isfinite(v) for v in values):
            rows.append((str(row.get("scenario_display", row.get("scenario", "scenario"))), values))
    if not rows:
        return ""

    width = 860
    row_h = 54
    height = 112 + len(rows) * row_h
    left, right, top, bottom = 285, 80, 74, 36
    plot_w = width - left - right
    max_val = max(v for _, vals in rows for v in vals if np.isfinite(v))
    max_axis = max(5.0, math.ceil(max_val / 5.0) * 5.0)
    colors = {"3y": "#2563eb", "7y": "#16a34a", "all": "#dc2626"}

    def x(value: float) -> float:
        return left + (float(value) / max_axis) * plot_w

    grid = ""
    for tick in np.linspace(0, max_axis, 6):
        xx = x(float(tick))
        grid += (
            f"<line x1='{xx:.1f}' y1='{top - 8}' x2='{xx:.1f}' y2='{height - bottom}' class='grid'/>"
            f"<text x='{xx:.1f}' y='{height - 10}' text-anchor='middle'>{tick:.0f}</text>"
        )

    bars = []
    for i, (scenario, values) in enumerate(rows):
        y0 = top + i * row_h
        bars.append(f"<text x='{left - 12}' y='{y0 + 25}' text-anchor='end'>{html.escape(scenario)}</text>")
        for j, ((key, label), value) in enumerate(zip(horizons, values)):
            if not np.isfinite(value):
                continue
            y = y0 + 7 + j * 14
            bar_w = max(1.0, x(value) - left)
            bars.append(
                f"<rect x='{left}' y='{y}' width='{bar_w:.1f}' height='10' rx='2' fill='{colors[key]}'/>"
                f"<text x='{x(value) + 6:.1f}' y='{y + 9}'>{value:.1f} pp</text>"
            )

    legend_x = left
    legend_y = 39
    legend = "".join(
        f"<rect x='{legend_x + i * 82}' y='{legend_y - 10}' width='12' height='12' fill='{colors[key]}'/>"
        f"<text x='{legend_x + i * 82 + 18}' y='{legend_y}'>{label}</text>"
        for i, (key, label) in enumerate(horizons)
    )

    return f"""
<svg class="reduction-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Absolute RIHT risk reduction under hypothetical thyroid-sparing dose scenarios">
  <style>
    .reduction-svg text {{ font: 12px Arial, sans-serif; fill: #374151; }}
    .reduction-svg .title {{ font: 14px Arial, sans-serif; font-weight: 700; fill: #111827; }}
    .reduction-svg .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .reduction-svg .axis {{ stroke: #374151; stroke-width: 1.2; }}
  </style>
  <text x="{left}" y="20" class="title">Predicted RIHT risk reduction (percentage points)</text>
  <text x="{left}" y="57">Higher values indicate larger absolute risk reduction</text>
  {legend}
  {grid}
  <line x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - bottom}" class="axis"/>
  <line x1="{left}" y1="{height - bottom}" x2="{left + plot_w}" y2="{height - bottom}" class="axis"/>
  {''.join(bars)}
</svg>
"""


def write_html_report(
    out_path: Path,
    summary: dict,
    features: pd.DataFrame,
    dvh_metrics: pd.DataFrame,
    ntcp_models: pd.DataFrame,
    risk_curve: pd.DataFrame,
    optimizability: pd.DataFrame,
    counterfactual: pd.DataFrame,
    zoom_png_name: str,
) -> None:
    risk_all = float(summary.get("risk_all", np.nan))
    risk_3y = float(summary.get("risk_3y", np.nan))
    risk_7y = float(summary.get("risk_7y", np.nan))
    opt_class = str(summary.get("optimizable_class", ""))
    phenotype = str(summary.get("phenotype", ""))
    ct_window_level = summary.get("ct_window_level", 50)
    ct_window_width = summary.get("ct_window_width", 400)
    hotspot_threshold_gy = summary.get("hotspot_threshold_gy", 40)
    dvh_lookup = _metric_lookup(dvh_metrics)
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RIHT Demo Report</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 0; color: #1f2937; background: #f5f7fb; }}
.page {{ max-width: 1180px; margin: 0 auto; padding: 26px 28px 48px; }}
.hero {{ background: #111827; color: white; padding: 24px 26px; border-radius: 10px; }}
h1 {{ font-size: 24px; margin: 0 0 8px; letter-spacing: 0; }}
h2 {{ font-size: 17px; margin: 28px 0 10px; }}
h3 {{ font-size: 14px; margin: 0 0 8px; color: #374151; }}
.subtle {{ color: #d1d5db; line-height: 1.45; max-width: 900px; }}
.grid {{ display: grid; grid-template-columns: 1.05fr 1.05fr 1.42fr 1.1fr 0.8fr 0.8fr 0.8fr; gap: 6px; margin-top: 18px; align-items: stretch; }}
.card {{ background: white; color: #111827; border-radius: 8px; padding: 10px 12px; min-height: 70px; min-width: 0; overflow: hidden; }}
.card span, .mini-card span {{ display: block; color: #6b7280; font-size: 12px; margin-bottom: 6px; }}
.card b {{ font-size: 20px; line-height: 1.1; white-space: nowrap; }}
.section {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px 20px; margin-top: 18px; }}
.section p {{ line-height: 1.5; margin: 7px 0; }}
.mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 8px; margin: 8px 0 14px; }}
.mini-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; background: #fafafa; min-width: 0; }}
.mini-card b {{ font-size: 16px; white-space: nowrap; }}
.table-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 8px; }}
table.data-table {{ border-collapse: collapse; width: 100%; font-size: 12px; background: white; }}
table.data-table th, table.data-table td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 9px; text-align: right; vertical-align: top; }}
table.data-table th:first-child, table.data-table td:first-child {{ text-align: left; }}
table.data-table th {{ background: #f3f4f6; color: #374151; font-weight: 700; }}
table.data-table tr:last-child td {{ border-bottom: 0; }}
.risk-panel {{ display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 16px; align-items: start; }}
details {{ margin-top: 8px; }}
summary {{ cursor: pointer; font-weight: 700; color: #374151; }}
pre {{ background: #111827; color: #e5e7eb; padding: 14px; border-radius: 8px; overflow: auto; font-size: 12px; }}
img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
@media (max-width: 980px) {{
  .grid, .mini-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .card b {{ white-space: normal; }}
  .risk-panel {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="page">
<div class="hero">
<h1>RIHT Single-Patient Demo Report</h1>
<div class="grid">
  <div class="card"><span>Auto segmentation</span><b>{'Yes' if summary.get("auto_segmentation_used") else 'No'}</b></div>
  <div class="card"><span>Input format</span><b>{html.escape(_input_format_text(summary))}</b></div>
  <div class="card"><span>Phenotype</span><b>{html.escape(phenotype.replace("_", "-"))}</b></div>
  <div class="card"><span>Risk group</span><b>{html.escape(_risk_group(summary))}</b></div>
  <div class="card"><span>Risk 3y</span><b>{_fmt(risk_3y)}</b></div>
  <div class="card"><span>Risk 7y</span><b>{_fmt(risk_7y)}</b></div>
  <div class="card"><span>Risk all</span><b>{_fmt(risk_all)}</b></div>
</div>
</div>

<div class="section">
<h2>DVH Summary</h2>
{_key_dvh_cards(dvh_metrics, summary)}
<details>
<summary>Show full DVH table</summary>
{_table(dvh_metrics)}
</details>
</div>

<div class="section">
<h2>Risk Curve</h2>
<div class="risk-panel">
  <div>{_risk_svg(risk_curve)}</div>
  <div>{_table(risk_curve)}</div>
</div>
</div>

<div class="section">
<h2>Thyroid Neighborhood</h2>
<p>Yellow=thyroid, red=target, white=thyroid dose &gt;={_fmt(hotspot_threshold_gy, 0)} Gy.</p>
<div class="mini-grid">
  <div class="mini-card"><span>CT display</span><b>WL {_fmt(ct_window_level, 0)} / WW {_fmt(ct_window_width, 0)}</b></div>
  <div class="mini-card"><span>Dose display</span><b>0-70 Gy</b></div>
  <div class="mini-card"><span>Hotspot line</span><b>{_fmt(hotspot_threshold_gy, 0)} Gy</b></div>
</div>
<p><b>Thyroid volume:</b> {_fmt(dvh_lookup.get("Thyroid volume (cc)"), 2, " cc")} &nbsp; <b>Thyroid Dmean:</b> {_fmt(dvh_lookup.get("Thyroid Dmean (Gy)"), 2, " Gy")}</p>
<img src="{html.escape(zoom_png_name)}">
</div>

<div class="section">
<h2>Thyroid-Sparing Optimizability</h2>
{_optimizability_summary(optimizability)}
<details>
<summary>Show full target adjacency metrics</summary>
{_table(optimizability)}
</details>
</div>

<div class="section">
<h2>Risk Reduction Under Hypothetical Thyroid-Sparing Dose Optimization</h2>
{_counterfactual_curve_svg(counterfactual)}
{_risk_reduction_bar_svg(counterfactual)}
{_counterfactual_display(counterfactual)}
</div>

<div class="section">
<h2>Model Features</h2>
{_table(features)}
</div>

<div class="section">
<h2>Published Model Reproduction</h2>
<p>One row per published formula or boundary rule implemented from the Table S2 reproduction code. The risk column shows the patient-level call; <code>fail</code> means the required inputs are unavailable.</p>
{_ntcp_display_table(ntcp_models)}
</div>
</div>
</body>
</html>"""
    out_path.write_text(html_text, encoding="utf-8")
