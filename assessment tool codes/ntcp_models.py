from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(min(float(x), 50.0), -50.0))))


def _get(values: dict[str, float], key: str) -> float:
    value = values.get(key, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan


def reproduce_ntcp_models(dvh: dict[str, float], n_stage: float | None = None) -> pd.DataFrame:
    volume = _get(dvh, "Thyroid volume (cc)")
    dmean = _get(dvh, "Thyroid Dmean (Gy)")
    rows: list[dict[str, Any]] = []

    def add(
        model: str,
        value: float | None,
        status: str,
        formula: str,
        note: str = "",
        risk_call: str | None = None,
        paper: str = "",
    ) -> None:
        numeric = value if value is not None and np.isfinite(value) else np.nan
        display_call = risk_call if risk_call is not None else ("fail" if status != "computed" else "")
        rows.append(
            {
                "model": model,
                "value": numeric,
                "risk_call": display_call,
                "paper": paper,
                "formula": formula,
                "note": note,
            }
        )

    if np.isfinite(dmean) and np.isfinite(volume):
        boomsma = sigmoid(0.011 + 0.062 * dmean - 0.19 * volume)
        add(
            "NTCP_Boomsma2012",
            boomsma,
            "computed",
            "sigmoid(0.011 + 0.062*DmeanGy - 0.19*VolumeCc)",
            "Probability-style NTCP model using Dmean and thyroid volume.",
            risk_call="high-risk" if boomsma >= 0.5 else "low-risk",
            paper="Boomsma et al. 2012",
        )
        ronjom = sigmoid(-1.235 + 0.1162 * dmean - 0.2873 * volume)
        add(
            "NTCP_Ronjom2013",
            ronjom,
            "computed",
            "sigmoid(-1.235 + 0.1162*DmeanGy - 0.2873*VolumeCc)",
            "Probability-style NTCP model using Dmean and thyroid volume.",
            risk_call="high-risk" if ronjom >= 0.5 else "low-risk",
            paper="Ronjom et al. 2013",
        )
    else:
        for model in ("NTCP_Boomsma2012", "NTCP_Ronjom2013"):
            add(model, None, "not available", "requires Thyroid Dmean (Gy) and Thyroid volume (cc)")

    def add_threshold(model: str, metric: str, threshold: float, high_when: str, note: str, paper: str) -> None:
        value = _get(dvh, metric)
        if not np.isfinite(value):
            add(model, None, "not available", f"requires {metric}", note, "", paper)
            return
        if high_when == "ge":
            high = value >= threshold
            formula = f"{metric} >= {threshold:g}"
        elif high_when == "gt":
            high = value > threshold
            formula = f"{metric} > {threshold:g}"
        elif high_when == "lt":
            high = value < threshold
            formula = f"{metric} < {threshold:g}"
        else:
            high = value <= threshold
            formula = f"{metric} <= {threshold:g}"
        add(model, value, "computed", formula, note, "high-risk" if high else "low-risk", paper)

    def add_fixed_score(
        model: str,
        value: float | None,
        formula: str,
        note: str,
        risk_call: str,
        paper: str,
    ) -> None:
        add(model, value, "computed" if value is not None and np.isfinite(value) else "not available", formula, note, risk_call, paper)

    add_threshold(
        "Bakhshandeh2013_Zhai2022_Dmean_ge45",
        "Thyroid Dmean (Gy)",
        45.0,
        "ge",
        "Dose-volume rule: mean thyroid dose >=45 Gy indicates elevated risk.",
        "Bakhshandeh et al. 2013; Zhai et al. 2022",
    )
    add_threshold(
        "Sachdev2014_V50_ge60pct",
        "Thyroid V50 (%)",
        60.0,
        "ge",
        "Dose-volume rule: V50 >=60% has been reported as an elevated-risk threshold.",
        "Sachdev et al. 2014",
    )
    add_threshold(
        "Zhai2022_V40_gt80pct",
        "Thyroid V40 (%)",
        80.0,
        "gt",
        "Planning-style rule: V40 >80% flags elevated thyroid dose burden.",
        "Zhai et al. 2022",
    )
    add_threshold(
        "Lee2016_Zhai2022_VS45_lt5cc",
        "Thyroid VS45 (cc)",
        5.0,
        "lt",
        "Sparing-volume rule: VS45 <5 cc indicates inadequate thyroid sparing.",
        "Lee et al. 2016; Zhai et al. 2022",
    )
    add_threshold(
        "Lee2016_Lertbutsayanukul2018_VS60_lt10cc",
        "Thyroid VS60 (cc)",
        10.0,
        "lt",
        "Sparing-volume rule: VS60 <10 cc indicates inadequate high-dose sparing.",
        "Lee et al. 2016; Lertbutsayanukul et al. 2018",
    )

    add_threshold(
        "Chow2022_VS60_lt10cc",
        "Thyroid VS60 (cc)",
        10.0,
        "lt",
        "Table S2 boundary implementation: high risk if thyroid sparing volume receiving <60 Gy is <10 cc.",
        "Chow et al. 2022",
    )

    add_threshold(
        "Xu2023_V40_gt85_binary",
        "Thyroid V40 (%)",
        85.0,
        "gt",
        "Table S2 boundary implementation: high risk if V40 >85%; V40 <=85% corresponds to the protected constraint group.",
        "Xu et al. 2023",
    )

    add_threshold(
        "Zhou2020_V50_gt24_binary",
        "Thyroid V50 (%)",
        24.0,
        "gt",
        "Table S2 boundary implementation: high risk if thyroid V50 >24%.",
        "Zhou et al. 2020",
    )

    v50 = _get(dvh, "Thyroid V50 (%)")
    if n_stage is not None and np.isfinite(float(n_stage)) and np.isfinite(volume) and np.isfinite(v50):
        zhou_score = float(float(n_stage) > 1) + float(volume <= 12.82) + float(v50 > 24.0)
        zhou_call = "high-risk" if zhou_score >= 2 else ("intermediate-risk" if zhou_score == 1 else "low-risk")
        add_fixed_score(
            "Zhou2020_Nstage_TV12p82_V50_ordinal",
            zhou_score,
            "I(N-stage>1) + I(Volume<=12.82cc) + I(V50>24%)",
            "Table S2 ordinal implementation from the published clinical-DVH risk factors.",
            zhou_call,
            "Zhou et al. 2020",
        )
    else:
        add(
            "Zhou2020_Nstage_TV12p82_V50_ordinal",
            None,
            "not available",
            "requires N-stage, thyroid volume, and V50%",
            "Table S2 ordinal implementation from the published clinical-DVH risk factors.",
            "",
            "Zhou et al. 2020",
        )

    v3060 = _get(dvh, "Thyroid V30-60 (%)")
    if np.isfinite(volume) and np.isfinite(v3060):
        if volume > 20.0:
            peng_call = "low-risk"
            peng_note = "Peng2020 stratum: thyroid volume >20 cc."
        elif v3060 > 80.0:
            peng_call = "high-risk"
            peng_note = "Peng2020 stratum: volume <=20 cc and V30-60 >80%."
        else:
            peng_call = "intermediate-risk"
            peng_note = "Peng2020 stratum: volume <=20 cc and V30-60 <=80%."
        add(
            "Peng2020_TV20_V30_60_80_ordinal",
            float({"low-risk": 0, "intermediate-risk": 1, "high-risk": 2}[peng_call]),
            "computed",
            "if Volume>20cc: score 0; else if V30-60<=80%: score 1; else score 2",
            peng_note,
            peng_call,
            "Peng et al. 2020",
        )
        add_fixed_score(
            "Peng2020_TV_le20_binary",
            float(volume <= 20.0),
            "High risk = thyroid volume <=20 cc",
            "Table S2 boundary sensitivity using thyroid volume alone.",
            "high-risk" if volume <= 20.0 else "low-risk",
            "Peng et al. 2020",
        )
        add_fixed_score(
            "Peng2020_TVle20_V30_60gt80_binary",
            float((volume <= 20.0) and (v3060 > 80.0)),
            "High risk = thyroid volume <=20 cc and V30-60 >80%",
            "Table S2 highest-risk boundary sensitivity.",
            "high-risk" if (volume <= 20.0 and v3060 > 80.0) else "low-risk",
            "Peng et al. 2020",
        )
    else:
        add(
            "Peng2020_TV20_V30_60_80_ordinal",
            None,
            "not available",
            "requires thyroid volume and V30-60%",
            "V30-60 is reconstructed as V30% - V60%.",
            "",
            "Peng et al. 2020",
        )
        add("Peng2020_TV_le20_binary", None, "not available", "requires thyroid volume", "", "", "Peng et al. 2020")
        add("Peng2020_TVle20_V30_60gt80_binary", None, "not available", "requires thyroid volume and V30-60%", "", "", "Peng et al. 2020")

    return pd.DataFrame(rows)
