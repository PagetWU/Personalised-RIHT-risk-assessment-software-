from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TIME_LABELS = ["1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "all"]
TIME_YEARS = {f"{i}y": float(i) for i in range(1, 10)}
TIME_YEARS["all"] = None


@dataclass
class PredictionResult:
    linear_predictor: float
    partial_hazard: float
    phenotype: str
    scaled_features: dict[str, float]
    risk_curve: pd.DataFrame


class CoxRIHTModel:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)
        self.spec = json.loads((self.asset_dir / "model_spec.json").read_text(encoding="utf-8"))
        self.feature_order = list(self.spec["feature_order"])
        self.coef = pd.read_csv(self.asset_dir / "full_qeh_model_coefficients.csv").set_index("feature")
        self.scaler = pd.read_csv(self.asset_dir / "full_qeh_scaler_parameters.csv").set_index("feature")
        self.baseline = pd.read_csv(self.asset_dir / "qeh_breslow_baseline_hazard.csv")
        means_path = self.asset_dir / "qeh_scaled_feature_means.csv"
        if means_path.exists():
            self.qeh_scaled_means = pd.read_csv(means_path).set_index("feature")["qeh_scaled_mean"]
        else:
            self.qeh_scaled_means = pd.Series(0.0, index=self.feature_order)

    def scale_features(self, raw_features: dict[str, float]) -> dict[str, float]:
        scaled = {}
        for feature in self.feature_order:
            value = raw_features.get(feature, np.nan)
            if value is None or not np.isfinite(float(value)):
                value = float(self.scaler.loc[feature, "impute_mean_from_QEH"])
            scaled[feature] = float(value) * float(self.scaler.loc[feature, "scale"]) + float(self.scaler.loc[feature, "min"])
        return scaled

    def linear_predictor(self, raw_features: dict[str, float]) -> tuple[float, dict[str, float]]:
        scaled = self.scale_features(raw_features)
        lp = 0.0
        for feature in self.feature_order:
            centered = scaled[feature] - float(self.qeh_scaled_means.loc[feature])
            lp += float(self.coef.loc[feature, "coef"]) * centered
        return float(lp), scaled

    def baseline_cumhaz_at(self, years: float | None) -> float:
        if years is None:
            return float(self.baseline["baseline_cumhaz"].iloc[-1])
        sub = self.baseline[self.baseline["time_years"] <= years]
        return 0.0 if sub.empty else float(sub["baseline_cumhaz"].iloc[-1])

    def risk_at(self, lp: float, years: float | None) -> float:
        h0 = self.baseline_cumhaz_at(years)
        return float(1.0 - np.exp(-h0 * np.exp(lp)))

    def risk_curve(self, lp: float) -> pd.DataFrame:
        rows = []
        for label in TIME_LABELS:
            years = TIME_YEARS[label]
            rows.append({"horizon": label, "years": years if years is not None else "all", "risk": self.risk_at(lp, years)})
        return pd.DataFrame(rows)

    @staticmethod
    def phenotype_from_curve(curve: pd.DataFrame) -> str:
        risk = dict(zip(curve["horizon"], curve["risk"]))
        r3 = float(risk["3y"])
        r7 = float(risk["7y"])
        if r3 < 0.13 and r7 < 0.50:
            return "persistent_low"
        if r3 >= 0.27 and r7 >= 0.65:
            return "early_accelerating"
        if r3 < 0.27 and r7 >= 0.65:
            return "late_accumulating"
        return "intermediate"

    def predict(self, raw_features: dict[str, float]) -> PredictionResult:
        lp, scaled = self.linear_predictor(raw_features)
        curve = self.risk_curve(lp)
        return PredictionResult(
            linear_predictor=lp,
            partial_hazard=float(np.exp(lp)),
            phenotype=self.phenotype_from_curve(curve),
            scaled_features=scaled,
            risk_curve=curve,
        )
