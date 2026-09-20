from __future__ import annotations

"""Windographer 4.2.25 WBL compatibility implementation.

Reverse engineering against the supplied 79117 EL160 WBL export indicates:

1. strict concurrent 10-minute samples;
2. four calendar quarters x 16 sectors based on REFERENCE direction;
3. target and reference Weibull parameters are calculated with the Openwind
   fitting method (match observed mean wind speed and mean wind-power-density
   proxy / third moment);
4. transform reference to target by y = Scale * x ** Exponent, where
      Exponent = k_ref / k_target
      Scale    = c_target / c_ref ** Exponent
5. a QxS subdivision with fewer than 50 concurrent time steps falls back to
   the all-data/global WBL model.  Empty or mathematically invalid groups also
   use the global model.

This is an empirical compatibility implementation for the supplied
Windographer 4.2.25 run.  Windographer can use a user-selected preferred
Weibull fitting algorithm in Long Term Adjustments; this run numerically
matches Openwind, so compatibility mode below uses Openwind.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .weibull_fit import openwind_weibull

MCP_WBL_MIN_SEGMENT_SAMPLES = 50


def sector_index(wind_direction: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pd.Series(wind_direction), errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = np.floor(((values[valid] % 360.0) + 11.25) / 22.5).astype(int) % 16
    return result


def _openwind_fit(values):
    """Fit a 2-parameter Weibull with Windographer/Openwind moment matching.

    Openwind fitting can include exact zero/calm values directly because it is
    based on the observed first and third moments rather than log(speed).
    The return tuple intentionally preserves the legacy diagnostic field shape
    expected by the rest of the application.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if len(x) < 3:
        raise RuntimeError("WBL失败：有效样本少于3个，无法执行Openwind Weibull拟合。")
    positive = x[x > 0.0]
    if len(positive) < 2:
        raise RuntimeError("WBL失败：正风速样本少于2个，无法拟合Weibull。")
    fit = openwind_weibull(x)
    zero_count = int(np.sum(x == 0.0))
    min_positive = float(np.min(positive))
    k = float(fit.k)
    c = float(fit.c)
    return (k, c, int(len(positive)), int(len(x)), zero_count,
            min_positive, 0.0, k, c, False)


def _r2(observed, predicted) -> float:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 2:
        return float("nan")
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0.0:
        return float("nan")
    sse = float(np.sum((y - p) ** 2))
    return float(1.0 - sse / sst)


@dataclass
class _WBLPower:
    scale_a: float
    exponent_b: float
    params: dict

    def predict(self, reference_speed) -> np.ndarray:
        xx = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        yy = np.full(len(xx), np.nan, dtype=float)
        ok = np.isfinite(xx) & (xx >= 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            yy[ok] = self.scale_a * np.power(xx[ok], self.exponent_b)
        yy[~np.isfinite(yy)] = np.nan
        return np.maximum(yy, 0.0)


def _fit_wbl_group(train: pd.DataFrame) -> _WBLPower:
    work = train[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["ref_ws"] >= 0.0) & (work["target_ws"] >= 0.0)].copy()
    if len(work) < 2:
        raise RuntimeError("WBL失败：有效并发样本少于2个。")

    (ref_k, ref_c, ref_positive_n, ref_fit_n, ref_zero_count, ref_min_positive,
     ref_zero_surrogate, ref_k_raw, ref_c_raw, ref_k_capped) = _openwind_fit(work["ref_ws"].to_numpy(float))
    (target_k, target_c, target_positive_n, target_fit_n, target_zero_count, target_min_positive,
     target_zero_surrogate, target_k_raw, target_c_raw, target_k_capped) = _openwind_fit(work["target_ws"].to_numpy(float))

    exponent_b = float(ref_k / target_k)
    scale_a = float(target_c / (ref_c ** exponent_b))
    if not np.isfinite(exponent_b) or not np.isfinite(scale_a) or exponent_b <= 0.0 or scale_a <= 0.0:
        raise RuntimeError("WBL失败：幂律参数无效。")

    predicted = scale_a * np.power(work["ref_ws"].to_numpy(float), exponent_b)
    r2 = _r2(work["target_ws"].to_numpy(float), predicted)

    return _WBLPower(scale_a, exponent_b, {
        "algorithm": "Openwind Weibull (match mean speed + mean wind-power-density proxy)",
        "zero_handling_hypothesis": "Openwind moment fit: exact zeros remain valid and are included directly in observed moments",
        "target_k": float(target_k),
        "target_c_mps": float(target_c),
        "target_k_raw": float(target_k_raw),
        "target_c_raw_mps": float(target_c_raw),
        "target_k_capped": bool(target_k_capped),
        "reference_k": float(ref_k),
        "reference_c_mps": float(ref_c),
        "reference_k_raw": float(ref_k_raw),
        "reference_c_raw_mps": float(ref_c_raw),
        "reference_k_capped": bool(ref_k_capped),
        "scale_a": float(scale_a),
        "exponent_b": float(exponent_b),
        "r2": float(r2),
        "group_samples": int(len(work)),
        "target_zero_count": int(target_zero_count),
        "reference_zero_count": int(ref_zero_count),
        "target_min_positive_mps": float(target_min_positive),
        "reference_min_positive_mps": float(ref_min_positive),
        "target_zero_surrogate_mps": float(target_zero_surrogate),
        "reference_zero_surrogate_mps": float(ref_zero_surrogate),
        "target_positive_fit_n": int(target_positive_n),
        "reference_positive_fit_n": int(ref_positive_n),
        "target_weibull_fit_n": int(target_fit_n),
        "reference_weibull_fit_n": int(ref_fit_n),
    })


def _quarter_label(q: int) -> str:
    return {1: "Jan - Mar", 2: "Apr - Jun", 3: "Jul - Sep", 4: "Oct - Dec"}[int(q)]


def _sector_label(s: int) -> str:
    lo = (float(s) * 22.5 - 11.25) % 360.0
    hi = (float(s) * 22.5 + 11.25) % 360.0
    if s == 0:
        return "348.75 - 11.25"
    return f"{lo:.2f} - {hi:.2f}"


@dataclass
class WindographerWBLModel:
    global_model: _WBLPower
    local_models: Dict[Tuple[int, int], _WBLPower]
    group_counts: Dict[Tuple[int, int], int]

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        local_time = pd.to_datetime(work["target_local_time"], errors="coerce")
        quarter = local_time.dt.quarter.to_numpy()
        sector = sector_index(work.get("ref_dir", pd.Series(np.nan, index=work.index)))
        prediction = self.global_model.predict(work["ref_ws"])
        for (q, s), model in self.local_models.items():
            mask = (quarter == q) & (sector == s)
            if mask.any():
                prediction[mask] = model.predict(work.loc[mask, "ref_ws"])
        return prediction

    def diagnostics(self) -> pd.DataFrame:
        rows = []
        for q in range(1, 5):
            for s in range(16):
                n = int(self.group_counts.get((q, s), 0))
                local = self.local_models.get((q, s))
                fallback = local is None
                model = self.global_model if fallback else local
                p = model.params
                rows.append({
                    "Year division": _quarter_label(q),
                    "Sector": _sector_label(s),
                    "Time Steps": n,
                    "Target k": p.get("target_k"),
                    "Target c (m/s)": p.get("target_c_mps"),
                    "Target raw k": p.get("target_k_raw"),
                    "Target k capped at 10": p.get("target_k_capped"),
                    "Reference k": p.get("reference_k"),
                    "Reference c (m/s)": p.get("reference_c_mps"),
                    "Reference raw k": p.get("reference_k_raw"),
                    "Reference k capped at 10": p.get("reference_k_capped"),
                    "Best-fit Scale": p.get("scale_a"),
                    "Best-fit Exponent": p.get("exponent_b"),
                    "R2": p.get("r2"),
                    "Fallback": "GLOBAL" if fallback else "LOCAL",
                    "Target zero count": 0 if fallback and n == 0 else p.get("target_zero_count"),
                    "Reference zero count": 0 if fallback and n == 0 else p.get("reference_zero_count"),
                    "Target min positive (m/s)": p.get("target_min_positive_mps"),
                    "Reference min positive (m/s)": p.get("reference_min_positive_mps"),
                    "Target zero surrogate (m/s)": p.get("target_zero_surrogate_mps"),
                    "Reference zero surrogate (m/s)": p.get("reference_zero_surrogate_mps"),
                    "Target positive original n": p.get("target_positive_fit_n"),
                    "Reference positive original n": p.get("reference_positive_fit_n"),
                    "Target Weibull fit n": p.get("target_weibull_fit_n"),
                    "Reference Weibull fit n": p.get("reference_weibull_fit_n"),
                    "Algorithm": p.get("algorithm"),
                    "Zero handling hypothesis": p.get("zero_handling_hypothesis"),
                })
        gp = self.global_model.params
        rows.append({
            "Year division": "All",
            "Sector": "All",
            "Time Steps": int(gp.get("group_samples", 0)),
            "Target k": gp.get("target_k"),
            "Target c (m/s)": gp.get("target_c_mps"),
            "Target raw k": gp.get("target_k_raw"),
            "Target k capped at 10": gp.get("target_k_capped"),
            "Reference k": gp.get("reference_k"),
            "Reference c (m/s)": gp.get("reference_c_mps"),
            "Reference raw k": gp.get("reference_k_raw"),
            "Reference k capped at 10": gp.get("reference_k_capped"),
            "Best-fit Scale": gp.get("scale_a"),
            "Best-fit Exponent": gp.get("exponent_b"),
            "R2": gp.get("r2"),
            "Fallback": "GLOBAL_MODEL",
            "Target zero count": gp.get("target_zero_count"),
            "Reference zero count": gp.get("reference_zero_count"),
            "Target min positive (m/s)": gp.get("target_min_positive_mps"),
            "Reference min positive (m/s)": gp.get("reference_min_positive_mps"),
            "Target zero surrogate (m/s)": gp.get("target_zero_surrogate_mps"),
            "Reference zero surrogate (m/s)": gp.get("reference_zero_surrogate_mps"),
            "Target positive original n": gp.get("target_positive_fit_n"),
            "Reference positive original n": gp.get("reference_positive_fit_n"),
            "Target Weibull fit n": gp.get("target_weibull_fit_n"),
            "Reference Weibull fit n": gp.get("reference_weibull_fit_n"),
            "Algorithm": gp.get("algorithm"),
            "Zero handling hypothesis": gp.get("zero_handling_hypothesis"),
        })
        return pd.DataFrame(rows)


def fit_windographer_wbl(concurrent: pd.DataFrame, minimum_local_samples: int = MCP_WBL_MIN_SEGMENT_SAMPLES) -> WindographerWBLModel:
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"]
    )
    work = work[(work["target_ws"] >= 0.0) & (work["ref_ws"] >= 0.0)].copy()
    if len(work) < 2:
        raise RuntimeError("WBL Q4×S16失败：全局有效样本少于2个。")

    work["quarter"] = work["target_local_time"].dt.quarter.astype(int)
    work["sector"] = sector_index(work["ref_dir"]).astype(int)

    global_model = _fit_wbl_group(work)
    group_counts: Dict[Tuple[int, int], int] = {}
    for q in range(1, 5):
        for s in range(16):
            group_counts[(q, s)] = int(((work["quarter"] == q) & (work["sector"] == s)).sum())

    local_models: Dict[Tuple[int, int], _WBLPower] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"]):
        if len(group) >= int(minimum_local_samples):
            try:
                local_models[(int(quarter), int(sector))] = _fit_wbl_group(group)
            except RuntimeError:
                # If a populated group cannot produce a valid Openwind Weibull fit,
                # use the global model rather than inventing another estimator.
                pass

    return WindographerWBLModel(global_model, local_models, group_counts)
