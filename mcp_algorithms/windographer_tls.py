from __future__ import annotations

"""Windographer 4.2.25-compatible Total Least Squares (TLS) MCP support.

Core TLS formula was recovered against Windographer 4.0.27 exports; the
local-subdivision minimum sample rule was updated from new Windographer 4.2.25
outputs.
The production fit uses all valid concurrent points (fit_min_speed=0).  The
configurable low-wind threshold is intended only for the app's repeated 50/50
sampling evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TLS_FIT_MIN_SPEED = 2.5
TLS_MIN_LOCAL_SAMPLES = 50  # default 10min; engine passes 24 for verified 1h WG mode  # default 10min; engine passes 24 for verified 1h WG mode


def sector_index(wind_direction: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pd.Series(wind_direction), errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = np.floor(((values[valid] % 360.0) + 11.25) / 22.5).astype(int) % 16
    return result


@dataclass
class _TLSLine:
    slope: float
    intercept: float
    sample_count: int

    def predict(self, reference_speed: pd.Series) -> np.ndarray:
        x = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        y = self.slope * x + self.intercept
        finite = np.isfinite(y)
        # Windographer speed output is non-negative even when the fitted line
        # extrapolates below zero.
        y[finite] = np.maximum(y[finite], 0.0)
        return y


def _fit_tls_line(frame: pd.DataFrame) -> _TLSLine:
    work = frame[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 2:
        raise ValueError("not enough points for TLS")

    x = work["ref_ws"].to_numpy(float)
    y = work["target_ws"].to_numpy(float)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    dy = y - ym
    sxx = float(np.sum(dx * dx))
    syy = float(np.sum(dy * dy))
    sxy = float(np.sum(dx * dy))

    # Two-parameter orthogonal/total least squares line used by Windographer:
    # y = m*x + b, minimizing squared perpendicular distances.
    if not (np.isfinite(sxx) and np.isfinite(syy) and np.isfinite(sxy)):
        raise ValueError("invalid TLS covariance terms")
    if sxx <= 1e-15 and syy <= 1e-15:
        raise ValueError("degenerate TLS sample cloud")
    if abs(sxy) <= 1e-15:
        # A strictly vertical principal axis cannot be represented as y=f(x).
        # Treat this local cell as singular so the caller can use the global
        # fallback rather than inventing a finite slope.
        raise ValueError("TLS principal axis is vertical/undefined")

    disc = float(np.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy))
    slope = float((syy - sxx + disc) / (2.0 * sxy))
    intercept = float(ym - slope * xm)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        raise ValueError("invalid TLS line")
    return _TLSLine(slope=slope, intercept=intercept, sample_count=int(len(work)))


@dataclass
class WindographerTLSModel:
    global_model: _TLSLine
    local_models: Dict[Tuple[int, int], _TLSLine]

    def predict_frame(self, frame: pd.DataFrame, *, local_utc_offset_hours: int) -> np.ndarray:
        work = frame.copy()
        if "target_local_time" not in work.columns:
            if "Timestamp_UTC" not in work.columns:
                raise ValueError("TLS prediction needs target_local_time or Timestamp_UTC")
            work["target_local_time"] = (
                pd.to_datetime(work["Timestamp_UTC"], errors="coerce")
                + pd.Timedelta(hours=int(local_utc_offset_hours))
            )
        local_time = pd.to_datetime(work["target_local_time"], errors="coerce")
        quarter = local_time.dt.quarter.to_numpy()
        sector = sector_index(work.get("ref_dir", pd.Series(np.nan, index=work.index)))
        pred = self.global_model.predict(work["ref_ws"])
        for (quarter_no, sector_no), model in self.local_models.items():
            mask = (quarter == quarter_no) & (sector == sector_no)
            if mask.any():
                pred[mask] = model.predict(work.loc[mask, "ref_ws"])
        return pred

    def diagnostics(self) -> pd.DataFrame:
        rows = [{
            "范围": "GLOBAL",
            "季度": np.nan,
            "扇区": np.nan,
            "样本数": self.global_model.sample_count,
            "Slope": self.global_model.slope,
            "Intercept": self.global_model.intercept,
        }]
        for (q, s), model in sorted(self.local_models.items()):
            rows.append({
                "范围": "LOCAL",
                "季度": q,
                "扇区": s,
                "样本数": model.sample_count,
                "Slope": model.slope,
                "Intercept": model.intercept,
            })
        return pd.DataFrame(rows)


def fit_windographer_tls(
    concurrent: pd.DataFrame,
    *,
    fit_min_speed: float = TLS_FIT_MIN_SPEED,
    minimum_local_samples: int = TLS_MIN_LOCAL_SAMPLES,
) -> WindographerTLSModel:
    """Fit Windographer-style two-parameter TLS globally and by Q4×S16.

    In this app the caller uses ``fit_min_speed`` only for repeated 50/50
    evaluation.  The production call explicitly passes ``0.0``, so all valid
    concurrent low-wind points participate in the final fit.

    Windographer 4.2.25 evidence on the supplied 79117 case shows that a
    Q4×S16 local subdivision is used only when it contains at least 50 valid
    time steps. Cells with fewer than 50 points fall back to the GLOBAL All/All
    TLS line. No empirical R2/slope/output-cap guard is applied.

    Note: the app's optional 1h TLS mode is an engineering comparison mode,
    not a native Windographer mode recovered from the supplied export. It uses
    the same numeric minimum (50 hourly samples) until separate 1h evidence is
    available.
    """
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"]
    )
    threshold = float(fit_min_speed)
    work = work[(work["target_ws"] >= threshold) & (work["ref_ws"] >= threshold)].copy()
    if len(work) < 2:
        raise ValueError("not enough concurrent TLS samples after low-speed filter")

    work["quarter"] = work["target_local_time"].dt.quarter
    work["sector"] = sector_index(work["ref_dir"]).astype(int)
    global_model = _fit_tls_line(work)
    local_models: Dict[Tuple[int, int], _TLSLine] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"]):
        if len(group) < int(minimum_local_samples):
            continue
        try:
            local_models[(int(quarter), int(sector))] = _fit_tls_line(group)
        except ValueError:
            # Singular/undefined local cells fall back directly to the GLOBAL All/All TLS line.
            continue
    return WindographerTLSModel(global_model=global_model, local_models=local_models)
