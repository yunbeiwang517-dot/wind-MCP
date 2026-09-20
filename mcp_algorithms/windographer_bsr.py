from __future__ import annotations

"""Windographer 4.2.25-compatible Bulk Speed Ratio (BSR) support only.

This module is deliberately isolated from the shared MCP Q4×S16 wrapper.
BSR uses a ratio of means and has its own local-model eligibility and direct
global fallback.  No target channel is named here: the caller supplies the
currently selected GUI target-speed column.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BSR_FIT_MIN_SPEED = 2.5
BSR_MIN_LOCAL_SAMPLES = 50  # default 10min; engine passes 24 for verified 1h WG mode  # default 10min; engine passes 24 for verified 1h WG mode
BSR_MIN_DCR = 0.50


def sector_index(wind_direction: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pd.Series(wind_direction), errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = np.floor(((values[valid] % 360.0) + 11.25) / 22.5).astype(int) % 16
    return result


def _inferred_step(timestamps: pd.Series) -> pd.Timedelta:
    values = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(values) < 2:
        return pd.Timedelta(minutes=10)
    minutes = values.diff().dropna().dt.total_seconds() / 60.0
    minutes = minutes[(minutes > 0) & np.isfinite(minutes)]
    return pd.Timedelta(minutes=float(minutes.median())) if len(minutes) else pd.Timedelta(minutes=10)


def build_concurrent_hours(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    target_time_column: str,
    target_speed_column: str,
    reference_time_column: str,
    reference_speed_column: str,
    reference_direction_column: str,
    local_utc_offset_hours: int,
    reference_shift_hours: int,
    minimum_dcr: float = BSR_MIN_DCR,
) -> pd.DataFrame:
    """Build BSR's temporary hourly concurrent frame.

    Input target timestamps are UTC in the app runtime.  They are converted to
    the mast's local calendar before left/left aggregation, so quarters follow
    target local time.  DCR affects this temporary hourly copy only.
    """
    target_work = target[[target_time_column, target_speed_column]].copy()
    target_work["target_local_time"] = (
        pd.to_datetime(target_work[target_time_column], errors="coerce")
        + pd.Timedelta(hours=int(local_utc_offset_hours))
    )
    target_work["target_ws"] = pd.to_numeric(target_work[target_speed_column], errors="coerce")
    target_work = target_work.dropna(subset=["target_local_time"]).sort_values("target_local_time")
    if target_work.empty:
        raise ValueError("empty BSR target timeline")

    step = _inferred_step(target_work["target_local_time"])
    coverage_start = target_work["target_local_time"].min()
    coverage_end = target_work["target_local_time"].max()
    series = pd.Series(target_work["target_ws"].to_numpy(float), index=target_work["target_local_time"])
    resampler = series.resample("1h", closed="left", label="left")
    mean = resampler.mean()
    count = resampler.count()
    expected_counts: list[int] = []
    for hour in mean.index:
        candidates = pd.date_range(hour, hour + pd.Timedelta(hours=1) - step, freq=step)
        expected_counts.append(max(1, int(((candidates >= coverage_start) & (candidates <= coverage_end)).sum())))
    expected = np.asarray(expected_counts, dtype=int)
    min_count = np.ceil(expected * float(minimum_dcr)).astype(int)
    hourly_values = mean.to_numpy(dtype=float, copy=True)
    hourly_values[count.to_numpy(dtype=int) < min_count] = np.nan
    hourly = pd.DataFrame({"target_local_time": mean.index, "target_ws": hourly_values})

    reference_work = reference[[reference_time_column, reference_speed_column, reference_direction_column]].copy()
    reference_work["target_local_time"] = (
        pd.to_datetime(reference_work[reference_time_column], errors="coerce")
        + pd.Timedelta(hours=int(reference_shift_hours) + int(local_utc_offset_hours))
    )
    reference_work["ref_ws"] = pd.to_numeric(reference_work[reference_speed_column], errors="coerce")
    reference_work["ref_dir"] = pd.to_numeric(reference_work[reference_direction_column], errors="coerce")
    reference_work = reference_work[["target_local_time", "ref_ws", "ref_dir"]]
    reference_work = reference_work.dropna(subset=["target_local_time"]).sort_values("target_local_time")
    reference_work = reference_work.drop_duplicates("target_local_time", keep="last")

    concurrent = hourly.merge(reference_work, on="target_local_time", how="inner")
    concurrent["Timestamp_UTC"] = concurrent["target_local_time"] - pd.Timedelta(hours=int(local_utc_offset_hours))
    concurrent["month"] = concurrent["target_local_time"].dt.month
    concurrent["main_sector"] = True
    return concurrent.sort_values("target_local_time").reset_index(drop=True)


@dataclass
class _RatioModel:
    ratio: float

    def predict(self, reference_speed: pd.Series) -> np.ndarray:
        speed = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        prediction = self.ratio * speed
        finite = np.isfinite(prediction)
        prediction[finite] = np.maximum(prediction[finite], 0.0)
        return prediction


def _fit_ratio(frame: pd.DataFrame) -> _RatioModel:
    work = frame[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("not enough BSR samples")
    mean_ref = float(work["ref_ws"].mean())
    mean_target = float(work["target_ws"].mean())
    if not np.isfinite(mean_ref) or mean_ref <= 0 or not np.isfinite(mean_target):
        raise ValueError("invalid BSR means")
    return _RatioModel(mean_target / mean_ref)


@dataclass
class WindographerBSRModel:
    global_model: _RatioModel
    local_models: Dict[Tuple[int, int], _RatioModel]

    def predict_frame(self, frame: pd.DataFrame, *, local_utc_offset_hours: int) -> np.ndarray:
        work = frame.copy()
        if "target_local_time" not in work.columns:
            if "Timestamp_UTC" not in work.columns:
                raise ValueError("BSR prediction needs target_local_time or Timestamp_UTC")
            work["target_local_time"] = (
                pd.to_datetime(work["Timestamp_UTC"], errors="coerce")
                + pd.Timedelta(hours=int(local_utc_offset_hours))
            )
        local_time = pd.to_datetime(work["target_local_time"], errors="coerce")
        quarter = local_time.dt.quarter.to_numpy()
        sector = sector_index(work.get("ref_dir", pd.Series(np.nan, index=work.index)))
        prediction = self.global_model.predict(work["ref_ws"])
        for (quarter_no, sector_no), model in self.local_models.items():
            mask = (quarter == quarter_no) & (sector == sector_no)
            if mask.any():
                prediction[mask] = model.predict(work.loc[mask, "ref_ws"])
        return prediction


def fit_windographer_bsr(
    concurrent: pd.DataFrame,
    *,
    fit_min_speed: float = BSR_FIT_MIN_SPEED,
    minimum_local_samples: int = BSR_MIN_LOCAL_SAMPLES,
) -> WindographerBSRModel:
    """Fit global BSR plus eligible Q4×S16 local BSR models.

    BSR remains ``mean(target) / mean(reference)`` with a zero intercept.
    The 2.5 m/s threshold applies only while fitting; finite low reference
    speeds are still predicted.  A sparse local group falls back directly to
    global BSR, without quarter-only or sector-only models.
    """
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"])
    work = work[(work["target_ws"] >= float(fit_min_speed)) & (work["ref_ws"] >= float(fit_min_speed))].copy()
    if work.empty:
        raise ValueError("not enough concurrent BSR samples after low-speed filter")
    work["quarter"] = work["target_local_time"].dt.quarter
    work["sector"] = sector_index(work["ref_dir"]).astype(int)
    global_model = _fit_ratio(work)
    local_models: Dict[Tuple[int, int], _RatioModel] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"]):
        if len(group) >= int(minimum_local_samples):
            local_models[(int(quarter), int(sector))] = _fit_ratio(group)
    return WindographerBSRModel(global_model=global_model, local_models=local_models)
