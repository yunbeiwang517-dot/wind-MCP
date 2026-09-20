from __future__ import annotations

"""Windographer 4.2.25-compatible Vertical Slice support.

This module intentionally contains only the VS-specific mechanics.  The other
MCP methods retain their existing workflow and shared thresholds.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


VS_SLICE_COUNT = 10
VS_MIN_SLICE_POINTS = 5
VS_MIN_LOCAL_SAMPLES = 50
VS_MIN_DCR = 0.50


def sector_index(wind_direction: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pd.Series(wind_direction), errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = np.floor(((values[valid] % 360.0) + 11.25) / 22.5).astype(int) % 16
    return result


def upper_bound(max_reference_speed: float) -> float:
    """Reproduce Windographer's automatic VS reference-speed upper limit."""
    maximum = float(max_reference_speed)
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("invalid maximum reference wind speed")
    if maximum > 10.0 + 1e-12:
        return float(2.0 * np.ceil(maximum / 2.0 - 1e-12))
    ceiling = max(float(np.ceil(maximum - 1e-12)), 0.1)
    candidate = 0.9 * ceiling
    return float(candidate if candidate >= maximum - 1e-12 else ceiling)


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
    minimum_dcr: float = VS_MIN_DCR,
) -> pd.DataFrame:
    """Create the VS-only left/left hourly concurrent frame.

    DCR applies only to the temporary target hourly series.  Returned time is
    target local time so Q1..Q4 always follows the target mast calendar.
    """
    target_work = target[[target_time_column, target_speed_column]].copy()
    target_work["target_local_time"] = (
        pd.to_datetime(target_work[target_time_column], errors="coerce")
        + pd.Timedelta(hours=int(local_utc_offset_hours))
    )
    target_work["target_ws"] = pd.to_numeric(target_work[target_speed_column], errors="coerce")
    target_work = target_work.dropna(subset=["target_local_time"]).sort_values("target_local_time")
    if target_work.empty:
        raise ValueError("empty target timeline")

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
    # pandas Copy-on-Write can return a read-only view here; VS DCR masks the
    # temporary hourly copy, never the user's original target data.
    hourly_values = mean.to_numpy(dtype=float, copy=True)
    hourly_values[count.to_numpy(int) < np.ceil(expected * float(minimum_dcr)).astype(int)] = np.nan
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
class _SliceModel:
    slope: float
    intercept: float
    px: np.ndarray
    py: np.ndarray
    upper: float
    bin_counts: np.ndarray

    def predict(self, reference_speed: pd.Series) -> np.ndarray:
        speed = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        prediction = np.interp(speed, self.px, self.py, left=self.py[0], right=np.nan)
        high = ~np.isfinite(prediction) & np.isfinite(speed)
        if high.any():
            prediction[high] = self.py[-1] + self.slope * (speed[high] - self.px[-1])
        finite = np.isfinite(prediction)
        prediction[finite] = np.maximum(prediction[finite], 0.0)
        return prediction


def _fit_group(frame: pd.DataFrame) -> _SliceModel:
    work = frame[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["ref_ws"] >= 0) & (work["target_ws"] >= 0)].copy()
    if len(work) < 2:
        raise ValueError("VS subgroup needs at least two samples")

    x = work["ref_ws"].to_numpy(float)
    y = work["target_ws"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    upper = upper_bound(float(np.max(x)))
    edges = np.linspace(0.0, upper, VS_SLICE_COUNT + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    controls: list[float] = []
    counts: list[int] = []
    for index, midpoint in enumerate(midpoints):
        in_slice = (x >= edges[index]) & (x < edges[index + 1])
        count = int(in_slice.sum())
        counts.append(count)
        controls.append(float(np.mean(y[in_slice])) if count >= VS_MIN_SLICE_POINTS else float(slope * midpoint + intercept))
    return _SliceModel(
        slope=float(slope),
        intercept=float(intercept),
        px=np.r_[0.0, midpoints],
        py=np.r_[0.0, np.asarray(controls, dtype=float)],
        upper=float(upper),
        bin_counts=np.asarray(counts, dtype=int),
    )


@dataclass
class WindographerVSModel:
    global_model: _SliceModel
    local_models: Dict[Tuple[int, int], _SliceModel]

    def predict_frame(
        self,
        frame: pd.DataFrame,
        *,
        local_utc_offset_hours: int,
    ) -> np.ndarray:
        work = frame.copy()
        if "target_local_time" not in work.columns:
            if "Timestamp_UTC" not in work.columns:
                raise ValueError("VS prediction needs target_local_time or Timestamp_UTC")
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


def fit_windographer_vs(
    concurrent: pd.DataFrame,
    *,
    minimum_local_samples: int = VS_MIN_LOCAL_SAMPLES,
) -> WindographerVSModel:
    """Fit global VS and the eligible quarter×sector VS models only."""
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"])
    work = work[(work["target_ws"] >= 0) & (work["ref_ws"] >= 0)].copy()
    if len(work) < 2:
        raise ValueError("not enough concurrent data for global VS")
    work["quarter"] = work["target_local_time"].dt.quarter
    work["sector"] = sector_index(work["ref_dir"]).astype(int)
    global_model = _fit_group(work)
    local_models: Dict[Tuple[int, int], _SliceModel] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"]):
        if len(group) >= int(minimum_local_samples):
            local_models[(int(quarter), int(sector))] = _fit_group(group)
    return WindographerVSModel(global_model=global_model, local_models=local_models)
