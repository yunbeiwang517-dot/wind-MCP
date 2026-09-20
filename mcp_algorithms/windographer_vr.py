from __future__ import annotations

"""Windographer 4.2.25-compatible Variance Ratio (VR) support only."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


VR_FIT_MIN_SPEED = 2.5
VR_MIN_LOCAL_SAMPLES = 50  # default 10min; engine passes 24 for verified 1h WG mode  # default 10min; engine passes 24 for verified 1h WG mode
VR_MIN_DCR = 0.50


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
    minimum_dcr: float = VR_MIN_DCR,
) -> pd.DataFrame:
    """Build left/left hourly target-local VR concurrent samples."""
    target_work = target[[target_time_column, target_speed_column]].copy()
    target_work["target_local_time"] = (
        pd.to_datetime(target_work[target_time_column], errors="coerce")
        + pd.Timedelta(hours=int(local_utc_offset_hours))
    )
    target_work["target_ws"] = pd.to_numeric(target_work[target_speed_column], errors="coerce")
    target_work = target_work.dropna(subset=["target_local_time"]).sort_values("target_local_time")
    if target_work.empty:
        raise ValueError("empty VR target timeline")

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
    hourly_values = mean.to_numpy(dtype=float, copy=True)
    hourly_values[count.to_numpy(dtype=int) < np.ceil(expected * float(minimum_dcr)).astype(int)] = np.nan
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
class _VRModel:
    slope: float
    intercept: float

    def predict(self, reference_speed: pd.Series) -> np.ndarray:
        speed = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        prediction = self.slope * speed + self.intercept
        finite = np.isfinite(prediction)
        prediction[finite] = np.maximum(prediction[finite], 0.0)
        return prediction


def _fit_group(frame: pd.DataFrame) -> _VRModel:
    work = frame[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 2:
        raise ValueError("not enough VR samples")
    ref = work["ref_ws"].to_numpy(float)
    target = work["target_ws"].to_numpy(float)
    ref_std = float(np.std(ref, ddof=0))
    target_std = float(np.std(target, ddof=0))
    if not np.isfinite(ref_std) or ref_std <= 0 or not np.isfinite(target_std):
        raise ValueError("invalid VR standard deviation")
    slope = target_std / ref_std
    intercept = float(np.mean(target) - slope * np.mean(ref))
    return _VRModel(float(slope), intercept)


@dataclass
class WindographerVRModel:
    global_model: _VRModel
    local_models: Dict[Tuple[int, int], _VRModel]

    def predict_frame(self, frame: pd.DataFrame, *, local_utc_offset_hours: int) -> np.ndarray:
        work = frame.copy()
        if "target_local_time" not in work.columns:
            if "Timestamp_UTC" not in work.columns:
                raise ValueError("VR prediction needs target_local_time or Timestamp_UTC")
            work["target_local_time"] = (
                pd.to_datetime(work["Timestamp_UTC"], errors="coerce")
                + pd.Timedelta(hours=int(local_utc_offset_hours))
            )
        quarter = pd.to_datetime(work["target_local_time"], errors="coerce").dt.quarter.to_numpy()
        sector = sector_index(work.get("ref_dir", pd.Series(np.nan, index=work.index)))
        prediction = self.global_model.predict(work["ref_ws"])
        for (quarter_no, sector_no), model in self.local_models.items():
            mask = (quarter == quarter_no) & (sector == sector_no)
            if mask.any():
                prediction[mask] = model.predict(work.loc[mask, "ref_ws"])
        return prediction


def fit_windographer_vr(
    concurrent: pd.DataFrame,
    *,
    fit_min_speed: float = VR_FIT_MIN_SPEED,
    minimum_local_samples: int = VR_MIN_LOCAL_SAMPLES,
) -> WindographerVRModel:
    """Fit global VR and eligible target-local Q4×S16 VR models only."""
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"])
    work = work[(work["target_ws"] >= float(fit_min_speed)) & (work["ref_ws"] >= float(fit_min_speed))].copy()
    if len(work) < 2:
        raise ValueError("not enough concurrent VR samples after low-speed filter")
    work["quarter"] = work["target_local_time"].dt.quarter
    work["sector"] = sector_index(work["ref_dir"]).astype(int)
    global_model = _fit_group(work)
    local_models: Dict[Tuple[int, int], _VRModel] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"]):
        if len(group) >= int(minimum_local_samples):
            local_models[(int(quarter), int(sector))] = _fit_group(group)
    return WindographerVRModel(global_model=global_model, local_models=local_models)
