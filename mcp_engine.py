from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from calendar import monthrange
from typing import Callable, Optional
import json
import math
import re

import numpy as np
import pandas as pd

from mcp_algorithms.windographer_bsr import build_concurrent_hours, fit_windographer_bsr
from mcp_algorithms.windographer_lls import fit_windographer_lls
from mcp_algorithms.windographer_tls import fit_windographer_tls
from mcp_algorithms.windographer_vr import fit_windographer_vr
from mcp_algorithms.windographer_vs import fit_windographer_vs
from mcp_algorithms.windographer_speedsort_v2 import fit_deterministic_speedsort, build_branch_consensus_speedsort
from mcp_algorithms.windographer_wbl import fit_windographer_wbl
from mcp_algorithms.adaptive_wbl import fit_adaptive_wbl
from mcp_algorithms.windographer_mts import MTSOptions, fit_mts

METHODS = ["BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL", "MTS"]
ENGINE_VERSION = "1.0.56"


def _height_tag_from_column(column_name: str, fallback: str = "Target") -> str:
    """Extract a Windographer-friendly height tag such as ``150m`` from a column name.

    Examples:
      Ch1_Anem_150.00m_ESE_Avg_m/s -> 150m
      ERA5_100m_WS                 -> 100m

    If no plausible height is present, return *fallback* instead of inventing a height.
    """
    text = str(column_name or "")
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[mM](?=[^A-Za-z]|$)", text)
    vals = []
    for token in matches:
        try:
            value = float(token)
        except Exception:
            continue
        if 1.0 <= value <= 500.0:
            vals.append(value)
    if not vals:
        return fallback
    value = vals[0]
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}m"
    compact = (f"{value:.2f}").rstrip("0").rstrip(".")
    return f"{compact}m"


def _export_column_names(measured_speed_col: str, era_speed_col: str) -> dict[str, str]:
    target_h = _height_tag_from_column(measured_speed_col, "Target")
    era_h = _height_tag_from_column(era_speed_col, "100m")
    return {
        "target_h": target_h,
        "era_h": era_h,
        "measured": f"Measured_WS_{target_h}",
        "era_ws": f"ERA5_WS_{era_h}",
        "era_wd": f"ERA5_WD_{era_h}",
    }


@dataclass
class EvalYear:
    start: pd.Timestamp
    end_inclusive: pd.Timestamp
    end_exclusive: pd.Timestamp
    coverage: float
    valid: int
    expected: int
    candidates: pd.DataFrame
    monthly: pd.DataFrame


@dataclass
class RunConfig:
    measured_path: str
    era_path: str
    measured_time_col: str
    measured_speed_col: str
    measured_direction_col: str
    era_time_col: str
    era_speed_col: str
    era_direction_col: str
    output_dir: str
    fit_min_speed: float = 2.5
    # Global MCP fitting resolution. V1.0.56: all eight methods use the same
    # dependency-correct concurrent pool (Target WS + Ref WS + Ref WD). In 1h
    # mode Target WS is independently aggregated with DCR>=50% before hourly
    # cross-dataset concurrency; Target WD never deletes a speed training point.
    fit_resolution: str = "10min"
    # MCP training-period scope for all eight methods.
    # "all_valid" = all valid concurrent measured/reference overlap;
    # "evaluation_year" = only valid concurrent samples inside the selected evaluation year.
    training_scope: str = "all_valid"
    shift_min: int = -12
    shift_max: int = 12
    export_10min: bool = True
    # Optional Windographer-style concurrent-period limitation. Blank/None means
    # use all valid overlap. Date-only end values are treated as inclusive days.
    concurrent_start: str | None = None
    concurrent_end: str | None = None
    # MTS comparison controls. Keep these explicit so cross-tower tests can
    # reproduce the exact parameter set used in Windographer.
    ss_random_seed: int = 20260907
    wbl_mode: str = "windographer_compat"
    mts_moving_average_hours: int = 3
    mts_random_seed: int = 20260831
    mts_edge_tolerance_states: float = 1.5
    mts_edge_max_attempts: int = 500
    mts_edge_mode: str = "right_rejection"
    # Deterministic-method validation (BSR/LLS/TLS/VR/VS/WBL only).
    # When enabled, perform repeated month-stratified 50% train / 50% validation
    # splits for diagnostics. Final exported models are ALWAYS refit once on the
    # full concurrent dataset after validation, matching the old iterative-MCP flow.
    deterministic_sampling_enabled: bool = True
    deterministic_sampling_runs: int = 50
    deterministic_sampling_seed: int = 42
    # 50/50 unit inside each calendar month: whole-day blocks or whole-hour blocks.
    deterministic_sampling_mode: str = "day"
    # SS/MTS use multiple stochastic realizations on the SAME full concurrent dataset.
    # Their seed-stability test is separate from deterministic-method 50/50 sampling.
    stochastic_realizations: int = 9


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read common CSV/TXT tables, including native Windographer text exports.

    Windographer exports prepend metadata lines (Created/Latitude/flags/...)
    before the real tab-delimited header beginning with ``Date/Time``.  Pandas'
    delimiter sniffer can misinterpret those metadata lines, so detect and skip
    them first.  This is an input-reader fix only; it does not alter any MCP
    method or numerical data after parsing.
    """
    errors = []
    suffix = path.suffix.lower()
    preferred_sep = "," if suffix == ".csv" else ("\t" if suffix == ".tsv" else None)
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")

    if suffix in {".txt", ".dat", ".tsv"}:
        for encoding in encodings:
            try:
                with path.open("r", encoding=encoding, errors="strict") as fh:
                    header_row = None
                    for i in range(120):
                        line = fh.readline()
                        if not line:
                            break
                        stripped = line.lstrip("\ufeff").strip("\r\n")
                        if stripped.startswith("Date/Time\t") or stripped.startswith("Timestamp\t"):
                            header_row = i
                            break
                if header_row is not None:
                    return pd.read_csv(
                        path, sep="\t", skiprows=header_row, engine="c",
                        encoding=encoding, low_memory=False,
                    )
            except Exception as exc:
                errors.append(f"{encoding}/windographer: {exc}")

    for encoding in encodings:
        if preferred_sep is not None:
            try:
                return pd.read_csv(path, sep=preferred_sep, engine="c", encoding=encoding, low_memory=False)
            except Exception as exc:
                errors.append(f"{encoding}/fast: {exc}")
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}/flex: {exc}")
    raise RuntimeError("无法读取文本数据：" + " | ".join(errors[-4:]))


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    if suffix == ".xls":
        return pd.read_excel(path)
    if suffix in {".csv", ".txt", ".dat", ".tsv"}:
        return _read_csv_flexible(path)
    raise ValueError(f"暂不支持文件类型：{suffix}。请使用 CSV/TXT/XLS/XLSX。")


def _parse_time_values(values: pd.Series) -> pd.Series:
    """Parse a candidate time column without mistaking ordinary numbers for nanoseconds.

    pandas.to_datetime([5.2, 6.1]) interprets the values as nanoseconds after
    1970-01-01.  That is disastrous for wind-speed columns accidentally chosen
    as time.  Numeric timestamps are accepted only when they match a known
    representation (Unix s/ms/us/ns or Excel serial days).
    """
    s = pd.Series(values, copy=False)
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        finite = x[np.isfinite(x)]
        if finite.empty:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        med = float(np.nanmedian(np.abs(finite.to_numpy(float))))
        # Excel serial dates, roughly 1954-2119.
        if 20000 <= med <= 80000:
            ts = pd.to_datetime(x, unit="D", origin="1899-12-30", errors="coerce")
        # Unix seconds / milliseconds / microseconds / nanoseconds.
        elif 1e8 <= med < 1e11:
            ts = pd.to_datetime(x, unit="s", origin="unix", errors="coerce")
        elif 1e11 <= med < 1e14:
            ts = pd.to_datetime(x, unit="ms", origin="unix", errors="coerce")
        elif 1e14 <= med < 1e17:
            ts = pd.to_datetime(x, unit="us", origin="unix", errors="coerce")
        elif 1e17 <= med < 1e20:
            ts = pd.to_datetime(x, unit="ns", origin="unix", errors="coerce")
        else:
            # Ordinary physical values (e.g. wind speed 5.7 m/s) are NOT dates.
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    else:
        ts = pd.to_datetime(s.astype("string").str.strip(), errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    return ts


def _time_column_quality(values: pd.Series) -> tuple[float, int, pd.Timedelta]:
    ts = _parse_time_values(values)
    valid = ts.dropna()
    if valid.empty:
        return 0.0, 0, pd.Timedelta(0)
    ratio = float(valid.size / max(len(ts), 1))
    unique = int(valid.nunique())
    span = pd.Timestamp(valid.max()) - pd.Timestamp(valid.min()) if unique >= 2 else pd.Timedelta(0)
    return ratio, unique, span


def _is_plausible_time_column(values: pd.Series, min_ratio: float = 0.7) -> bool:
    ratio, unique, span = _time_column_quality(values)
    # MCP needs a real time series.  Requiring >= 24 unique timestamps and a
    # positive span blocks constant/physical numeric columns while remaining
    # permissive for short files used during inspection.
    return ratio >= min_ratio and unique >= 24 and span > pd.Timedelta(0)


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    priority = ["timestamp", "time", "date", "datetime", "时间", "日期", "时刻"]
    cols = list(df.columns)
    for key in priority:
        for col in cols:
            if key in str(col).lower() and _is_plausible_time_column(df[col]):
                return str(col)
    best, best_ratio, best_unique = None, 0.0, 0
    for col in cols[:50]:
        ratio, unique, span = _time_column_quality(df[col])
        if ratio >= 0.7 and unique >= 24 and span > pd.Timedelta(0):
            if ratio > best_ratio or (ratio == best_ratio and unique > best_unique):
                best, best_ratio, best_unique = str(col), float(ratio), int(unique)
    return best


def resolve_time_column(df: pd.DataFrame, requested: str, role: str = "数据") -> tuple[str, bool]:
    """Return a safe time column; automatically recover from a UI mis-selection."""
    if requested in df.columns and _is_plausible_time_column(df[requested]):
        return requested, False
    detected = detect_time_column(df)
    if detected is None:
        req = requested if requested else "<空>"
        raise ValueError(f"{role}时间列‘{req}’不能解析为有效时间序列，且未能自动识别其它时间列。")
    return detected, detected != requested

def numeric_candidates(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() >= 0.5:
            result.append(str(col))
    return result


def suggest_speed_column(df: pd.DataFrame) -> Optional[str]:
    numeric = numeric_candidates(df)
    keys = ("speed", "ws", "wind speed", "风速", "平均风速")
    for key in keys:
        for col in numeric:
            if key in col.lower():
                return col
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce")
        q = s.quantile([0.01, 0.99])
        if len(q) == 2 and q.iloc[0] >= -1 and q.iloc[1] <= 60:
            return col
    return numeric[0] if numeric else None


def suggest_direction_column(df: pd.DataFrame) -> Optional[str]:
    numeric = numeric_candidates(df)
    keys = ("direction", "wd", "wind dir", "风向")
    for key in keys:
        for col in numeric:
            if key in col.lower():
                return col
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) and s.between(0, 360).mean() > 0.95 and s.max() > 90:
            return col
    return None


def _column_height_m(column_name: str) -> float | None:
    text = str(column_name or "")
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[mM](?=[^A-Za-z]|$)", text)
    for token in matches:
        try:
            value = float(token)
        except Exception:
            continue
        if 1.0 <= value <= 500.0:
            return value
    return None


def suggest_target_direction_column(df: pd.DataFrame, speed_col: str | None = None) -> Optional[str]:
    """Suggest the mast direction channel paired with the selected target speed.

    Windographer's Compare Sites/MCP concurrent count is built from simultaneous
    speed *and direction* availability.  Prefer an average vane/direction channel
    at the same height as the selected speed; otherwise choose the nearest-height
    valid direction channel.
    """
    numeric = numeric_candidates(df)
    candidates: list[tuple[float, str]] = []
    target_h = _column_height_m(speed_col or "")
    for col in numeric:
        low = str(col).lower()
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        # Direction-like numeric domain.  Allow 360 as a valid north encoding.
        frac = float(s.between(0, 360, inclusive="both").mean())
        if frac < 0.90:
            continue
        name_score = 0.0
        if any(k in low for k in ("vane", "direction", "wind dir", "wind_dir", "风向")):
            name_score += 100.0
        if "wd" in low:
            name_score += 80.0
        if "deg" in low or "degree" in low:
            name_score += 40.0
        if "avg" in low or "mean" in low or "平均" in low:
            name_score += 20.0
        if any(k in low for k in ("sd", "std", "gust", "max", "min")):
            name_score -= 60.0
        if name_score <= 0 and s.max() <= 90:
            # Avoid temperature and other low-range analog channels.
            continue
        h = _column_height_m(col)
        if target_h is not None and h is not None:
            # Same height dominates; otherwise nearest height wins.
            name_score += max(0.0, 80.0 - abs(h - target_h))
            if abs(h - target_h) < 1e-9:
                name_score += 120.0
        candidates.append((name_score, str(col)))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]
    return suggest_direction_column(df)


def _normalize_time(values: pd.Series) -> pd.Series:
    return _parse_time_values(values)

def _prepare_measured(df: pd.DataFrame, time_col: str, speed_col: str, direction_col: str) -> pd.DataFrame:
    raw_dir = pd.to_numeric(df[direction_col], errors="coerce")
    z = pd.DataFrame({
        "__time": _normalize_time(df[time_col]),
        "__speed": pd.to_numeric(df[speed_col], errors="coerce"),
        "__dir": raw_dir,
    })
    z = z.dropna(subset=["__time"]).sort_values("__time")
    # Keep physical low wind, only reject impossible negatives.
    z.loc[z["__speed"] < 0, "__speed"] = np.nan
    # Direction is used only as a concurrency gate here; do not synthesize it.
    invalid_dir = (~np.isfinite(z["__dir"])) | (z["__dir"] < 0) | (z["__dir"] > 360)
    z.loc[invalid_dir, "__dir"] = np.nan
    z.loc[z["__dir"].eq(360), "__dir"] = 0.0
    return z.drop_duplicates("__time", keep="last").reset_index(drop=True)


def _circular_mean_deg(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan
    r = np.deg2rad(x % 360.0)
    s, c = np.sin(r).mean(), np.cos(r).mean()
    if abs(s) < 1e-15 and abs(c) < 1e-15:
        return np.nan
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)


def _prepare_era(df: pd.DataFrame, time_col: str, speed_col: str, direction_col: str) -> pd.DataFrame:
    z = pd.DataFrame({
        "__time": _normalize_time(df[time_col]),
        "__speed": pd.to_numeric(df[speed_col], errors="coerce"),
        "__dir": pd.to_numeric(df[direction_col], errors="coerce") % 360.0,
    })
    z = z.dropna(subset=["__time"]).sort_values("__time")
    z.loc[z["__speed"] < 0, "__speed"] = np.nan

    # ERA5 files supplied by this project are already one unique row per exact
    # hour.  Avoid a Python-level circular-mean apply over ~180k hourly groups.
    # This changes no values; it only removes redundant resampling work.
    t = z["__time"]
    exact_hourly = (
        len(z) > 0
        and not t.duplicated().any()
        and (t.dt.minute.eq(0) & t.dt.second.eq(0) & t.dt.microsecond.eq(0)).all()
        and (len(z) < 2 or t.diff().dropna().eq(pd.Timedelta(hours=1)).all())
    )
    if exact_hourly:
        return z.dropna(subset=["__time", "__speed"]).reset_index(drop=True)

    # If denser/irregular, aggregate to the exact MCP 1h reference scale.
    zi = z.set_index("__time")
    speed = zi["__speed"].resample("1h", closed="left", label="left").mean()
    radians = np.deg2rad(zi["__dir"] % 360.0)
    sin_mean = np.sin(radians).resample("1h", closed="left", label="left").mean()
    cos_mean = np.cos(radians).resample("1h", closed="left", label="left").mean()
    direction = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)
    direction[(sin_mean.abs() < 1e-15) & (cos_mean.abs() < 1e-15)] = np.nan
    out = pd.DataFrame({"__time": speed.index, "__speed": speed.values, "__dir": direction.values})
    return out.dropna(subset=["__time", "__speed"]).reset_index(drop=True)


def _month_bounds(month_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month_start).replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    end_exclusive = start + pd.offsets.MonthBegin(1)
    end_inclusive = end_exclusive - pd.Timedelta(minutes=10)
    return start, pd.Timestamp(end_inclusive), pd.Timestamp(end_exclusive)


def _candidate_era_shifts(era: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp, shift_min: int, shift_max: int) -> list[int]:
    """Return integer ERA shifts that can cover the whole evaluation window.

    After shifting, ERA target-local timestamps are ``ERA_time + shift``.  A
    candidate is retained only when at least one user-allowed shift provides a
    complete hourly reference span for all 12 natural months.  This mirrors the
    original V0.8.4 rule that the evaluation-year candidate must be fillable by
    the supplied reference data; partial measured head/tail periods are allowed
    and are counted as missing coverage rather than rejected.
    """
    if era.empty:
        return []
    era_start = pd.Timestamp(era["__time"].min())
    era_end = pd.Timestamp(era["__time"].max())
    allowed = []
    for shift in range(int(shift_min), int(shift_max) + 1):
        shifted_start = era_start + pd.Timedelta(hours=shift)
        # ERA row at era_end represents the last hourly slot [era_end, era_end+1h).
        shifted_end_exclusive = era_end + pd.Timedelta(hours=shift + 1)
        if shifted_start <= start and shifted_end_exclusive >= end_exclusive:
            allowed.append(int(shift))
    return allowed


def select_evaluation_year(measured: pd.DataFrame, era: Optional[pd.DataFrame] = None, shift_min: int = -12, shift_max: int = 12) -> EvalYear:
    """Choose the highest-coverage 12-natural-month evaluation window.

    Important V0.8.4 behavior: "12 complete natural months" describes the
    *window boundaries*, not a requirement that the measured file itself must
    contain 100% of every month.  Partial head/tail months and internal gaps are
    valid; they are counted as missing points to be filled by MCP.  Candidate
    starts therefore run from 11 months before the first valid target month to
    the month containing the last valid target value.
    """
    if measured.empty:
        raise ValueError("测风数据为空，无法选择评价年。")

    valid_slots = measured.loc[measured["__speed"].notna(), ["__time", "__speed"]].copy()
    if valid_slots.empty:
        raise ValueError("目标风速列全部为空，无法选择评价年。")
    valid_slots["slot"] = valid_slots["__time"].dt.round("10min")
    valid_slots = valid_slots.drop_duplicates("slot", keep="last")
    valid_index = pd.DatetimeIndex(valid_slots["slot"]).sort_values()

    data_start = pd.Timestamp(valid_index.min())
    data_end = pd.Timestamp(valid_index.max())
    data_center = data_start + (data_end - data_start) / 2
    first_data_month = data_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    last_data_month = data_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    candidate_starts = pd.date_range(first_data_month - pd.DateOffset(months=11), last_data_month, freq="MS")

    rows = []
    for st in candidate_starts:
        st = pd.Timestamp(st)
        ex = st + pd.DateOffset(months=12)
        en = ex - pd.Timedelta(minutes=10)
        expected = int((ex - st).total_seconds() // 600)
        valid = int(((valid_index >= st) & (valid_index < ex)).sum())
        if valid < 1:
            continue

        allowed_shifts = list(range(int(shift_min), int(shift_max) + 1))
        if era is not None:
            allowed_shifts = _candidate_era_shifts(era, st, ex, shift_min, shift_max)
            if not allowed_shifts:
                continue

        # Head/tail slots outside the valid-target extent are explicitly MCP-fill
        # candidates, exactly as in the original program.
        idx = pd.date_range(st, en, freq="10min")
        outside = int(((idx < data_start) | (idx > data_end)).sum())
        center = st + (ex - st) / 2
        dist_days = abs((center - data_center).total_seconds()) / 86400.0
        rows.append({
            "评价年起点": st,
            "评价年终点": en,
            "评价年终点_不含": ex,
            "完整自然月数": 12,
            "理论10min点数": expected,
            "有效10min点数": valid,
            "需MCP补齐点数": expected - valid,
            "覆盖率(%)": valid / expected * 100.0 if expected else np.nan,
            "头尾补齐点数": outside,
            "距实测中心天数": dist_days,
            "ERA可完整覆盖平移(h)": ",".join(f"{x:+d}" for x in allowed_shifts),
        })

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        msg = (
            f"没有生成可用的12个完整自然月评价年候选。目标风速有效范围：{data_start:%Y-%m-%d %H:%M} 至 {data_end:%Y-%m-%d %H:%M}。"
        )
        if era is not None and not era.empty:
            msg += f" ERA范围：{pd.Timestamp(era['__time'].min()):%Y-%m-%d %H:%M} 至 {pd.Timestamp(era['__time'].max()):%Y-%m-%d %H:%M}。"
        raise ValueError(msg)

    candidates = candidates.sort_values(
        ["覆盖率(%)", "需MCP补齐点数", "头尾补齐点数", "距实测中心天数", "评价年起点"],
        ascending=[False, True, True, True, True], kind="mergesort"
    ).reset_index(drop=True)
    candidates["是否选中"] = ""
    candidates.loc[0, "是否选中"] = "是"

    best = candidates.iloc[0]
    start = pd.Timestamp(best["评价年起点"])
    end = pd.Timestamp(best["评价年终点"])
    end_exclusive = pd.Timestamp(best["评价年终点_不含"])

    monthly_rows = []
    for ms in pd.date_range(start, periods=12, freq="MS"):
        mst = pd.Timestamp(ms)
        mex = mst + pd.offsets.MonthBegin(1)
        men = mex - pd.Timedelta(minutes=10)
        expected = int((mex - mst).total_seconds() // 600)
        valid = int(((valid_index >= mst) & (valid_index < mex)).sum())
        monthly_rows.append({
            "年份": mst.year, "月份": mst.month, "月起点": mst, "月终点": men,
            "理论10min点数": expected, "有效10min点数": valid,
            "缺失/待MCP点数": expected - valid,
            "完整率(%)": valid / expected * 100.0 if expected else np.nan,
            "是否入选评价年": "是",
        })
    monthly = pd.DataFrame(monthly_rows)

    return EvalYear(
        start, end, end_exclusive, float(best["覆盖率(%)"]),
        int(best["有效10min点数"]), int(best["理论10min点数"]),
        candidates, monthly
    )


def _target_hourly_for_shift(measured: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    z = measured[(measured["__time"] >= start) & (measured["__time"] < end_exclusive)].copy()
    if z.empty:
        return pd.DataFrame(columns=["target_local_time", "target_ws"])
    # Exact same left/left hourly + 50%DCR concept used by V0.8.4 Windographer compatibility modules.
    s = pd.Series(z["__speed"].to_numpy(float), index=z["__time"])
    r = s.resample("1h", closed="left", label="left")
    mean, count = r.mean(), r.count()
    values = mean.to_numpy(dtype=float, copy=True)
    values[count.to_numpy(int) < 3] = np.nan  # full natural months => 6 expected ten-minute points/hour
    return pd.DataFrame({"target_local_time": mean.index, "target_ws": values})


def _target_hourly_any_available(measured: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Hourly measured mean for presentation/Final only: use any valid 10min point.

    IMPORTANT: this is deliberately separate from `_target_hourly_for_shift`.
    ERA timing and all hourly MCP training continue to use the 50%DCR rule.
    This helper only mirrors Windographer's displayed hourly measured value when
    an hour contains one or more valid 10-minute observations.
    """
    z = measured[(measured["__time"] >= start) & (measured["__time"] < end_exclusive)].copy()
    if z.empty:
        return pd.DataFrame(columns=["target_local_time", "target_ws"])
    s = pd.Series(z["__speed"].to_numpy(float), index=z["__time"])
    mean = s.resample("1h", closed="left", label="left").mean()
    return pd.DataFrame({"target_local_time": mean.index, "target_ws": mean.to_numpy(dtype=float)})


def _reference_10min_for_shift(era: pd.DataFrame, shift: int, *, start: pd.Timestamp | None = None, end_exclusive: pd.Timestamp | None = None) -> pd.DataFrame:
    ref_h = _full_reference_hourly(era, shift)
    if start is not None:
        ref_h = ref_h[ref_h["target_local_time"] >= pd.Timestamp(start).floor("1h")]
    if end_exclusive is not None:
        ref_h = ref_h[ref_h["target_local_time"] < pd.Timestamp(end_exclusive).ceil("1h")]
    ref10 = _expand_hourly_to_10min(ref_h)
    return ref10[["Timestamp", "ref_ws", "ref_dir"]].drop_duplicates("Timestamp", keep="last")


def _strict_concurrent_10min(measured: pd.DataFrame, era: pd.DataFrame, shift: int, *, start: pd.Timestamp | None = None, end_exclusive: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build Windographer-style 10-minute concurrent samples.

    A sample is concurrent only when target speed, target direction, reference
    speed and reference direction are all valid at the same processed 10-minute
    timestamp.  Hourly ERA/reference values are held across the six 10-minute
    slots, matching Windographer's processed reference timeline.
    """
    target = measured[["__time", "__speed", "__dir"]].copy()
    target["target_local_time"] = target["__time"].dt.round("10min")
    target["target_ws"] = pd.to_numeric(target["__speed"], errors="coerce")
    target["target_dir"] = pd.to_numeric(target["__dir"], errors="coerce")
    target = target[["target_local_time", "target_ws", "target_dir"]]
    target = target.drop_duplicates("target_local_time", keep="last")
    if start is not None:
        target = target[target["target_local_time"] >= pd.Timestamp(start)]
    if end_exclusive is not None:
        target = target[target["target_local_time"] < pd.Timestamp(end_exclusive)]

    ref_start = pd.Timestamp(start) if start is not None else (pd.Timestamp(target["target_local_time"].min()) if not target.empty else None)
    ref_end = pd.Timestamp(end_exclusive) if end_exclusive is not None else ((pd.Timestamp(target["target_local_time"].max()) + pd.Timedelta(minutes=10)) if not target.empty else None)
    ref10 = _reference_10min_for_shift(era, shift, start=ref_start, end_exclusive=ref_end).rename(columns={"Timestamp": "target_local_time"})
    merged = target.merge(ref10, on="target_local_time", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_ws", "target_dir", "ref_ws", "ref_dir"]
    ).copy()
    merged = merged[(merged["target_ws"] >= 0.0) & merged["target_dir"].between(0.0, 360.0, inclusive="both")]
    merged["Timestamp_UTC"] = merged["target_local_time"]
    merged["month"] = merged["target_local_time"].dt.month
    merged["main_sector"] = True
    return merged.sort_values("target_local_time").reset_index(drop=True)


def _wbl_concurrent_10min_speed_only(measured: pd.DataFrame, era: pd.DataFrame, shift: int, *, start: pd.Timestamp | None = None, end_exclusive: pd.Timestamp | None = None) -> pd.DataFrame:
    """V1.0.54 WBL-specific 10-minute concurrent samples.

    Windographer WBL speed fitting needs target wind speed plus reference
    speed/direction. Target wind direction is NOT a required concurrent
    channel. Hourly reference values are held across the six 10-minute slots,
    exactly as in the existing 10-minute prediction timeline.

    This intentionally leaves the generic eight-method 10-minute training set
    unchanged; only WBL receives this speed-specific concurrent set.
    """
    target = measured[["__time", "__speed"]].copy()
    target["target_local_time"] = pd.to_datetime(target["__time"]).dt.round("10min")
    target["target_ws"] = pd.to_numeric(target["__speed"], errors="coerce")
    target = target[["target_local_time", "target_ws"]].drop_duplicates("target_local_time", keep="last")
    if start is not None:
        target = target[target["target_local_time"] >= pd.Timestamp(start)]
    if end_exclusive is not None:
        target = target[target["target_local_time"] < pd.Timestamp(end_exclusive)]

    ref_start = pd.Timestamp(start) if start is not None else (pd.Timestamp(target["target_local_time"].min()) if not target.empty else None)
    ref_end = pd.Timestamp(end_exclusive) if end_exclusive is not None else ((pd.Timestamp(target["target_local_time"].max()) + pd.Timedelta(minutes=10)) if not target.empty else None)
    ref10 = _reference_10min_for_shift(era, shift, start=ref_start, end_exclusive=ref_end).rename(columns={"Timestamp": "target_local_time"})
    merged = target.merge(ref10, on="target_local_time", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_ws", "ref_ws", "ref_dir"]).copy()
    merged = merged[(merged["target_ws"] >= 0.0) & merged["ref_dir"].between(0.0, 360.0, inclusive="both")]
    # Keep a target_dir column for export compatibility, but do not use it as
    # a filter or a WBL input. Missing target direction therefore stays NaN.
    target_dir_map = measured[["__time", "__dir"]].copy()
    target_dir_map["target_local_time"] = pd.to_datetime(target_dir_map["__time"]).dt.round("10min")
    target_dir_map["target_dir"] = pd.to_numeric(target_dir_map["__dir"], errors="coerce")
    target_dir_map = target_dir_map[["target_local_time", "target_dir"]].drop_duplicates("target_local_time", keep="last")
    merged = merged.merge(target_dir_map, on="target_local_time", how="left")
    merged["Timestamp_UTC"] = merged["target_local_time"]
    merged["month"] = pd.to_datetime(merged["target_local_time"]).dt.month
    merged["main_sector"] = True
    cols = ["target_local_time", "target_ws", "target_dir", "ref_ws", "ref_dir", "Timestamp_UTC", "month", "main_sector"]
    return merged[cols].sort_values("target_local_time").reset_index(drop=True)


def find_best_shift(measured: pd.DataFrame, era: pd.DataFrame, eval_year: EvalYear, shift_min=-12, shift_max=12) -> tuple[int, pd.DataFrame]:
    rows = []
    for shift in range(int(shift_min), int(shift_max) + 1):
        m = _strict_concurrent_10min(
            measured, era, shift, start=eval_year.start, end_exclusive=eval_year.end_exclusive
        )
        if len(m) >= 144 and m["target_ws"].std() > 0 and m["ref_ws"].std() > 0:
            r = float(m["target_ws"].corr(m["ref_ws"]))
            r2 = r * r
        else:
            r, r2 = np.nan, np.nan
        rows.append({"ERA时间平移(h)": shift, "并发10min点数": len(m), "R": r, "R²": r2})
    diag = pd.DataFrame(rows)
    valid = diag[np.isfinite(pd.to_numeric(diag["R²"], errors="coerce"))].copy()
    if valid.empty:
        raise ValueError("搜索范围内没有足够的严格10min并发数据，无法计算R²。")
    valid["abs_shift"] = valid["ERA时间平移(h)"].abs()
    valid = valid.sort_values(["R²", "abs_shift", "ERA时间平移(h)"], ascending=[False, True, True], kind="mergesort")
    best_shift = int(valid.iloc[0]["ERA时间平移(h)"])
    diag["是否最佳"] = np.where(diag["ERA时间平移(h)"].eq(best_shift), "是", "")
    return best_shift, diag


def _full_reference_hourly(era: pd.DataFrame, shift: int) -> pd.DataFrame:
    z = era.copy()
    z["ERA原始时刻"] = z["__time"]
    z["target_local_time"] = z["__time"] + pd.Timedelta(hours=int(shift))
    z["ref_ws"] = z["__speed"]
    z["ref_dir"] = z["__dir"]
    return z[["ERA原始时刻", "target_local_time", "ref_ws", "ref_dir"]].sort_values("target_local_time").drop_duplicates("target_local_time", keep="last").reset_index(drop=True)


def _parse_optional_training_bound(value: str | None, *, is_end: bool = False) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, errors="raise")
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    ts = pd.Timestamp(ts)
    # Windographer date-range controls are day based. If the user supplied only
    # a date for the end bound, include that whole day by converting to the next
    # midnight as an exclusive bound. Exact date-times remain exact.
    has_clock = any(ch in text for ch in (":", "T"))
    if is_end and not has_clock:
        ts = ts + pd.Timedelta(days=1)
    return ts


def _apply_concurrent_limit(frame: pd.DataFrame, start: str | None, end: str | None, time_col: str) -> pd.DataFrame:
    out = frame.copy()
    st = _parse_optional_training_bound(start, is_end=False)
    en = _parse_optional_training_bound(end, is_end=True)
    if st is not None:
        out = out[pd.to_datetime(out[time_col]) >= st]
    if en is not None:
        out = out[pd.to_datetime(out[time_col]) < en]
    return out.reset_index(drop=True)


def _aggregate_target_independently_to_hourly(
    measured: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end_exclusive: pd.Timestamp | None = None,
    minimum_dcr: float = 0.50,
) -> pd.DataFrame:
    """Build processed 1h target channels BEFORE cross-dataset concurrency.

    V1.0.52 experimental Windographer ordering:
    1) target speed and target direction are each resampled from their own valid
       10-minute observations to a 60-minute processed value;
    2) each channel must independently satisfy the 50% DCR rule (>=3 of 6
       ten-minute observations for a full hour);
    3) cross-dataset concurrency with the processed reference is applied only
       afterwards.

    This intentionally differs from the legacy path, which first required every
    10-minute target/reference channel to be concurrent and only then averaged
    those surviving rows to 1h.
    """
    z = measured[["__time", "__speed", "__dir"]].copy().sort_values("__time")
    if start is not None:
        z = z[z["__time"] >= pd.Timestamp(start)]
    if end_exclusive is not None:
        z = z[z["__time"] < pd.Timestamp(end_exclusive)]
    if z.empty:
        return pd.DataFrame(columns=[
            "target_local_time", "target_ws", "target_dir",
            "target_speed_10min_count", "target_dir_10min_count", "expected_10min_count",
        ])

    z["hour"] = pd.to_datetime(z["__time"]).dt.floor("1h")
    rows = []
    min_count = int(math.ceil(6 * float(minimum_dcr)))
    for hour, g in z.groupby("hour", sort=True):
        ws = pd.to_numeric(g["__speed"], errors="coerce")
        wd = pd.to_numeric(g["__dir"], errors="coerce")
        ws_count = int(np.isfinite(ws.to_numpy(float)).sum())
        wd_count = int(np.isfinite(wd.to_numpy(float)).sum())
        target_ws = float(ws.mean()) if ws_count >= min_count else np.nan
        target_dir = _circular_mean_deg(wd) if wd_count >= min_count else np.nan
        rows.append({
            "target_local_time": pd.Timestamp(hour),
            "target_ws": target_ws,
            "target_dir": target_dir,
            "target_speed_10min_count": ws_count,
            "target_dir_10min_count": wd_count,
            "expected_10min_count": 6,
        })
    return pd.DataFrame(rows)


def _hourly_concurrent_after_independent_resample(
    measured: pd.DataFrame,
    era: pd.DataFrame,
    shift: int,
    *,
    start: pd.Timestamp | None = None,
    end_exclusive: pd.Timestamp | None = None,
    minimum_dcr: float = 0.50,
) -> pd.DataFrame:
    """Windographer candidate 1h training set: resample first, concur second.

    Target and reference datasets are processed independently to 60 minutes.
    Only after those processed hourly channels exist do we require target speed,
    target direction, reference speed and reference direction to be concurrent.
    """
    target_h = _aggregate_target_independently_to_hourly(
        measured, start=start, end_exclusive=end_exclusive, minimum_dcr=minimum_dcr
    )
    ref_h = _full_reference_hourly(era, shift).copy()
    if start is not None:
        ref_h = ref_h[ref_h["target_local_time"] >= pd.Timestamp(start).floor("1h")]
    if end_exclusive is not None:
        ref_h = ref_h[ref_h["target_local_time"] < pd.Timestamp(end_exclusive).ceil("1h")]

    merged = target_h.merge(
        ref_h[["target_local_time", "ref_ws", "ref_dir"]],
        on="target_local_time", how="inner"
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_ws", "target_dir", "ref_ws", "ref_dir"]
    ).copy()
    merged = merged[(merged["target_ws"] >= 0.0) & merged["target_dir"].between(0.0, 360.0, inclusive="both")]
    merged["Timestamp_UTC"] = merged["target_local_time"]
    merged["month"] = pd.to_datetime(merged["target_local_time"]).dt.month
    merged["main_sector"] = True
    return merged.sort_values("target_local_time").reset_index(drop=True)


def _hourly_wbl_concurrent_speed_only(
    measured: pd.DataFrame,
    era: pd.DataFrame,
    shift: int,
    *,
    start: pd.Timestamp | None = None,
    end_exclusive: pd.Timestamp | None = None,
    minimum_dcr: float = 0.50,
) -> pd.DataFrame:
    """V1.0.53 WBL-specific 1h training set.

    WBL speed fitting needs target wind speed plus reference speed/direction.
    Target direction is deliberately NOT a required concurrent channel.
    Target speed is averaged to 1h when its own DCR>=50%, then joined to
    native hourly reference speed/direction.
    """
    z = measured[["__time", "__speed"]].copy().sort_values("__time")
    if start is not None:
        z = z[z["__time"] >= pd.Timestamp(start)]
    if end_exclusive is not None:
        z = z[z["__time"] < pd.Timestamp(end_exclusive)]

    if z.empty:
        return pd.DataFrame(columns=[
            "target_local_time", "target_ws", "target_speed_10min_count",
            "expected_10min_count", "ref_ws", "ref_dir", "Timestamp_UTC",
            "month", "main_sector",
        ])

    z["hour"] = pd.to_datetime(z["__time"]).dt.floor("1h")
    min_count = int(math.ceil(6 * float(minimum_dcr)))
    rows = []
    for hour, g in z.groupby("hour", sort=True):
        ws = pd.to_numeric(g["__speed"], errors="coerce")
        ws_count = int(np.isfinite(ws.to_numpy(float)).sum())
        if ws_count < min_count:
            continue
        rows.append({
            "target_local_time": pd.Timestamp(hour),
            "target_ws": float(ws.mean()),
            "target_speed_10min_count": ws_count,
            "expected_10min_count": 6,
        })
    target_h = pd.DataFrame(rows)
    if target_h.empty:
        return target_h

    ref_h = _full_reference_hourly(era, shift).copy()
    if start is not None:
        ref_h = ref_h[ref_h["target_local_time"] >= pd.Timestamp(start).floor("1h")]
    if end_exclusive is not None:
        ref_h = ref_h[ref_h["target_local_time"] < pd.Timestamp(end_exclusive).ceil("1h")]

    merged = target_h.merge(
        ref_h[["target_local_time", "ref_ws", "ref_dir"]],
        on="target_local_time", how="inner"
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_ws", "ref_ws", "ref_dir"]
    ).copy()
    merged = merged[pd.to_numeric(merged["target_ws"], errors="coerce") >= 0.0]
    merged["Timestamp_UTC"] = merged["target_local_time"]
    merged["month"] = pd.to_datetime(merged["target_local_time"]).dt.month
    merged["main_sector"] = True
    return merged.sort_values("target_local_time").reset_index(drop=True)


def _aggregate_strict_concurrent_to_hourly(concurrent10: pd.DataFrame, minimum_dcr: float = 0.50) -> pd.DataFrame:
    """Aggregate already-concurrent 10min samples to Windographer-style 1h SS input."""
    if concurrent10.empty:
        return pd.DataFrame(columns=["target_local_time", "target_ws", "target_dir", "ref_ws", "ref_dir", "concurrent_10min_count", "expected_10min_count"])
    z = concurrent10.copy().sort_values("target_local_time")
    z["hour"] = pd.to_datetime(z["target_local_time"]).dt.floor("1h")
    coverage_start = pd.Timestamp(z["target_local_time"].min())
    coverage_end = pd.Timestamp(z["target_local_time"].max())

    rows = []
    for hour, g in z.groupby("hour", sort=True):
        hour = pd.Timestamp(hour)
        # Six expected slots for full interior hours; partial first/last hours
        # use only the slots that fall inside the actual overlap coverage.
        slots = pd.date_range(hour, hour + pd.Timedelta(minutes=50), freq="10min")
        expected = int(((slots >= coverage_start) & (slots <= coverage_end)).sum())
        expected = max(1, expected)
        count = int(len(g))
        if count < int(math.ceil(expected * float(minimum_dcr))):
            continue
        rows.append({
            "target_local_time": hour,
            "target_ws": float(pd.to_numeric(g["target_ws"], errors="coerce").mean()),
            "target_dir": _circular_mean_deg(g["target_dir"]),
            "ref_ws": float(pd.to_numeric(g["ref_ws"], errors="coerce").mean()),
            "ref_dir": _circular_mean_deg(g["ref_dir"]),
            "concurrent_10min_count": count,
            "expected_10min_count": expected,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Timestamp_UTC"] = out["target_local_time"]
    out["month"] = out["target_local_time"].dt.month
    out["main_sector"] = True
    return out.reset_index(drop=True)


def _full_concurrent_sets(measured: pd.DataFrame, era: pd.DataFrame, shift: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (strict 10min concurrent, 50%DCR hourly concurrent for SpeedSort)."""
    concurrent10 = _strict_concurrent_10min(measured, era, shift)
    concurrent1h = _aggregate_strict_concurrent_to_hourly(concurrent10, minimum_dcr=0.50)
    return concurrent10, concurrent1h


def _metrics(observed, predicted, scale: str) -> dict:
    o = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(predicted), errors="coerce").to_numpy(float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if len(o) == 0:
        return {"诊断时间尺度": scale, "样本数": 0, "R²": np.nan, "MBE(m/s)": np.nan, "MAE(m/s)": np.nan, "RMSE(m/s)": np.nan}
    err = p - o
    r2 = np.nan
    if len(o) >= 2 and np.std(o) > 0 and np.std(p) > 0:
        r = float(np.corrcoef(o, p)[0, 1])
        r2 = r * r
    return {
        "诊断时间尺度": scale, "样本数": len(o), "R²": r2,
        "MBE(m/s)": float(np.mean(err)), "MAE(m/s)": float(np.mean(np.abs(err))),
        "RMSE(m/s)": float(np.sqrt(np.mean(err ** 2))),
        "实测平均(m/s)": float(np.mean(o)), "拟合平均(m/s)": float(np.mean(p)),
    }


def _realization_seeds(master_seed: int, count: int) -> list[int]:
    """Deterministic list of reproducible child seeds; first seed is the master itself."""
    n = max(1, int(count))
    master = int(master_seed) & 0x7FFFFFFF
    seeds = [master]
    if n == 1:
        return seeds
    rng = np.random.default_rng(master)
    seen = {master}
    while len(seeds) < n:
        cand = int(rng.integers(0, 2147483647))
        if cand not in seen:
            seen.add(cand)
            seeds.append(cand)
    return seeds


def _wind_series_stats(values, prefix: str) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            f"{prefix}均值(m/s)": np.nan, f"{prefix}标准差(m/s)": np.nan,
            f"{prefix}P95(m/s)": np.nan, f"{prefix}WPD代理(v3)": np.nan,
        }
    xp = np.maximum(x, 0.0)
    return {
        f"{prefix}均值(m/s)": float(np.mean(x)),
        f"{prefix}标准差(m/s)": float(np.std(x)),
        f"{prefix}P95(m/s)": float(np.quantile(x, 0.95)),
        f"{prefix}WPD代理(v3)": float(np.mean(xp ** 3)),
    }


def _realization_row(method: str, run_no: int, seed: int, observed, train_pred, eval_pred, scale: str = "10min严格并发") -> dict:
    m = _metrics(observed, train_pred, scale)
    row = {
        "方法": method, "Run": int(run_no), "Seed": int(seed),
        "同期样本数": int(m.get("样本数", 0)),
        "同期Pearson_r2": m.get("R²", np.nan),
        "同期MBE(m/s)": m.get("MBE(m/s)", np.nan),
        "同期MAE(m/s)": m.get("MAE(m/s)", np.nan),
        "同期RMSE(m/s)": m.get("RMSE(m/s)", np.nan),
    }
    row.update(_wind_series_stats(train_pred, "同期Fit_"))
    row.update(_wind_series_stats(eval_pred, "评价年Fit_"))
    return row


def _choose_representative_realization(rows: list[dict]) -> tuple[pd.DataFrame, int]:
    """Pick the run closest to the component-wise median state, never the minimum-error run."""
    df = pd.DataFrame(rows).copy()
    if df.empty:
        raise RuntimeError("随机实现汇总为空，无法选择代表性Seed。")
    features = [
        "同期MBE(m/s)", "同期MAE(m/s)", "同期RMSE(m/s)",
        "同期Fit_均值(m/s)", "同期Fit_标准差(m/s)", "同期Fit_P95(m/s)", "同期Fit_WPD代理(v3)",
        "评价年Fit_均值(m/s)", "评价年Fit_标准差(m/s)", "评价年Fit_P95(m/s)", "评价年Fit_WPD代理(v3)",
    ]
    dist = np.zeros(len(df), dtype=float)
    used = np.zeros(len(df), dtype=float)
    for col in features:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        finite = np.isfinite(x)
        if not finite.any():
            continue
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x[finite] - med)))
        q25, q75 = np.nanquantile(x[finite], [0.25, 0.75])
        robust_scale = max(1.4826 * mad, float((q75 - q25) / 1.349) if q75 > q25 else 0.0, 1e-9)
        z = np.abs(x - med) / robust_scale
        good = np.isfinite(z)
        dist[good] += z[good]
        used[good] += 1.0
    dist = np.where(used > 0, dist / used, np.inf)
    idx = int(np.argmin(dist))
    selected_seed = int(df.iloc[idx]["Seed"])
    df["代表性距离"] = dist
    df["是否最终代表"] = "否"
    df.loc[df.index[idx], "是否最终代表"] = "是"
    return df, selected_seed


def _expand_hourly_to_10min(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    base = hourly.copy()
    parts = []
    for minute in (0, 10, 20, 30, 40, 50):
        p = base.copy()
        p["Timestamp"] = p["target_local_time"] + pd.Timedelta(minutes=minute)
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("Timestamp").reset_index(drop=True)




def _split_monthly_50_50(df: pd.DataFrame, seed: int, mode: str = "day") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each calendar month 50%/50% by whole day or whole hour blocks.

    ``mode='day'`` keeps every 10-minute point from the same day in the same half.
    ``mode='hour'`` does the same at hourly-block level. This prevents adjacent
    records inside a block from leaking across training and validation while each
    month remains represented in both halves whenever the month has >=2 blocks.
    """
    if len(df) < 4:
        raise ValueError("50%/50%抽样需要至少4个同期样本。")
    work = df.copy().reset_index(drop=True)
    ts = pd.to_datetime(work["target_local_time"], errors="coerce")
    mode_key = str(mode or "day").strip().lower()
    if mode_key not in {"day", "hour"}:
        raise ValueError(f"未知抽样方式：{mode}；请选择 day 或 hour。")
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    train_idx: list[int] = []
    valid_idx: list[int] = []
    ym = ts.dt.to_period("M")

    for _, month_idx in work.groupby(ym, sort=True).groups.items():
        month_idx = np.asarray(list(month_idx), dtype=int)
        month_ts = ts.iloc[month_idx]
        if mode_key == "day":
            units = month_ts.dt.floor("D")
        else:
            units = month_ts.dt.floor("h")

        # Group row indices by whole block, shuffle BLOCKS, then split blocks 50/50.
        unit_to_rows: dict[pd.Timestamp, list[int]] = {}
        for row_idx, unit in zip(month_idx.tolist(), units.tolist()):
            unit_to_rows.setdefault(pd.Timestamp(unit), []).append(int(row_idx))
        keys = list(unit_to_rows.keys())
        if len(keys) <= 1:
            target = train_idx if int(rng.integers(0, 2)) == 0 else valid_idx
            for key in keys:
                target.extend(unit_to_rows[key])
            continue

        shuffled_keys = [keys[i] for i in rng.permutation(len(keys))]
        n_train_units = max(1, min(len(shuffled_keys) // 2, len(shuffled_keys) - 1))
        train_keys = set(shuffled_keys[:n_train_units])
        for key, rows in unit_to_rows.items():
            (train_idx if key in train_keys else valid_idx).extend(rows)

    if not train_idx or not valid_idx:
        # Global block fallback for pathological data with singleton months.
        all_units = ts.dt.floor("D" if mode_key == "day" else "h")
        unit_to_rows: dict[pd.Timestamp, list[int]] = {}
        for row_idx, unit in enumerate(all_units.tolist()):
            unit_to_rows.setdefault(pd.Timestamp(unit), []).append(int(row_idx))
        keys = list(unit_to_rows.keys())
        if len(keys) < 2:
            raise ValueError("可用于50%/50%抽样的独立时间块不足2个。")
        shuffled_keys = [keys[i] for i in rng.permutation(len(keys))]
        n_train_units = max(1, min(len(shuffled_keys) // 2, len(shuffled_keys) - 1))
        train_keys = set(shuffled_keys[:n_train_units])
        train_idx, valid_idx = [], []
        for key, rows in unit_to_rows.items():
            (train_idx if key in train_keys else valid_idx).extend(rows)

    train = work.iloc[sorted(train_idx)].copy().reset_index(drop=True)
    valid = work.iloc[sorted(valid_idx)].copy().reset_index(drop=True)
    return train, valid


def _distribution_error(observed, predicted, bin_width: float = 1.0) -> float:
    """Distribution Error (DE) using a shared wind-speed frequency histogram.

    Both series use identical 1 m/s bins. Frequencies are expressed as percent
    before applying the Pearson-style distribution statistic, so DE remains a
    distribution-shape score rather than scaling with sample count.
    """
    o = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(predicted), errors="coerce").to_numpy(float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if len(o) == 0:
        return np.nan
    bw = max(float(bin_width), 1e-9)
    lo = min(0.0, math.floor(float(min(np.min(o), np.min(p))) / bw) * bw)
    hi = math.ceil(float(max(np.max(o), np.max(p))) / bw) * bw
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + bw
    bins = np.arange(lo, hi + bw * 1.000001, bw, dtype=float)
    if len(bins) < 2:
        bins = np.asarray([lo, lo + bw], dtype=float)
    obs_counts, _ = np.histogram(o, bins=bins)
    pred_counts, _ = np.histogram(p, bins=bins)
    obs_freq = obs_counts.astype(float) / max(float(obs_counts.sum()), 1.0) * 100.0
    pred_freq = pred_counts.astype(float) / max(float(pred_counts.sum()), 1.0) * 100.0
    positive = obs_freq > 0.0
    if not positive.any():
        return np.nan
    return float(np.sum(((pred_freq[positive] - obs_freq[positive]) ** 2) / obs_freq[positive]))

def _fit_named_deterministic(method: str, train: pd.DataFrame, fit_min_speed: float, wbl_mode: str, fit_resolution: str = "10min"):
    # Windographer 4.2.25 black-box evidence: BSR/LLS/TLS/VR use the same
    # resolution-dependent QxS minimum as WBL: 10min=50, 1h=24.
    # Keep 10min untouched because it already matches WG; only 1h changes.
    local_min = 24 if str(fit_resolution).strip().lower() == "1h" else 50
    if method == "BSR":
        return fit_windographer_bsr(train, fit_min_speed=float(fit_min_speed), minimum_local_samples=local_min)
    if method == "LLS":
        return fit_windographer_lls(train, fit_min_speed=float(fit_min_speed), minimum_local_samples=local_min)
    if method == "TLS":
        return fit_windographer_tls(train, fit_min_speed=float(fit_min_speed), minimum_local_samples=local_min)
    if method == "VR":
        return fit_windographer_vr(train, fit_min_speed=float(fit_min_speed), minimum_local_samples=local_min)
    if method == "VS":
        return fit_windographer_vs(train)
    if method == "WBL":
        mode_key = str(wbl_mode or "windographer_compat").strip().lower()
        if mode_key in {"windographer", "compat", "windographer_compat", "wg"}:
            wbl_min_local = 24 if str(fit_resolution).strip().lower() == "1h" else 50
            return fit_windographer_wbl(train, minimum_local_samples=wbl_min_local)
        return fit_adaptive_wbl(train)
    raise ValueError(f"未知确定性MCP方法：{method}")


def _predict_named_deterministic(method: str, model, frame: pd.DataFrame) -> np.ndarray:
    if method in {"BSR", "LLS", "TLS", "VR", "VS"}:
        return model.predict_frame(frame, local_utc_offset_hours=0)
    if method == "WBL":
        return model.predict_frame(frame)
    raise ValueError(f"未知确定性MCP方法：{method}")


def _deterministic_sampling_validation(
    concurrent_data: pd.DataFrame,
    *,
    fit_min_speed: float,
    wbl_mode: str,
    runs: int,
    master_seed: int,
    sampling_mode: str,
    fit_resolution: str = "10min",
    wbl_concurrent_data: pd.DataFrame | None = None,
    progress: Callable[[str], None],
) -> pd.DataFrame:
    """Repeated 50/50 holdout validation on the globally selected fit resolution.

    V1.0.56: every deterministic method receives the same dependency-correct
    concurrent pool (Target WS + Reference WS + Reference WD). The configurable
    low-speed threshold is an EVALUATION-TRAINING rule for BSR / LLS / TLS / VR
    only. VS and WBL always fit the complete sampled training half. The
    validation half is never low-speed filtered.
    """
    n_runs = max(1, int(runs))
    mode_key = str(sampling_mode or "day").strip().lower()
    mode_label = "月内按天抽" if mode_key == "day" else "月内按小时抽"
    resolution = str(fit_resolution or "10min").strip().lower()
    if resolution not in {"10min", "1h"}:
        raise ValueError(f"未知拟合分辨率：{fit_resolution}；请选择10min或1h。")
    scale_label = "1h" if resolution == "1h" else "10min"
    methods = ("BSR", "LLS", "TLS", "VR", "VS", "WBL")
    threshold_methods = {"BSR", "LLS", "TLS", "VR"}
    accum: dict[str, list[tuple[float, float, float]]] = {m: [] for m in methods}

    for i in range(n_runs):
        seed = int(master_seed) + i
        train_raw, valid_i = _split_monthly_50_50(concurrent_data, seed, mode=mode_key)
        if wbl_concurrent_data is not None:
            wbl_train_raw, wbl_valid_i = _split_monthly_50_50(wbl_concurrent_data, seed, mode=mode_key)
        else:
            wbl_train_raw, wbl_valid_i = train_raw, valid_i
        threshold = float(fit_min_speed)
        threshold_mask = (
            (pd.to_numeric(train_raw["target_ws"], errors="coerce") >= threshold)
            & (pd.to_numeric(train_raw["ref_ws"], errors="coerce") >= threshold)
        )
        threshold_count = int(threshold_mask.sum())
        progress(
            f"确定性方法50/50抽样 {i + 1}/{n_runs}：{scale_label}拟合；{mode_label}；"
            f"训练原始={len(train_raw):,}；BSR/LLS/TLS/VR阈值后={threshold_count:,}；"
            f"VS/WBL均使用八法共同三通道训练池；WBL训练={len(wbl_train_raw):,}、验证={len(wbl_valid_i):,}；"
            f"共同验证={len(valid_i):,}；seed={seed}"
        )
        for method in methods:
            try:
                method_threshold = threshold if method in threshold_methods else 0.0
                method_train = wbl_train_raw if method == "WBL" else train_raw
                method_valid = wbl_valid_i if method == "WBL" else valid_i
                model = _fit_named_deterministic(method, method_train, method_threshold, wbl_mode, fit_resolution=resolution)
                pred = _predict_named_deterministic(method, model, method_valid)
                m = _metrics(method_valid["target_ws"], pred, f"{scale_label}验证集50%")
                mbe = float(m.get("MBE(m/s)", np.nan))
                mae = float(m.get("MAE(m/s)", np.nan))
                de = _distribution_error(method_valid["target_ws"], pred)
                if np.isfinite(mbe) and np.isfinite(mae) and np.isfinite(de):
                    accum[method].append((abs(mbe), mae, de))
            except Exception as exc:
                progress(f"  {method} 本轮评价失败：{exc}")

    rows: list[dict] = []
    for method in methods:
        vals = accum[method]
        if vals:
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "方法": method,
                "拟合分辨率": scale_label,
                "平均|MBE|(m/s)": float(np.mean(arr[:, 0])),
                "平均MAE(m/s)": float(np.mean(arr[:, 1])),
                "平均DE(%)": float(np.mean(arr[:, 2])),
            })
        else:
            rows.append({"方法": method, "拟合分辨率": scale_label, "平均|MBE|(m/s)": np.nan, "平均MAE(m/s)": np.nan, "平均DE(%)": np.nan})
    return pd.DataFrame(rows)


def _fit_all(measured: pd.DataFrame, era: pd.DataFrame, eval_year: EvalYear, shift: int, fit_min_speed: float, progress: Callable[[str], None], *, concurrent_start: str | None = None, concurrent_end: str | None = None, ss_random_seed: int = 20260907, wbl_mode: str = "windographer_compat", mts_moving_average_hours: int = 3, mts_random_seed: int = 20260831, mts_edge_tolerance_states: float = 1.5, mts_edge_max_attempts: int = 500, mts_edge_mode: str = "right_rejection", deterministic_sampling_enabled: bool = True, deterministic_sampling_runs: int = 50, deterministic_sampling_seed: int = 42, deterministic_sampling_mode: str = "day", stochastic_realizations: int = 9, fit_resolution: str = "10min", training_scope: str = "all_valid"):
    # V1.0.56: all eight MCP methods now use the same dependency-correct
    # concurrent training pool that made WBL match Windographer:
    #   Target wind speed + Reference wind speed + Reference wind direction.
    # Target wind direction is retained only as optional metadata and is NEVER
    # allowed to delete an otherwise valid wind-speed training sample.
    # The old four-channel pools are retained only as diagnostic controls.
    legacy_strict10_all = _strict_concurrent_10min(measured, era, shift)
    concurrent10_all = _wbl_concurrent_10min_speed_only(measured, era, shift)
    legacy_allchannel1h_all = _hourly_concurrent_after_independent_resample(
        measured, era, shift, minimum_dcr=0.50
    )
    concurrent1h_all = _hourly_wbl_concurrent_speed_only(
        measured, era, shift, minimum_dcr=0.50
    )

    # V1.0.45: training-period scope is explicit instead of hard-wired.
    # all_valid      -> all valid concurrent measured/reference overlap
    # evaluation_year -> only valid concurrent samples inside selected evaluation year
    # Optional start/end fields further limit the selected base scope; blank fields do
    # not silently change the scope.
    scope = str(training_scope or "all_valid").strip().lower()
    if scope not in {"all_valid", "evaluation_year"}:
        raise ValueError(f"未知训练数据范围：{training_scope}；请选择全部有效测风数据或仅评价年有效测风数据。")

    coverage_starts = []
    coverage_ends = []
    if not concurrent10_all.empty:
        coverage_starts.append(pd.Timestamp(concurrent10_all["target_local_time"].min()))
        coverage_ends.append(pd.Timestamp(concurrent10_all["target_local_time"].max()) + pd.Timedelta(minutes=10))
    if not concurrent1h_all.empty:
        coverage_starts.append(pd.Timestamp(concurrent1h_all["target_local_time"].min()))
        coverage_ends.append(pd.Timestamp(concurrent1h_all["target_local_time"].max()) + pd.Timedelta(hours=1))
    if not coverage_starts:
        raise ValueError("目标与ERA没有可用于MCP拟合的同期数据。")
    base_start = min(coverage_starts)
    base_end = max(coverage_ends)
    scope_label = "全部有效同期"
    if scope == "evaluation_year":
        base_start = pd.Timestamp(eval_year.start)
        base_end = pd.Timestamp(eval_year.end_exclusive)
        scope_label = "仅评价年有效同期"

    user_start = _parse_optional_training_bound(concurrent_start, is_end=False)
    user_end = _parse_optional_training_bound(concurrent_end, is_end=True)
    train_start = max(base_start, user_start) if user_start is not None else base_start
    train_end = min(base_end, user_end) if user_end is not None else base_end
    if train_end <= train_start:
        raise ValueError(f"训练同期结束必须晚于开始：{train_start} ~ {train_end}")

    # Apply the requested training window to both the old diagnostic pool and
    # the new production pool.
    legacy_strict10 = legacy_strict10_all[
        (pd.to_datetime(legacy_strict10_all["target_local_time"]) >= train_start)
        & (pd.to_datetime(legacy_strict10_all["target_local_time"]) < train_end)
    ].copy().reset_index(drop=True)
    concurrent10 = _wbl_concurrent_10min_speed_only(
        measured, era, shift, start=train_start, end_exclusive=train_end
    )

    # V1.0.56 self-check: the production three-channel pool must be a superset
    # of the old TargetWS+TargetWD+RefWS+RefWD pool.  These extra timestamps are
    # now used by ALL eight methods, not WBL alone.
    _legacy_ts10 = pd.Index(pd.to_datetime(legacy_strict10["target_local_time"]))
    _extra_mask10 = ~pd.to_datetime(concurrent10["target_local_time"]).isin(_legacy_ts10)
    common10_extra = concurrent10.loc[_extra_mask10].copy().reset_index(drop=True)
    if len(concurrent10) < len(legacy_strict10):
        raise RuntimeError(
            f"V1.0.56自检失败：八法三通道10min并发点({len(concurrent10)})少于旧四通道严格并发点({len(legacy_strict10)})。"
        )

    # Preserve both old 1h constructions for diagnostics only.  Production 1h
    # fitting uses the already-verified WBL rule: Target WS independently
    # aggregates to 1h with DCR>=50%, then joins native hourly Ref WS/WD.
    legacy_concurrent1h = _aggregate_strict_concurrent_to_hourly(legacy_strict10, minimum_dcr=0.50)
    legacy_allchannel1h = _hourly_concurrent_after_independent_resample(
        measured, era, shift, start=train_start, end_exclusive=train_end, minimum_dcr=0.50
    )
    concurrent1h = _hourly_wbl_concurrent_speed_only(
        measured, era, shift, start=train_start, end_exclusive=train_end, minimum_dcr=0.50
    )

    resolution = str(fit_resolution or "10min").strip().lower()
    if resolution not in {"10min", "1h"}:
        raise ValueError(f"未知拟合分辨率：{fit_resolution}；请选择10min或1h。")
    fit_train = concurrent1h if resolution == "1h" else concurrent10
    # WBL no longer needs a special training pool: all eight methods share the
    # same dependency-correct pool.  This leaves the already-matched WBL 1h/10min
    # behavior numerically unchanged.
    wbl_fit_train = fit_train
    fit_label = (
        "1h仅Target风速+Reference风速/风向并发"
        if resolution == "1h"
        else "10min仅Target风速+Reference风速/风向并发"
    )
    wbl_fit_label = fit_label
    fit_step_minutes = 60.0 if resolution == "1h" else 10.0

    if len(concurrent10) < 300:
        raise ValueError(f"全部同期重叠期只有{len(concurrent10)}个三通道并发10min点，无法稳定拟合。")
    if len(fit_train) < 50:
        raise ValueError(f"选择{resolution}拟合后只有{len(fit_train)}个有效训练时间步，无法稳定拟合。")

    custom_limited = bool((concurrent_start and str(concurrent_start).strip()) or (concurrent_end and str(concurrent_end).strip()))
    label = scope_label + (" + 手动日期限制" if custom_limited else "")
    progress(
        f"MCP训练期（{label}）：{concurrent10['target_local_time'].min():%Y-%m-%d %H:%M} "
        f"至 {concurrent10['target_local_time'].max():%Y-%m-%d %H:%M}；"
        f"八法三通道10min点 {len(concurrent10):,}；旧四通道10min点 {len(legacy_strict10):,}；新增有效速度样本 {len(common10_extra):,}；"
        f"旧先10min并发再聚合1h点 {len(legacy_concurrent1h):,}；旧1h四通道并发点 {len(legacy_allchannel1h):,}；"
        f"八法三通道1h点 {len(concurrent1h):,}；本次全局拟合分辨率={resolution}，八法共同训练样本={len(fit_train):,}"
    )

    full_ref = _full_reference_hourly(era, shift)
    eval_ref = full_ref[
        (full_ref["target_local_time"] >= eval_year.start)
        & (full_ref["target_local_time"] < eval_year.end_exclusive)
    ].copy().reset_index(drop=True)

    # V1.0.48: prediction resolution remains truly independent; WBL local threshold is 50 for 10min and experimental 24 for 1h.
    # V1.0.47: prediction resolution follows fitting resolution for real.
    # 10min mode predicts directly on the expanded 10min reference timeline;
    # 1h mode predicts on the native hourly reference timeline.  The 1h CSV is
    # only a presentation/summary product and is never used as the source of
    # the 10min Fit in 10min mode.
    eval_ref10_native = _expand_hourly_to_10min(
        eval_ref[["target_local_time", "ref_ws", "ref_dir"]].copy()
    )
    if not eval_ref10_native.empty:
        eval_ref10_native["target_local_time"] = pd.to_datetime(eval_ref10_native["Timestamp"])
    pred_frame = (
        eval_ref[["target_local_time", "ref_ws", "ref_dir"]].copy()
        if resolution == "1h"
        else eval_ref10_native[["target_local_time", "ref_ws", "ref_dir"]].copy()
    )

    models = {"FitResolution": resolution, "FitTrainingSampleCount": int(len(fit_train))}
    native_predictions = {}
    metrics_rows = []
    deterministic_validation_summary = pd.DataFrame()

    if bool(deterministic_sampling_enabled):
        sampling_label = "月内按天抽" if str(deterministic_sampling_mode).lower() == "day" else "月内按小时抽"
        progress(
            f"确定性方法重复抽样评价：BSR/LLS/TLS/VR/VS/WBL；全局{resolution}拟合；"
            f"固定50%训练+50%验证，{sampling_label}，共{max(1, int(deterministic_sampling_runs))}次；"
            f"仅BSR/LLS/TLS/VR在抽样训练阶段执行阈值={float(fit_min_speed):g}m/s；"
            f"VS/WBL不执行该阈值；SS/MTS不参与。"
        )
        deterministic_validation_summary = _deterministic_sampling_validation(
            fit_train,
            fit_min_speed=float(fit_min_speed),
            wbl_mode=wbl_mode,
            runs=max(1, int(deterministic_sampling_runs)),
            master_seed=int(deterministic_sampling_seed),
            sampling_mode=str(deterministic_sampling_mode),
            fit_resolution=resolution,
            wbl_concurrent_data=wbl_fit_train,
            progress=progress,
        )
    else:
        progress(f"确定性方法50%/50%重复抽样已关闭：最终模型直接使用全部{fit_label}有效样本（含低风速）拟合。")

    regression_min_local = 24 if resolution == "1h" else 50
    progress(f"拟合 BSR（{fit_label}；局部最小样本={regression_min_local}）…")
    models["BSR"] = fit_windographer_bsr(fit_train, fit_min_speed=0.0, minimum_local_samples=regression_min_local)
    models["BSR_MinLocalSamples"] = int(regression_min_local)
    native_predictions["BSR"] = models["BSR"].predict_frame(pred_frame, local_utc_offset_hours=0)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": "BSR", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(fit_train["target_ws"], models["BSR"].predict_frame(fit_train, local_utc_offset_hours=0), fit_label)})

    progress(f"拟合 LLS（{fit_label}；局部最小样本={regression_min_local}）…")
    models["LLS"] = fit_windographer_lls(fit_train, fit_min_speed=0.0, minimum_local_samples=regression_min_local)
    models["LLS_MinLocalSamples"] = int(regression_min_local)
    native_predictions["LLS"] = models["LLS"].predict_frame(pred_frame, local_utc_offset_hours=0)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": "LLS", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(fit_train["target_ws"], models["LLS"].predict_frame(fit_train, local_utc_offset_hours=0), fit_label)})

    progress(f"拟合 TLS（{fit_label}；最终阈值=0；局部最小样本={regression_min_local}）…")
    models["TLS"] = fit_windographer_tls(fit_train, fit_min_speed=0.0, minimum_local_samples=regression_min_local)
    models["TLS_MinLocalSamples"] = int(regression_min_local)
    models["TLS_FitResolution"] = resolution
    models["TLS_TrainingSampleCount"] = int(len(fit_train))
    native_predictions["TLS"] = models["TLS"].predict_frame(pred_frame, local_utc_offset_hours=0)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": "TLS", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(fit_train["target_ws"], models["TLS"].predict_frame(fit_train, local_utc_offset_hours=0), fit_label)})

    progress(f"拟合 VR（{fit_label}；局部最小样本={regression_min_local}）…")
    models["VR"] = fit_windographer_vr(fit_train, fit_min_speed=0.0, minimum_local_samples=regression_min_local)
    models["VR_MinLocalSamples"] = int(regression_min_local)
    native_predictions["VR"] = models["VR"].predict_frame(pred_frame, local_utc_offset_hours=0)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": "VR", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(fit_train["target_ws"], models["VR"].predict_frame(fit_train, local_utc_offset_hours=0), fit_label)})

    progress(f"拟合 VS（{fit_label}；局部最小样本=50）…")
    models["VS"] = fit_windographer_vs(fit_train)
    native_predictions["VS"] = models["VS"].predict_frame(pred_frame, local_utc_offset_hours=0)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": "VS", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(fit_train["target_ws"], models["VS"].predict_frame(fit_train, local_utc_offset_hours=0), fit_label)})

    # SpeedSort follows the same globally selected fitting resolution.
    ss_run_count = max(1, int(stochastic_realizations))
    if ss_run_count > 1 and ss_run_count % 2 == 0:
        ss_run_count += 1
        progress(f"SpeedSort分支投票要求奇数Seed，自动调整为 {ss_run_count} 个实现…")
    ss_seeds = _realization_seeds(int(ss_random_seed), ss_run_count)
    progress(f"拟合 SpeedSort-WG4.2.25（{fit_label}；{ss_run_count} Seed分支稳定）…")

    ss_seeded_models: list[tuple[int, object]] = []
    ss_rows: list[dict] = []
    for i, seed in enumerate(ss_seeds, start=1):
        progress(f"SpeedSort分支实现 {i}/{ss_run_count}，seed={seed}")
        ss_model_i, _ = fit_deterministic_speedsort(
            fit_train, full_ref[["target_local_time", "ref_ws", "ref_dir"]], random_seed=int(seed)
        )
        ss_seeded_models.append((int(seed), ss_model_i))
        ss_train_pred_i = ss_model_i.predict_frame(fit_train)
        ss_eval_pred_i = ss_model_i.predict_frame(pred_frame)
        ss_rows.append(_realization_row("SpeedSort", i, seed, fit_train["target_ws"], ss_train_pred_i, ss_eval_pred_i, fit_label))

    models["SpeedSort"] = build_branch_consensus_speedsort(ss_seeded_models, master_seed=int(ss_random_seed))
    models["SS_SelectedSeed"] = int(ss_random_seed)
    models["SS_BranchVoteSeedCount"] = int(ss_run_count)
    native_predictions["SpeedSort"] = models["SpeedSort"].predict_frame(pred_frame)

    ss_summary = pd.DataFrame(ss_rows)
    if len(ss_summary):
        ss_summary["是否主Seed"] = np.where(
            pd.to_numeric(ss_summary["Seed"], errors="coerce").eq(int(ss_random_seed)), "是", "否"
        )
        ss_summary["说明"] = f"{resolution}全样本单Seed实现仅用于稳定性与分支投票；最终QxS参数来自获胜分支中的真实Seed，不平均预测"
        models["SS_RealizationSummary"] = ss_summary

    ss_metric = {"方法": "SpeedSort", "评价模式": f"{resolution}全样本{ss_run_count}Seed分支投票", **_metrics(fit_train["target_ws"], models["SpeedSort"].predict_frame(fit_train), fit_label)}
    ss_metric.update({"稳定性Seed次数": ss_run_count, "主Seed": int(ss_random_seed), "最终模型": "QxS分支多数投票+真实Seed代表模型"})
    metrics_rows.append(ss_metric)

    wbl_min_local = 24 if resolution == "1h" else 50
    progress(f"拟合 WBL-Windographer 4.2.25/Openwind兼容模式（{wbl_fit_label}；局部最小样本={wbl_min_local}）…")
    models["WBL_WGCompat"] = fit_windographer_wbl(wbl_fit_train, minimum_local_samples=wbl_min_local)
    models["WBL_MinLocalSamples"] = int(wbl_min_local)
    models["WBL_TrainingSampleCount"] = int(len(wbl_fit_train))
    models["WBL_FitData"] = wbl_fit_train
    progress(f"拟合 WBL-Adaptive（{wbl_fit_label}；内部多Weibull策略）…")
    models["WBL_Adaptive"] = fit_adaptive_wbl(wbl_fit_train)
    mode_key = str(wbl_mode or "windographer_compat").strip().lower()
    if mode_key in {"windographer", "compat", "windographer_compat", "wg"}:
        models["WBL"] = models["WBL_WGCompat"]
        selected_wbl_label = "Windographer 4.2.25/Openwind兼容"
    else:
        models["WBL"] = models["WBL_Adaptive"]
        selected_wbl_label = "WBL-Adaptive增强"
    progress(f"WBL主结果模式：{selected_wbl_label}")
    native_predictions["WBL"] = models["WBL"].predict_frame(pred_frame)
    if not bool(deterministic_sampling_enabled):
        metrics_rows.append({"方法": f"WBL[{selected_wbl_label}]", "评价模式": f"全样本直接拟合[{resolution}]", **_metrics(wbl_fit_train["target_ws"], models["WBL"].predict_frame(wbl_fit_train), wbl_fit_label)})

    # MTS also follows the same globally selected fitting resolution.
    mts_run_count = max(3, int(stochastic_realizations))
    mts_seeds = _realization_seeds(int(mts_random_seed), mts_run_count)
    progress(
        f"拟合 MTS主体一次（原生{resolution}；移动平均={int(mts_moving_average_hours)}h；"
        f"边界={mts_edge_mode}），随后生成{mts_run_count}个随机实现…"
    )

    # Always keep a 10-minute evaluation reference for the optional 07 output.
    ref10 = _expand_hourly_to_10min(eval_ref)
    ref10 = ref10.rename(columns={"Timestamp": "timestamp"})[["timestamp", "ref_ws", "ref_dir"]]

    if resolution == "1h":
        mts_ref_native = eval_ref.rename(columns={"target_local_time": "timestamp"})[["timestamp", "ref_ws", "ref_dir"]].copy()
    else:
        mts_ref_native = ref10.copy()
    mts_train_native = fit_train.rename(columns={"target_local_time": "timestamp"})[["timestamp", "target_ws", "ref_ws", "ref_dir"]].copy()

    options = MTSOptions(
        direction_sectors=16, reference_bin_size=1.0, target_bin_size=1.0,
        moving_average_hours=int(mts_moving_average_hours), min_ref_bin_points=10, seasonality_min_points=50,
        markov_states=25, edge_mode=str(mts_edge_mode), edge_candidates=48,
        edge_match_tolerance_states=float(mts_edge_tolerance_states),
        edge_rejection_max_attempts=int(mts_edge_max_attempts),
        random_seed=int(mts_random_seed), time_step_minutes=fit_step_minutes,
    )
    mts = fit_mts(mts_train_native, options=options)
    mts_rows: list[dict] = []
    mts_train_ref = mts_train_native[["timestamp", "ref_ws", "ref_dir"]].copy()
    for i, seed in enumerate(mts_seeds, start=1):
        progress(f"MTS随机实现 {i}/{mts_run_count}，seed={seed}")
        mts_train_pred_i = mts.build_prediction_cache(mts_train_ref, seed=int(seed))
        mts_eval_pred_i = mts.build_prediction_cache(mts_ref_native, seed=int(seed))
        mts_rows.append(_realization_row("MTS", i, seed, mts_train_native["target_ws"], mts_train_pred_i, mts_eval_pred_i, fit_label))
    mts_summary, mts_selected_seed = _choose_representative_realization(mts_rows)
    progress(f"MTS最终代表性seed={mts_selected_seed}（按多指标中位状态距离选择，不挑最低误差）")
    mts.options.random_seed = int(mts_selected_seed)
    mts_prediction_native = mts.build_prediction_cache(mts_ref_native, seed=int(mts_selected_seed))
    mts_train_selected = mts.build_prediction_cache(mts_train_ref, seed=int(mts_selected_seed))
    models["MTS"] = mts
    models["MTS_RealizationSummary"] = mts_summary
    models["MTS_SelectedSeed"] = int(mts_selected_seed)
    mts_metric = {"方法": "MTS", "评价模式": f"{resolution}固定全样本，仅换Seed", **_metrics(mts_train_native["target_ws"], mts_train_selected, fit_label)}
    mts_metric.update({"随机实现次数": mts_run_count, "最终代表性Seed": int(mts_selected_seed)})
    metrics_rows.append(mts_metric)

    if bool(deterministic_sampling_enabled) and not deterministic_validation_summary.empty:
        dsum = deterministic_validation_summary.copy()
        dsum.loc[dsum["方法"].eq("WBL"), "方法"] = f"WBL[{selected_wbl_label}]"
        metrics_rows = dsum.to_dict("records") + metrics_rows
    models["DeterministicValidationSummary"] = deterministic_validation_summary

    # Normalize the seven non-MTS methods to a true 10-minute prediction table.
    # - 10min fitting: use the direct 10min model predictions.
    # - 1h fitting: hold the independently fitted hourly prediction across six slots.
    if resolution == "10min":
        standard10 = pred_frame.rename(columns={"target_local_time": "timestamp"}).copy()
        for name, pred in native_predictions.items():
            standard10[f"{name}_拟合风速(m/s)"] = np.asarray(pred, dtype=float)
    else:
        hp = pred_frame.copy()
        for name, pred in native_predictions.items():
            hp[f"{name}_拟合风速(m/s)"] = np.asarray(pred, dtype=float)
        standard10 = _expand_hourly_to_10min(hp).rename(columns={"Timestamp": "timestamp"})
        standard10["timestamp"] = pd.to_datetime(standard10["timestamp"])
    keep_cols = ["timestamp", "ref_ws", "ref_dir"] + [
        f"{name}_拟合风速(m/s)" for name in ("BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL")
    ]
    standard10 = standard10[keep_cols].drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

    # Build the 1h export strictly as a summary product.  In 10min fitting mode
    # it is the hourly mean of the direct 10min predictions; in 1h fitting mode
    # it is the native hourly prediction.
    hourly_out = eval_ref.copy()
    target_hour = _target_hourly_any_available(measured, eval_year.start, eval_year.end_exclusive)
    hourly_out = hourly_out.merge(target_hour, on="target_local_time", how="left")
    for name in ("BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL"):
        fit_col = f"{name}_拟合风速(m/s)"
        filled_col = f"{name}_补齐后风速(m/s)"
        if resolution == "1h":
            pred = np.asarray(native_predictions[name], dtype=float)
            hourly_out[fit_col] = pred
        else:
            hmean = (
                standard10.set_index("timestamp")[fit_col]
                .resample("1h", closed="left", label="left")
                .mean()
            )
            hourly_out[fit_col] = hourly_out["target_local_time"].map(hmean)
        hourly_out[filled_col] = hourly_out["target_ws"].where(
            hourly_out["target_ws"].notna(), hourly_out[fit_col]
        )

    # Normalize MTS output to the common 10-minute export grid. In 1h fitting
    # mode the hourly MTS prediction is simply held across the six 10-minute slots.
    if resolution == "1h":
        mts_hour_df = mts_ref_native.copy()
        mts_hour_df["MTS_拟合风速(m/s)"] = mts_prediction_native
        hp = mts_hour_df.rename(columns={"timestamp": "target_local_time"})[["target_local_time", "ref_ws", "ref_dir", "MTS_拟合风速(m/s)"]]
        exp = _expand_hourly_to_10min(hp).rename(columns={"Timestamp": "timestamp"})
        mts10 = exp[["timestamp", "ref_ws", "ref_dir", "MTS_拟合风速(m/s)"]].copy()
        hourly_out["MTS_拟合风速_10min均值到1h(m/s)"] = np.asarray(mts_prediction_native, dtype=float)
    else:
        mts10 = ref10.copy()
        mts10["MTS_拟合风速(m/s)"] = mts_prediction_native
        mts_hour = mts10.set_index("timestamp")["MTS_拟合风速(m/s)"].resample("1h", closed="left", label="left").mean()
        hourly_out["MTS_拟合风速_10min均值到1h(m/s)"] = hourly_out["target_local_time"].map(mts_hour)

    models["LegacyStrictConcurrent10"] = legacy_strict10
    models["LegacyConcurrent1h"] = legacy_concurrent1h
    models["LegacyAllChannelConcurrent1h"] = legacy_allchannel1h
    models["Common10_ExtraVsLegacy"] = common10_extra
    # Backward-compatible aliases retained for existing diagnostics/scripts.
    models["WBL10_ExtraConcurrency"] = common10_extra
    models["EngineVersion"] = ENGINE_VERSION
    return concurrent10, concurrent1h, full_ref, eval_ref, hourly_out, standard10, ref10, mts10, pd.DataFrame(metrics_rows), models


def _build_evaluation_10min_output(
    measured: pd.DataFrame, eval_year: EvalYear, standard10: pd.DataFrame, mts10: pd.DataFrame,
    measured_speed_col: str, era_speed_col: str,
) -> pd.DataFrame:
    """Build 07 from the native prediction timeline, never from the 06 hourly summary."""
    names = _export_column_names(measured_speed_col, era_speed_col)
    target_h = names["target_h"]
    idx = pd.date_range(eval_year.start, eval_year.end_inclusive, freq="10min")
    out = pd.DataFrame({"Timestamp": idx})
    measured10 = measured.copy()
    measured10["Timestamp"] = measured10["__time"].dt.round("10min")
    measured10 = measured10.drop_duplicates("Timestamp", keep="last")
    measured_map = measured10.set_index("Timestamp")["__speed"]
    out[names["measured"]] = out["Timestamp"].map(measured_map)

    pred10 = standard10.copy()
    pred10["timestamp"] = pd.to_datetime(pred10["timestamp"])
    pred10 = pred10.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    out[names["era_ws"]] = out["Timestamp"].map(pred10["ref_ws"])
    out[names["era_wd"]] = out["Timestamp"].map(pred10["ref_dir"])

    for method in ("BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL"):
        internal_pred = f"{method}_拟合风速(m/s)"
        fit_col = f"{method}_Fit_WS_{target_h}"
        filled_col = f"{method}_Filled_WS_{target_h}"
        out[fit_col] = out["Timestamp"].map(pred10[internal_pred])
        out[filled_col] = out[names["measured"]].where(out[names["measured"]].notna(), out[fit_col])
        observed_mask = out[names["measured"]].notna()
        if observed_mask.any():
            original_values = out.loc[observed_mask, names["measured"]].to_numpy(dtype=float)
            final_values = out.loc[observed_mask, filled_col].to_numpy(dtype=float)
            if not np.allclose(final_values, original_values, equal_nan=True, rtol=0.0, atol=0.0):
                raise AssertionError(f"{method} Filled修改了原始有效10min实测值，已阻止导出。")

    mts_map = mts10.set_index("timestamp")["MTS_拟合风速(m/s)"]
    mts_fit = f"MTS_Fit_WS_{target_h}"
    mts_filled = f"MTS_Filled_WS_{target_h}"
    out[mts_fit] = out["Timestamp"].map(mts_map)
    out[mts_filled] = out[names["measured"]].where(out[names["measured"]].notna(), out[mts_fit])
    observed_mask = out[names["measured"]].notna()
    if observed_mask.any():
        original_values = out.loc[observed_mask, names["measured"]].to_numpy(dtype=float)
        final_values = out.loc[observed_mask, mts_filled].to_numpy(dtype=float)
        if not np.allclose(final_values, original_values, equal_nan=True, rtol=0.0, atol=0.0):
            raise AssertionError("MTS Filled修改了原始有效10min实测值，已阻止导出。")
    return out


def _build_wbl_compare_10min(
    measured: pd.DataFrame, eval_year: EvalYear, eval_ref: pd.DataFrame, models: dict,
    measured_speed_col: str, era_speed_col: str, fit_resolution: str,
) -> pd.DataFrame:
    names = _export_column_names(measured_speed_col, era_speed_col)
    target_h = names["target_h"]
    idx = pd.date_range(eval_year.start, eval_year.end_inclusive, freq="10min")
    out = pd.DataFrame({"Timestamp": idx})
    measured10 = measured.copy()
    measured10["Timestamp"] = measured10["__time"].dt.round("10min")
    measured10 = measured10.drop_duplicates("Timestamp", keep="last")
    out[names["measured"]] = out["Timestamp"].map(measured10.set_index("Timestamp")["__speed"])

    base_h = eval_ref[["target_local_time", "ref_ws", "ref_dir"]].copy()
    resolution = str(fit_resolution or "10min").strip().lower()
    if resolution == "10min":
        pred_frame = _expand_hourly_to_10min(base_h)
        pred_frame["target_local_time"] = pd.to_datetime(pred_frame["Timestamp"])
        pred_frame = pred_frame[["target_local_time", "ref_ws", "ref_dir"]].copy()
        ref10 = pred_frame.rename(columns={"target_local_time": "Timestamp"}).set_index("Timestamp")
    else:
        pred_frame = base_h
        ref10 = _expand_hourly_to_10min(base_h).drop_duplicates("Timestamp", keep="last").set_index("Timestamp")
    out[names["era_ws"]] = out["Timestamp"].map(ref10["ref_ws"])
    out[names["era_wd"]] = out["Timestamp"].map(ref10["ref_dir"])

    for label, key in (("WBL_WGCompat", "WBL_WGCompat"), ("WBL_Adaptive", "WBL_Adaptive")):
        model = models.get(key)
        if model is None:
            continue
        pred = np.asarray(model.predict_frame(pred_frame), dtype=float)
        if resolution == "10min":
            pp = pred_frame[["target_local_time"]].copy()
            pp["pred"] = pred
            ep = pp.rename(columns={"target_local_time": "Timestamp"}).set_index("Timestamp")
        else:
            hp = pred_frame.copy()
            hp["pred"] = pred
            ep = _expand_hourly_to_10min(hp).drop_duplicates("Timestamp", keep="last").set_index("Timestamp")
        fit_col = f"{label}_Fit_WS_{target_h}"
        filled_col = f"{label}_Filled_WS_{target_h}"
        out[fit_col] = out["Timestamp"].map(ep["pred"])
        out[filled_col] = out[names["measured"]].where(out[names["measured"]].notna(), out[fit_col])
    return out


def _build_hourly_export(hourly_out: pd.DataFrame, measured_speed_col: str, era_speed_col: str) -> pd.DataFrame:
    """Return a presentation-only 1h table with ASCII headers; model internals stay unchanged."""
    names = _export_column_names(measured_speed_col, era_speed_col)
    target_h = names["target_h"]
    rename = {
        "ERA原始时刻": "ERA5_Raw_Timestamp",
        "target_local_time": "Timestamp",
        "ref_ws": names["era_ws"],
        "ref_dir": names["era_wd"],
        "target_ws": names["measured"],
    }
    for method in ("BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL"):
        rename[f"{method}_拟合风速(m/s)"] = f"{method}_Fit_WS_{target_h}"
        rename[f"{method}_补齐后风速(m/s)"] = f"{method}_Filled_WS_{target_h}"
    rename["MTS_拟合风速_10min均值到1h(m/s)"] = f"MTS_Fit_WS_{target_h}"
    export = hourly_out.rename(columns=rename).copy()

    # V1.0.42b: the user-facing 1h time axis must be the same aligned local
    # Timestamp used by the 10min output.  Keep the pre-shift ERA timestamp
    # only as a trailing diagnostic column so charting/import software cannot
    # accidentally pick it as the primary time axis.
    ordered = []
    if "Timestamp" in export.columns:
        ordered.append("Timestamp")
    ordered.extend(c for c in export.columns if c not in {"Timestamp", "ERA5_Raw_Timestamp"})
    if "ERA5_Raw_Timestamp" in export.columns:
        ordered.append("ERA5_Raw_Timestamp")
    return export.loc[:, ordered]

def run(config: RunConfig, progress: Optional[Callable[[str], None]] = None) -> dict:
    progress = progress or (lambda _msg: None)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress("读取测风数据…")
    measured_raw = read_table(config.measured_path)
    progress("读取ERA数据…")
    era_raw = read_table(config.era_path)

    measured_time_col, measured_time_fixed = resolve_time_column(measured_raw, config.measured_time_col, "测风数据")
    era_time_col, era_time_fixed = resolve_time_column(era_raw, config.era_time_col, "ERA数据")
    if measured_time_fixed:
        progress(f"[时间列自动纠正] UI选择的测风时间列‘{config.measured_time_col}’不是有效日期列，已自动改为‘{measured_time_col}’。")
    if era_time_fixed:
        progress(f"[时间列自动纠正] UI选择的ERA时间列‘{config.era_time_col}’不是有效日期列，已自动改为‘{era_time_col}’。")
    progress(f"实际使用列：测风时间={measured_time_col}；目标风速={config.measured_speed_col}；目标风向={config.measured_direction_col}；ERA时间={era_time_col}；ERA风速={config.era_speed_col}；ERA风向={config.era_direction_col}")

    if config.measured_speed_col not in measured_raw.columns:
        raise ValueError(f"测风目标风速列不存在：{config.measured_speed_col}")
    if config.measured_direction_col not in measured_raw.columns:
        raise ValueError(f"测风目标风向列不存在：{config.measured_direction_col}")
    for col, label in [(config.era_speed_col, "ERA风速列"), (config.era_direction_col, "ERA风向列")]:
        if col not in era_raw.columns:
            raise ValueError(f"{label}不存在：{col}")
    if measured_time_col in {config.measured_speed_col, config.measured_direction_col}:
        raise ValueError("测风时间列不能与目标风速/风向列相同。")
    if era_time_col in {config.era_speed_col, config.era_direction_col}:
        raise ValueError("ERA时间列不能与ERA风速/风向列相同。")

    measured = _prepare_measured(measured_raw, measured_time_col, config.measured_speed_col, config.measured_direction_col)
    era = _prepare_era(era_raw, era_time_col, config.era_speed_col, config.era_direction_col)
    if measured.empty:
        raise ValueError("测风时间列解析后为空。请检查时间列。")
    if era.empty:
        raise ValueError("ERA时间列解析后为空。请检查ERA时间列。")

    progress(f"测风解析范围：{measured['__time'].min():%Y-%m-%d %H:%M} 至 {measured['__time'].max():%Y-%m-%d %H:%M}；目标风速有效点 {int(measured['__speed'].notna().sum()):,}；目标风向有效点 {int(measured['__dir'].notna().sum()):,}")
    progress(f"ERA解析范围：{era['__time'].min():%Y-%m-%d %H:%M} 至 {era['__time'].max():%Y-%m-%d %H:%M}；有效小时 {len(era):,}")
    progress("选择边界为连续12个完整自然月、且ERA可覆盖的最高覆盖率评价年…")
    eval_year = select_evaluation_year(measured, era, config.shift_min, config.shift_max)
    progress(f"评价年：{eval_year.start:%Y-%m-%d} 至 {eval_year.end_inclusive:%Y-%m-%d}，覆盖率 {eval_year.coverage:.2f}%")
    progress("搜索ERA时间平移（R²最高）…")
    shift, shift_diag = find_best_shift(measured, era, eval_year, config.shift_min, config.shift_max)
    best_r2 = float(shift_diag.loc[shift_diag["ERA时间平移(h)"].eq(shift), "R²"].iloc[0])
    progress(f"最佳ERA时间平移：{shift:+d} h，R²={best_r2:.4f}")

    concurrent10, concurrent1h, full_ref, eval_ref, hourly_out, standard10, ref10, mts10, metrics, models = _fit_all(
        measured, era, eval_year, shift, config.fit_min_speed, progress,
        concurrent_start=config.concurrent_start,
        concurrent_end=config.concurrent_end,
        ss_random_seed=config.ss_random_seed,
        wbl_mode=config.wbl_mode,
        mts_moving_average_hours=config.mts_moving_average_hours,
        mts_random_seed=config.mts_random_seed,
        mts_edge_tolerance_states=config.mts_edge_tolerance_states,
        mts_edge_max_attempts=config.mts_edge_max_attempts,
        mts_edge_mode=config.mts_edge_mode,
        deterministic_sampling_enabled=config.deterministic_sampling_enabled,
        deterministic_sampling_runs=config.deterministic_sampling_runs,
        deterministic_sampling_seed=config.deterministic_sampling_seed,
        deterministic_sampling_mode=config.deterministic_sampling_mode,
        stochastic_realizations=config.stochastic_realizations,
        fit_resolution=config.fit_resolution,
        training_scope=config.training_scope,
    )

    progress("写出CSV结果…")
    legacy_strict10 = models.get("LegacyStrictConcurrent10")
    common10_extra = models.get("Common10_ExtraVsLegacy")
    legacy_allchannel1h = models.get("LegacyAllChannelConcurrent1h")
    _audit_lines = [
        f"EngineVersion={models.get('EngineVersion', ENGINE_VERSION)}",
        f"FitResolution={config.fit_resolution}",
        f"AllMethods10minTrainingCount={len(concurrent10)}",
        f"Legacy4Channel10minCount={len(legacy_strict10) if isinstance(legacy_strict10, pd.DataFrame) else -1}",
        f"AllMethods10minExtraVsLegacy={len(common10_extra) if isinstance(common10_extra, pd.DataFrame) else -1}",
        f"AllMethods1hTrainingCount={len(concurrent1h)}",
        f"Legacy4Channel1hCount={len(legacy_allchannel1h) if isinstance(legacy_allchannel1h, pd.DataFrame) else -1}",
        f"WBLFitCount={int(models.get('WBL_TrainingSampleCount', -1))}",
        "V1.0.59正式规则：BSR/LLS/TLS/VR/VS/SpeedSort/WBL/MTS在所选拟合分辨率下，共用Target_WS + Reference_WS + Reference_WD训练并发池；Target_WD不参与样本有效性判定。",
        "WBL的1h/10min已验证逻辑保持不变；本版仅把其训练样本规则推广到其他七法。",
        "ERA时间平移R²搜索仍保持原严格同期逻辑，不在本版修改。",
    ]
    _audit_text = "\n".join(_audit_lines)
    (out / "00_运行版本与训练样本自检.txt").write_text(_audit_text, encoding="utf-8-sig")
    # Backward-compatible filename retained so existing user workflow still finds it.
    (out / "00_运行版本与WBL训练样本自检.txt").write_text(_audit_text, encoding="utf-8-sig")
    eval_year.candidates.to_csv(out / "01_评价年候选窗口_12个完整自然月.csv", index=False, encoding="utf-8-sig")
    eval_year.monthly.to_csv(out / "02_逐月完整率_评价年标记.csv", index=False, encoding="utf-8-sig")
    shift_diag.to_csv(out / "03_ERA时间平移_R2诊断.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "04_八种MCP方法_拟合诊断指标.csv", index=False, encoding="utf-8-sig")
    det_summary = models.get("DeterministicValidationSummary")
    if isinstance(det_summary, pd.DataFrame) and not det_summary.empty:
        det_summary.to_csv(out / "04A_确定性方法_50比50重复抽样汇总.csv", index=False, encoding="utf-8-sig")
    # V1.0.56 production training pools: all eight methods use these same
    # dependency-correct three-channel concurrent samples.
    concurrent10.to_csv(out / "05A_全同期三通道并发10min训练数据.csv", index=False, encoding="utf-8-sig")
    # Keep the old 05A filename as a compatibility alias; "严格" now means
    # strict across the channels actually required by the MCP speed models.
    concurrent10.to_csv(out / "05A_全同期严格并发10min训练数据.csv", index=False, encoding="utf-8-sig")

    legacy_strict10 = models.get("LegacyStrictConcurrent10")
    if isinstance(legacy_strict10, pd.DataFrame):
        legacy_strict10.to_csv(out / "05A0_旧四通道严格并发10min_仅对照.csv", index=False, encoding="utf-8-sig")
    legacy_concurrent1h = models.get("LegacyConcurrent1h")
    if isinstance(legacy_concurrent1h, pd.DataFrame):
        legacy_concurrent1h.to_csv(out / "05C_旧方式_先10min四通道并发再聚合1h_对照.csv", index=False, encoding="utf-8-sig")
    legacy_allchannel1h = models.get("LegacyAllChannelConcurrent1h")
    if isinstance(legacy_allchannel1h, pd.DataFrame):
        legacy_allchannel1h.to_csv(out / "05D_旧方式_先各自聚合1h再四通道并发_对照.csv", index=False, encoding="utf-8-sig")

    concurrent1h.to_csv(out / "05_全同期三通道并发1h训练数据.csv", index=False, encoding="utf-8-sig")
    concurrent1h.to_csv(out / "05_全同期并发1h训练数据.csv", index=False, encoding="utf-8-sig")

    selected_fit_data = concurrent1h if str(config.fit_resolution).lower() == "1h" else concurrent10
    selected_fit_data.to_csv(
        out / f"05B_本次八法实际拟合训练数据_{str(config.fit_resolution).lower()}.csv",
        index=False, encoding="utf-8-sig"
    )
    # WBL compatibility copy: from V1.0.56 onward this is intentionally the
    # same training pool as 05B, because every method follows the WBL rule.
    selected_fit_data.to_csv(
        out / f"05E_WBL实际拟合训练数据_{str(config.fit_resolution).lower()}_不要求Target风向.csv",
        index=False, encoding="utf-8-sig"
    )
    common10_extra = models.get("Common10_ExtraVsLegacy")
    if isinstance(common10_extra, pd.DataFrame):
        common10_extra.to_csv(out / "05F_八法10min相对旧四通道新增的有效训练点.csv", index=False, encoding="utf-8-sig")

    dep_rows = []
    for method in ("BSR", "LLS", "TLS", "VR", "VS", "SpeedSort", "WBL", "MTS"):
        dep_rows.append({
            "方法": method,
            "拟合分辨率": str(config.fit_resolution),
            "必需Target通道": "Target_WS",
            "必需Reference通道": "Reference_WS + Reference_WD",
            "Target_WD参与并发": "否",
            "实际训练样本数": int(len(selected_fit_data)),
            "说明": "V1.0.59：八法仍统一WBL并发规则；BSR/LLS/TLS/VR局部MIN按分辨率10min=50、1h=24",
        })
    pd.DataFrame(dep_rows).to_csv(out / "05G_八法训练通道依赖与样本数.csv", index=False, encoding="utf-8-sig")
    hourly_export = _build_hourly_export(hourly_out, config.measured_speed_col, config.era_speed_col)
    hourly_export.to_csv(out / "06_八种MCP方法_评价年拟合结果_1h.csv", index=False, encoding="utf-8-sig")
    tls_model = models.get("TLS")
    if tls_model is not None and hasattr(tls_model, "diagnostics"):
        tls_diag = tls_model.diagnostics()
        tls_diag.insert(0, "拟合分辨率", str(models.get("FitResolution", config.fit_resolution)))
        tls_diag.to_csv(out / "22_TLS模型诊断.csv", index=False, encoding="utf-8-sig")
    ss_model = models.get("SpeedSort")
    if ss_model is not None and hasattr(ss_model, "diagnostics"):
        ss_model.diagnostics().to_csv(out / "08_SpeedSort模型诊断.csv", index=False, encoding="utf-8-sig")
    ss_runs = models.get("SS_RealizationSummary")
    if isinstance(ss_runs, pd.DataFrame) and not ss_runs.empty:
        ss_runs.to_csv(out / "19_SS随机实现汇总.csv", index=False, encoding="utf-8-sig")
    wbl_compat = models.get("WBL_WGCompat")
    if wbl_compat is not None and hasattr(wbl_compat, "diagnostics"):
        wbl_diag = wbl_compat.diagnostics().copy()
        wbl_diag.insert(0, "拟合分辨率", str(config.fit_resolution))
        wbl_diag.insert(1, "MIN_LOCAL_SAMPLES", int(models.get("WBL_MinLocalSamples", 24 if str(config.fit_resolution).lower() == "1h" else 50)))
        wbl_diag.to_csv(out / "14_WBL_Windographer兼容模型诊断.csv", index=False, encoding="utf-8-sig")
    wbl_adaptive = models.get("WBL_Adaptive")
    if wbl_adaptive is not None:
        if hasattr(wbl_adaptive, "candidate_diagnostics"):
            wbl_adaptive.candidate_diagnostics().to_csv(out / "15_WBL_Adaptive候选评分.csv", index=False, encoding="utf-8-sig")
        if hasattr(wbl_adaptive, "selection_diagnostics"):
            wbl_adaptive.selection_diagnostics().to_csv(out / "16_WBL_Adaptive最终选择.csv", index=False, encoding="utf-8-sig")
        compare_rows = []
        fit_diag = models.get("WBL_FitData")
        if not isinstance(fit_diag, pd.DataFrame) or fit_diag.empty:
            fit_diag = concurrent1h if str(config.fit_resolution).lower() == "1h" else concurrent10
        fit_diag_label = "1h仅Target风速+Reference风速/风向并发" if str(config.fit_resolution).lower() == "1h" else "10min仅Target风速+Reference风速/风向并发"
        for label, model in (("Windographer4.2.25_Openwind兼容", wbl_compat), ("Adaptive", wbl_adaptive)):
            if model is not None:
                compare_rows.append({"WBL模式": label, "拟合分辨率": str(config.fit_resolution), **_metrics(fit_diag["target_ws"], model.predict_frame(fit_diag), fit_diag_label)})
        pd.DataFrame(compare_rows).to_csv(out / "17_WBL双模式同期拟合指标.csv", index=False, encoding="utf-8-sig")
    mts_runs = models.get("MTS_RealizationSummary")
    if isinstance(mts_runs, pd.DataFrame) and not mts_runs.empty:
        mts_runs.to_csv(out / "20_MTS随机实现汇总.csv", index=False, encoding="utf-8-sig")
    seed_rows = [
        {"方法": "SpeedSort", "随机主种子": int(config.ss_random_seed), "稳定性实现次数": int(models.get("SS_BranchVoteSeedCount", config.stochastic_realizations)), "最终模型Seed": "QxS多数分支投票（参数来自真实获胜Seed）"},
        {"方法": "MTS", "随机主种子": int(config.mts_random_seed), "随机实现次数": int(config.stochastic_realizations), "最终代表性Seed": int(models.get("MTS_SelectedSeed", config.mts_random_seed))},
    ]
    pd.DataFrame(seed_rows).to_csv(out / "21_SS_MTS最终代表性随机种子.csv", index=False, encoding="utf-8-sig")

    mts_model = models.get("MTS")
    if mts_model is not None:
        mts_diag = {
            "name": getattr(mts_model, "name", "MTS"),
            "params": getattr(mts_model, "params", {}),
            "note": getattr(mts_model, "note", ""),
        }
        (out / "09_MTS模型参数诊断.json").write_text(
            json.dumps(mts_diag, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raw_p = getattr(mts_model, "raw_percentile_train", None)
        if raw_p is not None:
            raw_p.to_csv(out / "10_MTS_RawPercentile.csv", index=False, encoding="utf-8-sig")
        sm_p = getattr(mts_model, "smoothed_percentile_train", None)
        if sm_p is not None:
            sm_p.to_csv(out / "11_MTS_SmoothedPercentile.csv", index=False, encoding="utf-8-sig")
        sq = getattr(mts_model, "seasonality_q_train", None)
        if sq is not None:
            sq.to_csv(out / "12_MTS_SeasonalityNormalizedQ.csv", index=False, encoding="utf-8-sig")
        trans = getattr(mts_model, "transition", None)
        if trans is not None:
            pd.DataFrame(trans).to_csv(out / "13_MTS_MarkovTransitionMatrix_25x25.csv", index=True, encoding="utf-8-sig")
        # V1.0.42: MTS keeps Quarter × Sector × ReferenceBin JPD; Markov gap filling follows official right-edge rejection behaviour.
        # Export both a compact long table and per-Q×S Windographer-like matrices so future WG exports
        # can be checked cell-by-cell without reverse engineering the model again.
        jpd_long_fn = getattr(mts_model, "jpd_long_table", None)
        if callable(jpd_long_fn):
            jpd_long = jpd_long_fn()
            if isinstance(jpd_long, pd.DataFrame) and not jpd_long.empty:
                jpd_long.to_csv(out / "14_MTS_JPD_QxS_Long.csv", index=False, encoding="utf-8-sig")
        jpd_matrix_fn = getattr(mts_model, "jpd_matrix", None)
        if callable(jpd_matrix_fn):
            jpd_dir = out / "14_MTS_JPD_QxS矩阵"
            jpd_dir.mkdir(parents=True, exist_ok=True)
            nsec = int(getattr(mts_model, "n_sectors", 16))
            for q in range(1, 5):
                for sec in range(1, nsec + 1):
                    mat = jpd_matrix_fn(q, sec)
                    if isinstance(mat, pd.DataFrame) and not mat.empty:
                        mat.to_csv(jpd_dir / f"MTS_JPD_Q{q}_S{sec:02d}.csv", index=True, encoding="utf-8-sig")

    ten_min_path = None
    if config.export_10min:
        progress("生成评价年10min拟合/补齐结果…")
        ten = _build_evaluation_10min_output(measured, eval_year, standard10, mts10, config.measured_speed_col, config.era_speed_col)
        ten_min_path = out / "07_八种MCP方法_评价年拟合与补齐结果_10min.csv"
        ten.to_csv(ten_min_path, index=False, encoding="utf-8-sig")
        wbl_compare = _build_wbl_compare_10min(measured, eval_year, eval_ref, models, config.measured_speed_col, config.era_speed_col, config.fit_resolution)
        wbl_compare.to_csv(out / "18_WBL双模式评价年拟合与补齐对比_10min.csv", index=False, encoding="utf-8-sig")

    summary = {
        "评价年起点": str(eval_year.start),
        "评价年终点": str(eval_year.end_inclusive),
        "评价年覆盖率(%)": eval_year.coverage,
        "评价年有效10min点数": eval_year.valid,
        "评价年理论10min点数": eval_year.expected,
        "最佳ERA时间平移(h)": shift,
        "最佳R²": best_r2,
        "BSR_LLS_TLS_VR抽样评价训练低风速阈值(m/s)": config.fit_min_speed,
        "全局拟合分辨率": config.fit_resolution,
        "训练数据范围": config.training_scope,
        "全局拟合训练样本数": int(models.get("FitTrainingSampleCount", 0)),
        "TLS训练样本数": int(models.get("TLS_TrainingSampleCount", 0)),
        "方法": METHODS,
        "目标高度标签": _export_column_names(config.measured_speed_col, config.era_speed_col)["target_h"],
        "目标风向列": config.measured_direction_col,
        "ERA高度标签": _export_column_names(config.measured_speed_col, config.era_speed_col)["era_h"],
        "结果列名规则": "纯英文ASCII；Measured/ERA5 + Fit/Filled + WS/WD + 高度",
        "MCP训练期起点": str(concurrent10["target_local_time"].min()),
        "MCP训练期终点": str(concurrent10["target_local_time"].max()),
        "MCP训练严格并发10min点数": int(len(concurrent10)),
        "本次八法实际拟合训练点数": int(models.get("FitTrainingSampleCount", 0)),
        "SpeedSort训练点数": int(models.get("FitTrainingSampleCount", 0)),
        "MCP训练期规则": f"训练范围由UI显式选择：all_valid=全部有效测风×ERA同期，evaluation_year=仅评价年有效同期；手动开始/结束仅进一步收窄所选范围。V1.0.59继续：八种方法在10min与1h都统一采用与已验证WBL相同的必要通道并发：Target风速+Reference风速+Reference风向；Target风向仅作元数据，不参与训练样本有效性判定。本次训练范围={config.training_scope}，全局拟合分辨率={config.fit_resolution}。",
        "MCP训练同期限制_开始": config.concurrent_start or "",
        "MCP训练同期限制_结束": config.concurrent_end or "",
        "范围说明": "本工具只执行MCP评价年拟合/补齐，不执行后续20年LTC长期订正",
        "普通7法10min规则": "10min拟合时：七个非MTS方法直接在评价年10min参考时间轴上预测，07直接使用该原生10min Fit；06仅把这些10min Fit按小时求均值用于展示。1h拟合时：七法在评价年1h参考时间轴上预测，再保持到6个10min点生成07。Filled/Final逐10min保留原始有效实测，仅在缺失10min位置使用对应分辨率的Fit。",
        "训练DCR规则": "ERA平移R²搜索保持原严格同期逻辑不变。正式MCP训练：10min直接要求Target WS+Reference WS+Reference WD同一处理时间步有效；1h先仅对Target WS独立按1h聚合且DCR>=50%，再与原生1h Reference WS/WD并发。Target WD缺测不再删除任何八法速度训练样本；旧四通道10min/1h集合仅作为诊断对照导出。",
        "确定性方法50比50抽样开关": bool(config.deterministic_sampling_enabled),
        "确定性方法50比50抽样次数": int(config.deterministic_sampling_runs),
        "确定性方法抽样主Seed": int(config.deterministic_sampling_seed),
        "确定性方法抽样方式": "月内按天抽" if str(config.deterministic_sampling_mode).lower() == "day" else "月内按小时抽",
        "确定性方法抽样规则": f"BSR/LLS/TLS/VR/VS/WBL参与50%训练+50%验证抽样评价，并统一在本次{config.fit_resolution}三通道训练集上抽样。可调低风速阈值仅作用于BSR/LLS/TLS/VR的抽样训练半样本；VS/WBL使用全部抽中训练样本。最终生产模型对全部方法重新使用完整{config.fit_resolution}共同训练集拟合，包含低于阈值的真实低风速。SS/MTS不参与50/50抽样。",
        "TLS规则": f"两参数Total Least Squares/Orthogonal Least Squares；4季度×16个Reference direction扇区；每格拟合y=m*x+b；局部有效样本n<50时回退GLOBAL。TLS不再拥有独立分辨率开关，统一服从全局拟合分辨率={config.fit_resolution}。最终生产拟合阈值固定0m/s；可调低风速阈值仅用于50/50抽样评价训练半样本。",
        "SpeedSort规则": f"Windographer 4.2.25兼容主算法保持不变，但训练数据统一服从全局拟合分辨率={config.fit_resolution}；Reference WS<1m/s按Seed随机分扇区；季度×16扇区独立排序；LOCAL Transition=min(4,QxS Reference均值/2)；GLOBAL Transition=min(4,长期Reference均值/2)；分支投票与九轮边界判据保持V1.0.42b逻辑。",
        "SpeedSort随机主种子": int(config.ss_random_seed),
        "SpeedSort稳定性实现次数": int(config.stochastic_realizations),
        "SpeedSort最终模型随机种子": "QxS多数分支投票；主Seed=" + str(int(config.ss_random_seed)),
        "SpeedSort低风速随机阈值(m/s)": 1.0,
        "SpeedSort训练中<1m/s随机分扇区点数": int(getattr(models.get("SpeedSort"), "low_speed_randomized_count", 0)),
        "SpeedSort长期参考中<1m/s随机分扇区点数": 0,
        "WBL主结果模式": str(config.wbl_mode),
        "WBL兼容规则": f"Windographer 4.2.25经验兼容主线；训练数据统一服从全局拟合分辨率={config.fit_resolution}；10min与1h分别独立重建训练样本并重新拟合；V1.0.56起WBL与其他七法共用同一必要通道并发池：Target风速+Reference风速/风向，不要求Target风向；4季度×16参考风向扇区；Weibull参数使用Openwind法（匹配平均风速与三阶矩/风功率密度代理）；10min局部Time Steps<50回退GLOBAL，1h局部Time Steps<24回退GLOBAL。",
        "WBL自适应规则": "WBL-Adaptive是单一增强模式；内部候选不作为独立MCP方法暴露。每个GLOBAL/季度/季度×16扇区均使用全部同期样本各拟合一次并按最终拟合误差+分布一致性评分（MAE/RMSE/Bias/WPD/P50/P90/P95/KS）；季度/局部模型只有在稳定且不劣于更宽层级回退时才采用；指数/尾部爆炸仅触发回退，不裁剪最终预测。",
        "MTS规则": f"MTS主体训练数据统一服从全局拟合分辨率={config.fit_resolution}；10min模式原生时间步=10min，1h模式原生时间步=60min。JPD仍按4季度×16扇区×Reference 1m/s Bin；稀疏bin与Markov Fill Gaps规则保持V1.0.42b逻辑。若勾选07且使用1h拟合，MTS小时预测保持到该小时6个10min点。",
        "MTS移动平均(h)": int(config.mts_moving_average_hours),
        "MTS移动平均时间步": int(getattr(models.get("MTS"), "params", {}).get("百分位移动平均(时间步)", 0)),
        "MTS随机主种子": int(config.mts_random_seed),
        "MTS随机实现次数": int(config.stochastic_realizations),
        "MTS最终代表性随机种子": int(models.get("MTS_SelectedSeed", config.mts_random_seed)),
        "MTS边界模式": str(config.mts_edge_mode),
        "MTS边界容差(状态数)": float(config.mts_edge_tolerance_states),
        "MTS最大重抽次数": int(config.mts_edge_max_attempts),
    }
    (out / "00_本次运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("完成。")
    return {"summary": summary, "metrics": metrics, "output_dir": str(out), "ten_min_path": str(ten_min_path) if ten_min_path else None}
