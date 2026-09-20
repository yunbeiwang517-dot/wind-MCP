from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd


@dataclass
class MTSOptions:
    """Matrix Time Series 参数。

    公开的 Windographer MTS 结构：
    JPD -> percentile time series -> Markov reconstruction -> inverse JPD。
    Windographer 未公开所有内部离散化/随机细节，因此 Markov state 数、
    edge candidate 数和随机种子属于本实现的可复现工程参数。
    """
    # V0.9 定型为 Windographer 式 MTS，不再提供 EMTM/实验分支。
    direction_sectors: int = 16
    reference_bin_size: float = 1.0
    target_bin_size: float = 1.0
    moving_average_hours: int = 3
    min_ref_bin_points: int = 10
    seasonality_window_days: int = 7
    seasonality_min_points: int = 50
    markov_states: int = 25
    # V1.0.42: Windographer Fill Gaps / Markov reconstruction help explicitly
    # states that a complete random scenario is discarded and regenerated when
    # its right edge matches the first measured value after the gap poorly.
    # Therefore right_rejection is the WG-Compat default.  The old fixed-pool
    # best-of-N branch remains only for regression comparison.
    edge_mode: str = "right_rejection"
    edge_candidates: int = 48
    edge_match_tolerance_states: float = 1.5
    edge_rejection_max_attempts: int = 500
    random_seed: int = 20260831
    build_full_seasonality_cache: bool = True
    time_step_minutes: Optional[float] = None


class MTSModel:
    def __init__(self, train: pd.DataFrame, options: Optional[MTSOptions] = None):
        self.options = options or MTSOptions()
        self.name = "MTS（Windographer式）"
        self.group = "Windographer时序MCP"
        self.note = (
            "目标原生时间步4季度×16扇区JPD条件分布 + 百分位时间序列 + 365×24小时季节性CDF（50点、日期优先扩窗） + 25状态一阶Markov时序重建；"
            "WG-Compat默认按Fill Gaps帮助采用右端拒绝整条重抽；参考风速bin少于10点时回退当前季度×扇区LLS，整季度完全无同期样本时才使用跨季度同扇区JPD兼容回退。"
        )
        self.params: Dict[str, object] = {}
        self._prediction_cache: Optional[pd.Series] = None
        self._fit(train)

    # ---------- public ----------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        z = self._prepare_reference(df)
        if self._prediction_cache is not None:
            idx = pd.DatetimeIndex(z["timestamp"])
            cached = self._prediction_cache.reindex(idx)
            if cached.notna().all():
                return cached.to_numpy(dtype=float)
        return self._synthesize(z, seed=self.options.random_seed)

    def build_prediction_cache(self, reference: pd.DataFrame, *, seed: Optional[int] = None) -> np.ndarray:
        z = self._prepare_reference(reference)
        pred = self._synthesize(z, seed=self.options.random_seed if seed is None else int(seed))
        self._prediction_cache = pd.Series(pred, index=pd.DatetimeIndex(z["timestamp"]))
        return pred

    # ---------- fit ----------
    @staticmethod
    def _prepare_reference(df: pd.DataFrame) -> pd.DataFrame:
        z = df.copy()
        z["timestamp"] = pd.to_datetime(z["timestamp"], errors="coerce")
        z["ref_ws"] = pd.to_numeric(z["ref_ws"], errors="coerce")
        if "ref_dir" in z.columns:
            z["ref_dir"] = pd.to_numeric(z["ref_dir"], errors="coerce") % 360.0
        z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "ref_ws"])
        z = z[z["ref_ws"] >= 0].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        return z

    def _fit(self, train: pd.DataFrame):
        z = train.copy()
        z["timestamp"] = pd.to_datetime(z["timestamp"], errors="coerce")
        z["ref_ws"] = pd.to_numeric(z["ref_ws"], errors="coerce")
        z["target_ws"] = pd.to_numeric(z["target_ws"], errors="coerce")
        if "ref_dir" in z.columns:
            z["ref_dir"] = pd.to_numeric(z["ref_dir"], errors="coerce") % 360.0
        z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "ref_ws", "target_ws"])
        z = z[(z["ref_ws"] >= 0) & (z["target_ws"] >= 0)].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if len(z) < 200:
            raise RuntimeError(f"MTS失败：全同期有效原生时间步仅{len(z)}，至少需要200个时间步。")
        self.train = z
        ts_idx = pd.DatetimeIndex(z["timestamp"])
        diffs = pd.Series(ts_idx).diff().dropna().dt.total_seconds().div(60.0)
        inferred_step = float(diffs[diffs > 0].median()) if (diffs > 0).any() else 60.0
        configured_step = self.options.time_step_minutes
        self.step_minutes = float(configured_step) if configured_step is not None else inferred_step
        if not np.isfinite(self.step_minutes) or self.step_minutes <= 0:
            raise RuntimeError("MTS失败：无法识别目标原生时间步。")
        self.step_timedelta = pd.to_timedelta(self.step_minutes, unit="m")
        self.steps_per_day = max(1, int(round(1440.0 / self.step_minutes)))
        self.has_direction = "ref_dir" in z.columns and z["ref_dir"].notna().sum() >= 50 and int(self.options.direction_sectors) > 1
        self.n_sectors = max(1, int(self.options.direction_sectors if self.has_direction else 1))
        self.ref_bin = max(float(self.options.reference_bin_size), 0.1)
        self.tar_bin = max(float(self.options.target_bin_size), 0.1)
        self.min_points = max(3, int(self.options.min_ref_bin_points))

        z["_quarter"] = self._quarter(z["timestamp"])
        z["_sector"] = self._sector(z["ref_dir"] if "ref_dir" in z.columns else pd.Series(np.nan, index=z.index))
        z["_ref_bin"] = self._ref_bin_index(z["ref_ws"].to_numpy(float))

        self._build_lls(z)
        self._build_jpd(z)
        p_raw = self._build_percentiles(z)
        self.raw_percentile_train = pd.DataFrame({
            "timestamp": z["timestamp"].to_numpy(),
            "p_raw": p_raw,
        })
        p = self._smooth_percentiles(z["timestamp"], p_raw)
        self.smoothed_percentile_train = pd.DataFrame({
            "timestamp": z["timestamp"].to_numpy(),
            "p_smoothed": p,
        })
        self.percentile_train = pd.DataFrame({"timestamp": z["timestamp"].to_numpy(), "p": p})
        self.percentile_train = self.percentile_train.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "p"]).sort_values("timestamp").reset_index(drop=True)
        if len(self.percentile_train) < 150:
            raise RuntimeError("MTS失败：可用于Markov训练的百分位序列不足150个原生时间步。")

        self._build_seasonality_and_markov()
        self.params = {
            "季节分组": "4季度（Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec）",
            "方向扇区": self.n_sectors,
            "JPD条件维度": "Quarter × Sector × ReferenceBin",
            "参考风速bin(m/s)": self.ref_bin,
            "目标风速bin(m/s)": self.tar_bin,
            "原生时间步(min)": float(self.step_minutes),
            "Seasonality CDF": "365×24小时",
            "百分位移动平均(h)": int(self.options.moving_average_hours),
            "百分位移动平均(时间步)": self._moving_average_window_steps(),
            "百分位移动平均规则": "centered symmetric; even nominal step count is promoted to the next odd count (3h@10min = 19 points)",
            "稀疏回退阈值": self.min_points,
            "稀疏bin回退层级": "Quarter×Sector LLS；季度扇区不足20点再退到pooled Sector LLS",
            "整季度零样本回退": "pooled-across-quarter同Sector×ReferenceBin JPD；pooled<10再Sector LLS（真实WG导出反推）",
            "季度同期样本数": {f"Q{q+1}": int(self.quarter_counts.get(q, 0)) for q in range(4)},
            "稀疏Percentile平滑": "Raw NaN保持NaN，不允许moving-average补回Markov训练",
            "Markov状态数": int(self.options.markov_states),
            "Seasonality窗口(±天)": int(self.options.seasonality_window_days),
            "Seasonality最小样本": int(self.options.seasonality_min_points),
            "边缘匹配模式": str(self.options.edge_mode),
            "边缘匹配容差(状态数)": float(self.options.edge_match_tolerance_states),
            "右端拒绝最大重抽次数": int(self.options.edge_rejection_max_attempts),
            "同期样本数": int(len(z)),
            "JPD有效条件格": int(sum(1 for c in self.jpd.values() if c["count"] >= self.min_points)),
            "JPD稀疏条件格": int(sum(1 for c in self.jpd.values() if c["count"] < self.min_points)),
            "百分位样本数": int(len(self.percentile_train)),
        }

    @staticmethod
    def _quarter(timestamp) -> np.ndarray:
        """Calendar quarter index, 0..3: Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec."""
        idx = pd.DatetimeIndex(pd.to_datetime(timestamp, errors="coerce"))
        month = idx.month.to_numpy(dtype=int)
        q = (month - 1) // 3
        return np.clip(q, 0, 3).astype(int)

    def _sector(self, direction) -> np.ndarray:
        if self.n_sectors <= 1 or direction is None:
            n = len(direction) if direction is not None else len(self.train)
            return np.zeros(n, dtype=int)
        d = pd.to_numeric(direction, errors="coerce").to_numpy(float)
        width = 360.0 / self.n_sectors
        sec = np.floor(((d + width / 2.0) % 360.0) / width)
        sec[~np.isfinite(d)] = 0
        return sec.astype(int)

    def _ref_bin_index(self, ws: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(ws, dtype=float), 0.0)
        return np.floor(x / self.ref_bin).astype(int)

    def _target_bin_index(self, ws: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(ws, dtype=float), 0.0)
        return np.floor(x / self.tar_bin).astype(int)

    def _build_lls(self, z: pd.DataFrame):
        x_all = z["ref_ws"].to_numpy(float)
        y_all = z["target_ws"].to_numpy(float)
        self.global_lls = self._polyfit_safe(x_all, y_all)

        # Pooled direction-sector LLS: final fallback when a quarterly sector
        # itself has too little data, or when an entire yearly division is absent.
        self.sector_lls: Dict[int, Tuple[float, float]] = {}
        for sec in range(self.n_sectors):
            g = z[z["_sector"].eq(sec)]
            if len(g) >= 20:
                self.sector_lls[sec] = self._polyfit_safe(
                    g["ref_ws"].to_numpy(float), g["target_ws"].to_numpy(float), fallback=self.global_lls
                )
            else:
                self.sector_lls[sec] = self.global_lls

        # With yearly divisions enabled, the sparse-bin backup has to stay in
        # the same subdivision.  Otherwise Q1/Q2/Q3 are silently mixed back
        # together exactly where the JPD is sparse.
        self.quarter_sector_lls: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.quarter_counts: Dict[int, int] = {}
        for quarter in range(4):
            qg = z[z["_quarter"].eq(quarter)]
            self.quarter_counts[quarter] = int(len(qg))
            for sec in range(self.n_sectors):
                g = qg[qg["_sector"].eq(sec)]
                fallback = self.sector_lls.get(sec, self.global_lls)
                if len(g) >= 20:
                    self.quarter_sector_lls[(quarter, sec)] = self._polyfit_safe(
                        g["ref_ws"].to_numpy(float), g["target_ws"].to_numpy(float), fallback=fallback
                    )
                else:
                    self.quarter_sector_lls[(quarter, sec)] = fallback

    @staticmethod
    def _polyfit_safe(x: np.ndarray, y: np.ndarray, fallback: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or np.std(x) <= 1e-10:
            if fallback is not None:
                return fallback
            return 0.0, float(np.nanmean(y)) if len(y) else 0.0
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    def _build_jpd(self, z: pd.DataFrame):
        # Windographer MTS JPD with the user's selected 4 yearly divisions:
        # Quarter × direction sector × reference-speed bin.
        self.jpd: Dict[Tuple[int, int, int], dict] = {}
        # Compatibility fallback used ONLY if a whole yearly division contains
        # no concurrent samples at all.  It deliberately does not replace the
        # official <10-point sparse-bin LLS backup inside an existing quarter.
        self.pooled_jpd: Dict[Tuple[int, int], dict] = {}
        target_bins = self._target_bin_index(z["target_ws"].to_numpy(float))
        z = z.copy()
        if "_quarter" not in z.columns:
            z["_quarter"] = self._quarter(z["timestamp"])
        z["_target_bin"] = target_bins
        for (quarter, sec, rb), g in z.groupby(["_quarter", "_sector", "_ref_bin"], sort=False):
            counts = g["_target_bin"].value_counts().sort_index()
            bins = counts.index.to_numpy(dtype=int)
            cnt = counts.to_numpy(dtype=float)
            self.jpd[(int(quarter), int(sec), int(rb))] = {
                "count": int(len(g)),
                "bins": bins,
                "counts": cnt,
                "cdf": np.cumsum(cnt) / max(float(np.sum(cnt)), 1.0),
            }
        for (sec, rb), g in z.groupby(["_sector", "_ref_bin"], sort=False):
            counts = g["_target_bin"].value_counts().sort_index()
            bins = counts.index.to_numpy(dtype=int)
            cnt = counts.to_numpy(dtype=float)
            self.pooled_jpd[(int(sec), int(rb))] = {
                "count": int(len(g)),
                "bins": bins,
                "counts": cnt,
                "cdf": np.cumsum(cnt) / max(float(np.sum(cnt)), 1.0),
            }

    def _condition_cell(self, quarter: int, sec: int, rb: int) -> Optional[dict]:
        return self.jpd.get((int(quarter), int(sec), int(rb)))

    def _pooled_condition_cell(self, sec: int, rb: int) -> Optional[dict]:
        return self.pooled_jpd.get((int(sec), int(rb)))

    def jpd_long_table(self) -> pd.DataFrame:
        """Diagnostic long table for all Q×S×reference-bin JPD cells."""
        rows = []
        for (quarter, sec, rb), cell in sorted(self.jpd.items()):
            count_map = {int(b): float(c) for b, c in zip(cell["bins"], cell["counts"])}
            for tb, cnt in sorted(count_map.items()):
                rows.append({
                    "Quarter": int(quarter) + 1,
                    "Sector": int(sec) + 1,
                    "Sector_Center_deg": float((sec * 360.0 / self.n_sectors) % 360.0),
                    "Reference_Bin_Lower_mps": float(rb * self.ref_bin),
                    "Reference_Bin_Upper_mps": float((rb + 1) * self.ref_bin),
                    "Target_Bin_Lower_mps": float(tb * self.tar_bin),
                    "Target_Bin_Upper_mps": float((tb + 1) * self.tar_bin),
                    "Count": int(round(cnt)),
                    "Cell_Total": int(cell["count"]),
                    "Frequency_pct_of_Cell": float(100.0 * cnt / max(cell["count"], 1)),
                })
        return pd.DataFrame(rows)

    def jpd_matrix(self, quarter: int, sector: int) -> pd.DataFrame:
        """Return a Windographer-like frequency-% matrix for one 1-based quarter and sector."""
        q = int(quarter) - 1
        sidx = int(sector) - 1
        cells = {(rb): cell for (qq, ss, rb), cell in self.jpd.items() if qq == q and ss == sidx}
        if not cells:
            return pd.DataFrame()
        ref_bins = sorted(cells)
        target_bins = sorted({int(tb) for cell in cells.values() for tb in cell["bins"]})
        total = sum(int(cells[rb]["count"]) for rb in ref_bins)
        mat = np.zeros((len(ref_bins), len(target_bins)), dtype=float)
        for i, rb in enumerate(ref_bins):
            cell = cells[rb]
            lookup = {int(tb): float(cnt) for tb, cnt in zip(cell["bins"], cell["counts"])}
            for j, tb in enumerate(target_bins):
                mat[i, j] = 100.0 * lookup.get(tb, 0.0) / max(total, 1)
        idx = [f"{rb*self.ref_bin:g} to {(rb+1)*self.ref_bin:g}" for rb in ref_bins]
        cols = [f"{tb*self.tar_bin:g} to {(tb+1)*self.tar_bin:g}" for tb in target_bins]
        out = pd.DataFrame(mat, index=idx, columns=cols)
        out["All"] = out.sum(axis=1)
        all_row = out.sum(axis=0)
        all_row.name = "All"
        out = pd.concat([out, all_row.to_frame().T], axis=0)
        return out

    def _cdf_target(self, cell: dict, target_ws: float) -> float:
        if cell is None or cell["count"] < self.min_points or not np.isfinite(target_ws):
            return np.nan
        tb = max(int(math.floor(max(target_ws, 0.0) / self.tar_bin)), 0)
        bins = cell["bins"]
        counts = cell["counts"]
        total = float(np.sum(counts))
        if total <= 0:
            return np.nan
        before = float(np.sum(counts[bins < tb]))
        this = float(np.sum(counts[bins == tb]))
        frac = float((max(target_ws, 0.0) - tb * self.tar_bin) / self.tar_bin)
        if this <= 0:
            # 空目标bin：在最近的累计概率上落点。
            return float(np.clip(before / total, 0.001, 0.999))
        return float(np.clip((before + np.clip(frac, 0.0, 1.0) * this) / total, 0.001, 0.999))

    def _ppf_target(self, cell: dict, p: float) -> float:
        if cell is None or cell["count"] < self.min_points or not np.isfinite(p):
            return np.nan
        bins = cell["bins"]
        counts = cell["counts"]
        total = float(np.sum(counts))
        if total <= 0:
            return np.nan
        q = float(np.clip(p, 0.001, 0.999))
        cdf = np.cumsum(counts) / total
        j = int(np.searchsorted(cdf, q, side="left"))
        j = min(max(j, 0), len(bins) - 1)
        prev = float(cdf[j - 1]) if j > 0 else 0.0
        prob = float(counts[j] / total)
        frac = 0.5 if prob <= 1e-12 else float(np.clip((q - prev) / prob, 0.0, 1.0))
        return float(max(0.0, bins[j] * self.tar_bin + frac * self.tar_bin))

    def _ppf_target_array(self, cell: dict, p: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=float), 0.001, 0.999)
        bins = cell["bins"]
        counts = cell["counts"]
        cdf = cell["cdf"]
        total = max(float(cell["count"]), 1.0)
        j = np.searchsorted(cdf, q, side="left")
        j = np.clip(j, 0, len(bins) - 1)
        prev = np.where(j > 0, cdf[np.maximum(j - 1, 0)], 0.0)
        prob = counts[j] / total
        frac = np.where(prob > 1e-12, np.clip((q - prev) / prob, 0.0, 1.0), 0.5)
        return np.maximum(0.0, bins[j] * self.tar_bin + frac * self.tar_bin)

    @staticmethod
    def _empirical_ppf_presorted(vals: np.ndarray, q: float) -> float:
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return 0.5
        xp = (np.arange(len(vals), dtype=float) + 0.5) / len(vals)
        return float(np.interp(np.clip(q, 0.001, 0.999), xp, vals, left=vals[0], right=vals[-1]))

    def _build_percentiles(self, z: pd.DataFrame) -> np.ndarray:
        out = np.full(len(z), np.nan, dtype=float)
        quarter = z["_quarter"].to_numpy(int) if "_quarter" in z.columns else self._quarter(z["timestamp"])
        sec = z["_sector"].to_numpy(int)
        rb = z["_ref_bin"].to_numpy(int)
        y = z["target_ws"].to_numpy(float)
        # 对稀疏ref bin，Windographer正式预测会退回LLS。百分位序列本身没有可靠JPD，
        # 因此这里不把这些点硬塞进Markov训练，避免伪造概率结构。
        for i in range(len(z)):
            cell = self._condition_cell(quarter[i], sec[i], rb[i])
            if cell is not None and cell["count"] >= self.min_points:
                out[i] = self._cdf_target(cell, y[i])
        return out

    def _moving_average_window_steps(self) -> int:
        """Return a centered symmetric window length in native time steps.

        Windographer's 3 h example is centered on the current time step.  When
        the nominal number of native steps is even (3 h / 10 min = 18), a
        truly centered window must contain an odd number of points.  Cross-check
        against exported Windographer Percentiles data shows 19 points for 10
        minute data, so promote an even nominal count to the next odd count.
        """
        nominal = max(1, int(round(float(self.options.moving_average_hours) * 60.0 / self.step_minutes)))
        if nominal > 1 and nominal % 2 == 0:
            nominal += 1
        return nominal

    def _smooth_percentiles(self, timestamp: pd.Series, p: np.ndarray) -> np.ndarray:
        # Windographer式 centered moving average。3h@10min => 19 points
        # (current point + 9 before + 9 after), not 18.
        win = self._moving_average_window_steps()
        if win <= 1:
            return p
        s = pd.Series(p, index=pd.DatetimeIndex(timestamp))
        full = pd.date_range(s.index.min(), s.index.max(), freq=self.step_timedelta)
        s = s.reindex(full)
        sm = s.rolling(window=win, center=True, min_periods=max(1, int(math.ceil(win * 0.5)))).mean()
        out = sm.reindex(pd.DatetimeIndex(timestamp)).to_numpy(dtype=float, copy=True)
        # Official MTS uses LLS rather than synthesis when a reference-speed bin
        # has fewer than 10 concurrent points.  Do not let the moving average
        # manufacture a percentile at a time step whose raw JPD percentile is
        # undefined; otherwise sparse-bin points leak back into Markov training.
        out[~np.isfinite(np.asarray(p, dtype=float))] = np.nan
        return out

    # ---------- seasonality ----------
    def _doy_hour(self, ts: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
        """Windographer式 seasonality profile：365天 × 24小时。

        即使目标是10min，一个小时内的6个点也共享同一个“小时CDF”；
        Markov链本身仍按10min推进。闰年2月29日并入相邻的365日历索引。
        """
        idx = pd.DatetimeIndex(ts)
        doy = idx.dayofyear.to_numpy(dtype=int, copy=True)
        leap_after_feb = np.asarray(idx.is_leap_year, dtype=bool) & (np.asarray(idx.month, dtype=int) > 2)
        doy = doy - leap_after_feb.astype(int)
        # 2月29日映射到2月28日所在日序，避免创造第366个profile。
        feb29 = (np.asarray(idx.month, dtype=int) == 2) & (np.asarray(idx.day, dtype=int) == 29)
        doy[feb29] = 59
        doy = np.clip(doy, 1, 365)
        hour = idx.hour.to_numpy(dtype=int)
        return doy, hour

    def _prepare_hour_profiles(self):
        p = self.percentile_train
        self._hour_profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for hour in range(24):
            g = p[p["_hour"].eq(hour)]
            self._hour_profiles[hour] = (g["_doy"].to_numpy(dtype=int), g["p"].to_numpy(dtype=float))
        self._profile_doy = p["_doy"].to_numpy(dtype=int)
        self._profile_hour = p["_hour"].to_numpy(dtype=int)
        self._all_profile_values = p["p"].to_numpy(dtype=float)

    @staticmethod
    def _circular_hour_distance(hours: np.ndarray, target_hour: int) -> np.ndarray:
        d = np.abs(np.asarray(hours, dtype=int) - int(target_hour))
        return np.minimum(d, 24 - d)

    def _seasonal_values_uncached(self, td: int, thour: int) -> np.ndarray:
        """Build one Windographer-style seasonality CDF sample pool.

        Expansion order follows the published behaviour: begin with the same
        hour and a ±day window; if the pool has fewer than 50 valid values,
        expand the day-of-year dimension first until the full annual cycle is
        included; only then expand symmetrically into neighbouring hours.
        """
        if not hasattr(self, "_hour_profiles"):
            self._prepare_hour_profiles()
        min_n = max(10, int(self.options.seasonality_min_points))
        base_window = max(1, int(self.options.seasonality_window_days))

        cd, cp = self._hour_profiles.get(int(thour), (np.array([], dtype=int), np.array([], dtype=float)))
        if len(cp):
            dist = np.abs(cd - int(td))
            dist = np.minimum(dist, 365 - dist)
            # Date dimension first: ±7d, ±14d, ... until all calendar days.
            radius = base_window
            while True:
                vals = cp[dist <= radius]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= min_n or radius >= 182:
                    break
                radius = min(182, radius + base_window)
            if len(vals) >= min_n:
                return vals
        else:
            vals = np.array([], dtype=float)

        # Still sparse after using all dates at the same hour: only now expand
        # hour-of-day symmetrically, retaining all dates.  At radius 12 this
        # covers all 24 hours (circular distance).
        all_h = getattr(self, "_profile_hour", np.array([], dtype=int))
        all_p = getattr(self, "_all_profile_values", np.array([], dtype=float))
        if len(all_p):
            hdist = self._circular_hour_distance(all_h, int(thour))
            for hr_radius in range(1, 13):
                vals = all_p[hdist <= hr_radius]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= min_n:
                    return vals
            vals = all_p[np.isfinite(all_p)]
            if len(vals):
                return vals
        return np.array([0.5])

    def _build_seasonality_cache(self):
        self._seasonality_cache: Dict[Tuple[int, int], np.ndarray] = {}
        for hour in range(24):
            for td in range(1, 366):
                self._seasonality_cache[(td, hour)] = np.sort(self._seasonal_values_uncached(td, hour))

    def _seasonal_values(self, ts: pd.Timestamp) -> np.ndarray:
        target_doy, target_hour = self._doy_hour([pd.Timestamp(ts)])
        key = (int(target_doy[0]), int(target_hour[0]))
        cache = getattr(self, "_seasonality_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        vals = np.sort(self._seasonal_values_uncached(*key))
        if cache is not None:
            cache[key] = vals
        return vals

    @staticmethod
    def _empirical_cdf_presorted(vals: np.ndarray, x: float) -> float:
        # seasonality cache 已经排序；这里禁止重复 sort，MTS多Seed随机实现会快很多。
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0 or not np.isfinite(x):
            return np.nan
        left = int(np.searchsorted(vals, x, side="left"))
        right = int(np.searchsorted(vals, x, side="right"))
        rank = 0.5 * (left + right)
        return float(np.clip((rank + 0.5) / (len(vals) + 1.0), 0.001, 0.999))

    @staticmethod
    def _empirical_cdf(vals: np.ndarray, x: float) -> float:
        vals = np.sort(np.asarray(vals, dtype=float))
        vals = vals[np.isfinite(vals)]
        return MTSModel._empirical_cdf_presorted(vals, x)

    @staticmethod
    def _empirical_ppf(vals: np.ndarray, q: float) -> float:
        vals = np.sort(np.asarray(vals, dtype=float))
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.5
        xp = (np.arange(len(vals), dtype=float) + 0.5) / len(vals)
        return float(np.interp(np.clip(q, 0.001, 0.999), xp, vals, left=vals[0], right=vals[-1]))

    def _build_seasonality_and_markov(self):
        p = self.percentile_train.copy()
        doy, hour = self._doy_hour(p["timestamp"])
        p["_doy"] = doy
        p["_hour"] = hour
        self.percentile_train = p
        self._prepare_hour_profiles()
        self._seasonality_cache = {}
        if bool(self.options.build_full_seasonality_cache):
            self._build_seasonality_cache()
        q = np.full(len(p), np.nan, dtype=float)
        for i, row in p.iterrows():
            vals = self._seasonal_values(pd.Timestamp(row["timestamp"]))
            q[i] = self._empirical_cdf_presorted(vals, float(row["p"]))
        p["q"] = q
        self.seasonality_q_train = p[["timestamp", "p", "q"]].copy()
        p = p.dropna(subset=["q"]).reset_index(drop=True)
        self.percentile_train = p
        self.known_p = pd.Series(p["p"].to_numpy(float), index=pd.DatetimeIndex(p["timestamp"]))
        self.known_q = pd.Series(p["q"].to_numpy(float), index=pd.DatetimeIndex(p["timestamp"]))

        nstate = max(5, int(self.options.markov_states))
        self.nstate = nstate
        states = np.minimum((p["q"].to_numpy(float) * nstate).astype(int), nstate - 1)
        counts = np.zeros((nstate, nstate), dtype=float)
        ts = pd.DatetimeIndex(p["timestamp"])
        for i in range(len(p) - 1):
            if ts[i + 1] - ts[i] == self.step_timedelta:
                counts[states[i], states[i + 1]] += 1.0

        # Windographer式MTM：不向从未出现的转移人为加概率。
        # 某状态若在训练样本中完全没有后继，则保守地保持在自身状态。
        trans = np.zeros_like(counts)
        row_sum = counts.sum(axis=1)
        for st in range(nstate):
            if row_sum[st] > 0:
                trans[st] = counts[st] / row_sum[st]
            else:
                trans[st, st] = 1.0
        self.transition = trans
        self.transition_cdf = np.cumsum(trans, axis=1)
        self.transition_cdf[:, -1] = 1.0
        hist = np.bincount(states, minlength=nstate).astype(float)
        if hist.sum() <= 0:
            hist[:] = 1.0
        self.stationary = hist / hist.sum()
        self.state_values: Dict[int, np.ndarray] = {}
        for st in range(nstate):
            vals = p.loc[states == st, "q"].to_numpy(float)
            self.state_values[st] = vals if len(vals) else np.array([(st + 0.5) / nstate])

    # ---------- synthesis ----------
    def _state(self, q: float) -> int:
        return int(np.clip(math.floor(float(np.clip(q, 0.0, 0.999999)) * self.nstate), 0, self.nstate - 1))

    def _draw_q_for_state(self, state: int, rng: np.random.Generator) -> float:
        vals = self.state_values.get(int(state))
        if vals is None or len(vals) == 0:
            return float((state + rng.random()) / self.nstate)
        return float(vals[int(rng.integers(0, len(vals)))])

    def _draw_path(self, n: int, start_q: Optional[float], rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=float)
        if start_q is None or not np.isfinite(start_q):
            u0 = float(rng.random())
            state = int(np.searchsorted(np.cumsum(self.stationary), u0, side="right"))
            state = min(state, self.nstate - 1)
        else:
            state = self._state(float(start_q))
        out = np.empty(n, dtype=float)
        u = rng.random(n)
        # 预先生成每个状态内的随机取样索引，避免每一步调用np.random.choice。
        for i in range(n):
            state = int(np.searchsorted(self.transition_cdf[state], u[i], side="right"))
            if state >= self.nstate:
                state = self.nstate - 1
            vals = self.state_values.get(state)
            if vals is None or len(vals) == 0:
                out[i] = (state + 0.5) / self.nstate
            else:
                out[i] = vals[int(rng.integers(0, len(vals)))]
        return out

    def _fill_q_sequence(self, timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
        known = self.known_q.reindex(timestamps)
        # pandas 3 Copy-on-Write may return a read-only array here.  The MTS
        # Markov reconstruction intentionally fills missing percentile states
        # in place, so take an owned writable copy without changing values.
        q = known.to_numpy(dtype=float, copy=True)
        rng = np.random.default_rng(int(seed))
        missing = ~np.isfinite(q)
        if not missing.any():
            return q

        # 连续缺口逐段生成；有右边界时从若干Markov候选中挑选最能衔接右边界的路径。
        i = 0
        while i < len(q):
            if not missing[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(q) and missing[j + 1] and timestamps[j + 1] - timestamps[j] == self.step_timedelta:
                j += 1
            n = j - i + 1
            start_q = q[i - 1] if i > 0 and np.isfinite(q[i - 1]) and timestamps[i] - timestamps[i - 1] == self.step_timedelta else None
            end_q = q[j + 1] if j + 1 < len(q) and np.isfinite(q[j + 1]) and timestamps[j + 1] - timestamps[j] == self.step_timedelta else None
            if end_q is None:
                path = self._draw_path(n, start_q, rng)
            else:
                mode = str(getattr(self.options, "edge_mode", "right_rejection") or "right_rejection").strip().lower()
                next_state = self._state(float(end_q))
                tol = max(0.0, float(self.options.edge_match_tolerance_states)) / float(self.nstate)

                if mode == "legacy_best_of_n":
                    # V1.0.6 regression branch: fixed candidate pool with a
                    # numerical penalty on BOTH ends.  Kept only so different
                    # towers can be A/B checked without changing old results.
                    base_candidates = max(4, int(self.options.edge_candidates))
                    if n > self.steps_per_day * 90:
                        nc = min(base_candidates, 6)
                    elif n > self.steps_per_day * 31:
                        nc = min(base_candidates, 12)
                    elif n > self.steps_per_day * 7:
                        nc = min(base_candidates, 24)
                    else:
                        nc = base_candidates
                    best, best_score = None, np.inf
                    best_acceptable, best_acceptable_score = None, np.inf
                    prev_state = self._state(float(start_q)) if start_q is not None and np.isfinite(start_q) else None
                    for _ in range(nc):
                        cand = self._draw_path(n, start_q, rng)
                        first_state = self._state(float(cand[0]))
                        last_state = self._state(float(cand[-1]))
                        end_prob = max(float(self.transition[last_state, next_state]), 1e-12)
                        end_gap = abs(float(cand[-1]) - float(end_q))
                        if prev_state is None:
                            start_gap = 0.0
                            start_prob = 1.0
                        else:
                            start_gap = abs(float(cand[0]) - float(start_q))
                            start_prob = max(float(self.transition[prev_state, first_state]), 1e-12)
                        score = start_gap + end_gap - 0.01 * (math.log(start_prob) + math.log(end_prob))
                        if score < best_score:
                            best, best_score = cand, score
                        acceptable = end_gap <= tol and (prev_state is None or start_gap <= tol)
                        if acceptable and score < best_acceptable_score:
                            best_acceptable, best_acceptable_score = cand, score
                    path = best_acceptable if best_acceptable is not None else best
                else:
                    # V1.0.7 Windographer-candidate branch.  The left edge is
                    # already conditioned by starting the Markov chain from the
                    # last known state.  Do NOT add an extra start-value penalty.
                    # Draw a complete scenario, inspect ONLY the right edge, and
                    # reject/rebuild it until the gap joins the first known state
                    # after the gap within tolerance.  If no scenario is accepted
                    # within the safety cap, use the best right-edge candidate so
                    # the run always terminates deterministically.
                    max_attempts = max(1, int(getattr(self.options, "edge_rejection_max_attempts", 500)))
                    best, best_score = None, np.inf
                    accepted = None
                    for _ in range(max_attempts):
                        cand = self._draw_path(n, start_q, rng)
                        # Windographer's Fill Gaps help describes the acceptance
                        # test in terms of how well the LAST synthetic value
                        # matches the FIRST measured value after the gap.  The
                        # left edge is already conditioned by start_q.
                        end_gap = abs(float(cand[-1]) - float(end_q))
                        if end_gap < best_score:
                            best, best_score = cand, end_gap
                        if end_gap <= tol:
                            accepted = cand
                            break
                    path = accepted if accepted is not None else best
            q[i:j + 1] = path
            i = j + 1
        return np.clip(q, 0.001, 0.999)

    def _synthesize(self, ref: pd.DataFrame, seed: int) -> np.ndarray:
        if ref.empty:
            return np.array([], dtype=float)
        z = ref.sort_values("timestamp").reset_index().rename(columns={"index": "_orig_index"})
        ts = pd.DatetimeIndex(z["timestamp"])
        q = self._fill_q_sequence(ts, seed=int(seed))
        knownp = self.known_p.reindex(ts).to_numpy(dtype=float)
        p = knownp.copy()
        missp = ~np.isfinite(p)
        if missp.any():
            doy, hour = self._doy_hour(ts)
            for i in np.flatnonzero(missp):
                key = (int(doy[i]), int(hour[i]))
                vals = self._seasonality_cache.get(key)
                if vals is None:
                    vals = np.sort(self._seasonal_values_uncached(*key))
                    self._seasonality_cache[key] = vals
                p[i] = self._empirical_ppf_presorted(vals, q[i])

        quarter = self._quarter(z["timestamp"])
        sec = self._sector(z["ref_dir"] if "ref_dir" in z.columns else pd.Series(np.nan, index=z.index))
        rb = self._ref_bin_index(z["ref_ws"].to_numpy(float))
        x = z["ref_ws"].to_numpy(float)
        y = np.full(len(z), np.nan, dtype=float)

        # Batch inverse-JPD by (yearly division, direction sector, reference bin).
        # Two different sparse situations must NOT be conflated:
        #   A) The quarter exists, but this ref bin has <10 samples -> official LLS backup.
        #   B) The whole quarter has zero concurrent samples -> no quarterly JPD exists at all.
        #      In that case use a pooled-across-yearly-divisions JPD for the same
        #      direction/ref bin when it is sufficiently populated, preserving the
        #      stochastic MTS behaviour.  This fallback is an empirical WG-compat
        #      inference from real exports and is explicitly recorded in diagnostics.
        keys = np.column_stack([quarter, sec, rb])
        for key in np.unique(keys, axis=0):
            qidx, sidx, ridx = int(key[0]), int(key[1]), int(key[2])
            mask = (quarter == qidx) & (sec == sidx) & (rb == ridx)
            cell = self._condition_cell(qidx, sidx, ridx)
            quarter_n = int(self.quarter_counts.get(qidx, 0))
            if cell is not None and cell["count"] >= self.min_points:
                y[mask] = self._ppf_target_array(cell, p[mask])
            elif quarter_n <= 0:
                pooled = self._pooled_condition_cell(sidx, ridx)
                if pooled is not None and pooled["count"] >= self.min_points:
                    y[mask] = self._ppf_target_array(pooled, p[mask])
                else:
                    slope, intercept = self.sector_lls.get(sidx, self.global_lls)
                    y[mask] = np.maximum(0.0, slope * x[mask] + intercept)
            else:
                slope, intercept = self.quarter_sector_lls.get(
                    (qidx, sidx), self.sector_lls.get(sidx, self.global_lls)
                )
                y[mask] = np.maximum(0.0, slope * x[mask] + intercept)

        out = pd.Series(y, index=z["_orig_index"]).sort_index().to_numpy(float)
        return np.maximum(out, 0.0)


def fit_mts(train: pd.DataFrame, options: Optional[MTSOptions] = None) -> MTSModel:
    return MTSModel(train, options=options)
