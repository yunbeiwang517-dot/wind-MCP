from __future__ import annotations

"""Windographer 4.2.25 SpeedSort compatibility implementation.

This module keeps only behavior supported by the published SpeedSort method plus
direct black-box tests against Windographer 4.2.25.  The exact 4.2.25 critical
sample lookup used to choose the full versus origin-constrained local regression
is not public; the implementation therefore uses an explicitly marked empirical
compatibility proxy constrained by the supplied boundary tests.

Confirmed/recovered 4.2.25 behavior used here:
1. Fit SpeedSort on strict concurrent samples; target and reference speeds are
   independently sorted within each yearly-division x reference-direction sector.
2. Concurrent reference speeds below 1 m/s are randomly allocated to direction
   sectors. The exact Windographer PRNG is not public, so a deterministic user seed
   is retained for reproducibility.
3. GLOBAL fallback dog-leg threshold remains min(4 m/s, long-term reference mean / 2).
   For an installed QxS LOCAL model, Windographer 4.2.25 Regression Parameters and
   controlled black-box tests show that Transition is QxS-specific:
   min(4 m/s, mean(reference speed in that QxS fitting subdivision) / 2). The mean
   is therefore evaluated after the <1 m/s random sector allocation used by the
   concurrent fitting sample.
4. Windographer 4.2.25 no longer applies the legacy 4.0.27 rule that forced a
   sector's visible cutoff to zero merely because one yearly division was empty.
5. A QxS subdivision with fewer than 24 concurrent samples does not install a
   local model and falls back to GLOBAL. Boundary tests showed N=23 -> GLOBAL and
   N=24 -> LOCAL for an otherwise perfect linear case.
6. The full high-speed line uses the King-Hurley modified OLR geometry (origin to
   high-speed centroid rotation, OLS in rotated coordinates, rotate back).
7. The sparse/local fallback is a zero-intercept orthogonal TLS fit using only the
   ranked points at/above the fit cutoff.
8. Full versus origin-constrained LOCAL selection uses the TOTAL QxS Time Steps
   against a correlation-dependent critical count. Direct 4.2.25 tests constrain
   the current compatibility envelope: r=1 has minimum 24; p4 N27 remains Origin
   while N35 is Full; p5 N80 remains Origin while N88 is Full. The exact UL lookup
   is unpublished, so the isolated critical function remains an empirical proxy.
9. Real-data regression tables show an additional sparse-calm regime: candidate
   QxS Transition below 1 m/s with N<50 falls back GLOBAL; for 50<=N<100 and
   sorted-pair correlation<0.90 the supplied 4.2.25 tables use Origin.
10. Because <1 m/s reference points are randomly allocated to sectors, sparse QxS
   branch identity can change by seed. The engine can stabilize GLOBAL/Origin/Full
   using an odd-number multi-seed majority vote and select one real seed realization
   nearest the winning branch median (no averaging of predictions).
11. Below a positive prediction cutoff, the dog-leg is the straight line from the
   origin to the high-speed line evaluated at the cutoff. Negative predictions are
   clipped to zero.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd


MIN_LOCAL_SAMPLES = 24
WG425_CRITICAL_BASE = 24.0
N_SECTORS = 16
LOW_SPEED_RANDOM_SECTOR_THRESHOLD = 1.0
DEFAULT_SS_RANDOM_SEED = 20260907


def sector_index(wind_direction: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pd.Series(wind_direction), errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = np.floor(((values[valid] % 360.0) + 11.25) / 22.5).astype(int) % N_SECTORS
    return result


def randomized_sector_index(
    reference_speed: pd.Series,
    wind_direction: pd.Series,
    timestamps: pd.Series,
    random_seed: int = DEFAULT_SS_RANDOM_SEED,
) -> np.ndarray:
    """Return reference-direction sectors with WG-style low-speed randomization.

    Repeated Windographer SpeedSort fits change QxS Time Steps only when
    reference speeds below 1 m/s are present. Removing those points makes the
    regression table deterministic. The original SpeedSort method also describes
    low reference speeds as randomly distributed among direction sectors.

    Windographer's exact legacy PRNG/seed is not exposed, so this implementation
    uses a stable timestamp+seed SplitMix64 mapping. It is uniform over 16 sectors
    and changes when the seed changes. This function is used only for the
    short-term concurrent fitting sample; long-term cutoff construction and final
    prediction use the actual reference direction.
    """
    ws = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
    sectors = sector_index(wind_direction)
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    low = np.isfinite(ws) & (ws < LOW_SPEED_RANDOM_SECTOR_THRESHOLD) & np.isfinite(sectors) & ts.notna().to_numpy()
    if not low.any():
        return sectors

    # Stable 64-bit SplitMix64. Avoid Python's salted hash and NumPy RNG stream
    # position dependence so filtering/reordering does not change a timestamp's
    # assigned sector for a fixed seed.
    tvals = ts.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
    x = tvals[low].astype(np.uint64, copy=False) ^ np.uint64(int(random_seed) & 0xFFFFFFFFFFFFFFFF)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    sectors[low] = (z & np.uint64(15)).astype(int)
    return sectors


def sector_label(sector: int) -> str:
    lo = ((int(sector) * 22.5) - 11.25) % 360.0
    hi = ((int(sector) * 22.5) + 11.25) % 360.0
    return f"{lo:.2f}°-{hi:.2f}°"


def quarter_label(quarter: int) -> str:
    return {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}.get(int(quarter), str(quarter))


def _modified_olr_origin_centroid_axis(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """King-Hurley modified OLR used by Windographer SpeedSort.

    The residual direction is perpendicular to the line from the origin to the
    data centroid, rather than perpendicular to the x-axis.  Equivalently:
      1) rotate the origin->centroid axis onto the horizontal axis;
      2) perform ordinary least squares with an intercept in the rotated frame;
      3) rotate the fitted line back.

    This reproduces the observed Windographer high-speed Intercept/Slope geometry; PCA
    total-least-squares does not.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("SpeedSort modified OLR requires at least two paired points")
    xm, ym = float(np.mean(x)), float(np.mean(y))
    if not (np.isfinite(xm) and np.isfinite(ym)):
        raise ValueError("SpeedSort modified OLR centroid is invalid")

    theta = float(np.arctan2(ym, xm))
    c, ss = float(np.cos(theta)), float(np.sin(theta))
    u = x * c + y * ss
    v = -x * ss + y * c
    uc = u - float(np.mean(u))
    vc = v - float(np.mean(v))
    denom = float(np.sum(uc * uc))
    if denom <= 1e-15:
        raise ValueError("SpeedSort modified OLR is singular in rotated coordinates")
    b_rot = float(np.sum(uc * vc) / denom)

    dx = c - b_rot * ss
    dy = ss + b_rot * c
    if abs(dx) <= 1e-15:
        raise ValueError("SpeedSort modified OLR produced a vertical line")
    slope = float(dy / dx)
    intercept = float(ym - slope * xm)
    return slope, intercept



def _tls_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Orthogonal TLS constrained through the origin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sxx = float(np.sum(x * x))
    syy = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    if abs(sxy) <= 1e-15:
        return 1.0
    disc = np.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)
    return float((syy - sxx + disc) / (2.0 * sxy))

def _sorted_pair_correlation(group: pd.DataFrame) -> float:
    """Pearson correlation after independent sorting/rank pairing."""
    try:
        x, y = _sorted_pairs(group)
    except ValueError:
        return np.nan
    if len(x) < 3 or float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _critical_sample_count_from_correlation(correlation: float) -> int:
    """Empirical Windographer 4.2.25 critical Time-Step proxy.

    The published method states that the critical value depends on correlation,
    but UL does not expose the exact 4.2.25 lookup. Black-box boundary tests used
    here constrain the envelope as follows (TOTAL QxS Time Steps, not high-N):

      * r=1: N=23 -> GLOBAL, N=24 -> Full LOCAL;
      * r≈0.969: N=27 -> Origin, N=35 -> Full;
      * r≈0.854: N=80 -> Origin, N=88/89 -> Full;
      * lower-correlation p8/p12 cases stay Origin at N=44/52.

    ``quality = 2*r²-1`` with exponent 1.6 is the smoothest isolated proxy found
    that satisfies all currently supplied boundaries simultaneously. It is a
    compatibility envelope, not claimed as UL's unpublished internal table.
    """
    if not np.isfinite(correlation):
        return 10**9
    r = min(abs(float(correlation)), 1.0)
    if r <= 1e-12:
        return 10**9
    r2 = r * r
    if r2 >= 1.0 - 1e-12:
        return MIN_LOCAL_SAMPLES
    quality = 2.0 * r2 - 1.0
    if quality <= 1e-12:
        return 10**9
    n = int(np.ceil(WG425_CRITICAL_BASE / (quality ** 1.6)))
    return max(MIN_LOCAL_SAMPLES, n)

def _r2_linear(group: pd.DataFrame, slope: float, intercept: float = 0.0) -> float:
    x = pd.to_numeric(group["ref_ws"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group["target_ws"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 2:
        return np.nan
    pred = float(slope) * x + float(intercept)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0.0:
        return np.nan
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _r2_original(group: pd.DataFrame, slope: float, intercept: float, cutoff: float, low_slope: float) -> float:
    """Windographer-like R² diagnostic: apply the fitted curve to unsorted pairs."""
    x = pd.to_numeric(group["ref_ws"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group["target_ws"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x, y = x[ok], y[ok]
    if len(y) < 2:
        return np.nan
    pred = np.where(x < float(cutoff), float(low_slope) * x, float(slope) * x + float(intercept))
    pred = np.maximum(pred, 0.0)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0.0:
        return np.nan
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


@dataclass(frozen=True)
class SpeedSortModel:
    mode: str
    slope: float
    intercept: float
    cutoff: float
    fit_cutoff: float
    low_slope: float
    sample_count: int
    high_sample_count: int
    r2: float

    def predict(self, reference_speed: pd.Series) -> np.ndarray:
        x = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        result = np.full(len(x), np.nan)
        valid = np.isfinite(x) & (x >= 0.0)
        cutoff = max(0.0, float(self.cutoff))
        if self.mode in {"centroid", "identity", "origin"}:
            result[valid] = self.slope * x[valid]
        elif cutoff > 1e-12:
            low = valid & (x < cutoff)
            high = valid & ~low
            result[low] = self.low_slope * x[low]
            result[high] = self.slope * x[high] + self.intercept
        else:
            result[valid] = self.slope * x[valid] + self.intercept
        result[valid] = np.maximum(result[valid], 0.0)
        return result


def _sorted_pairs(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(group["ref_ws"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group["target_ws"], errors="coerce").to_numpy(float)
    x = np.sort(x[np.isfinite(x) & (x >= 0.0)])
    y = np.sort(y[np.isfinite(y) & (y >= 0.0)])
    n = min(len(x), len(y))
    if n < 2:
        raise ValueError("SpeedSort有效样本少于2个")
    return x[:n], y[:n]


def _fit_full(group: pd.DataFrame, fit_cutoff: float, prediction_cutoff: float | None = None) -> SpeedSortModel:
    x, y = _sorted_pairs(group)
    fit_cutoff = max(0.0, float(fit_cutoff))
    prediction_cutoff = fit_cutoff if prediction_cutoff is None else max(0.0, float(prediction_cutoff))
    high = x >= fit_cutoff
    high_count = int(high.sum())
    if high_count < 2:
        raise ValueError("SpeedSort cutoff以上有效排序点少于2个")
    slope, intercept = _modified_olr_origin_centroid_axis(x[high], y[high])
    if fit_cutoff > 1e-12:
        y_cut = max(0.0, slope * fit_cutoff + intercept)
        low_slope = y_cut / fit_cutoff
    else:
        low_slope = max(0.0, slope)
    # Regression-table R² follows the hidden fitted dog-leg even when the
    # displayed/prediction cutoff is 0.000.
    r2 = _r2_original(group, slope, intercept, fit_cutoff, low_slope)
    return SpeedSortModel(
        mode="full",
        slope=float(slope),
        intercept=float(intercept),
        cutoff=float(prediction_cutoff),
        fit_cutoff=float(fit_cutoff),
        low_slope=float(low_slope),
        sample_count=int(len(x)),
        high_sample_count=high_count,
        r2=float(r2) if np.isfinite(r2) else np.nan,
    )


def _fit_origin_orthogonal(group: pd.DataFrame, fit_cutoff: float, prediction_cutoff: float | None = None) -> SpeedSortModel:
    """WG4.2.25 sparse LOCAL branch: zero-intercept orthogonal fit above cutoff."""
    x, y = _sorted_pairs(group)
    fit_cutoff = max(0.0, float(fit_cutoff))
    prediction_cutoff = fit_cutoff if prediction_cutoff is None else max(0.0, float(prediction_cutoff))
    high = x >= fit_cutoff
    high_count = int(np.sum(high))
    if high_count < 2:
        raise ValueError("SpeedSort origin branch cutoff以上有效排序点少于2个")
    # 4.2.25 black-box tests show the origin branch is fitted to the same
    # high-speed ranked portion used by the full branch, not to all ranked points.
    slope = _tls_through_origin(x[high], y[high])
    r2 = _r2_linear(group, slope, 0.0)
    return SpeedSortModel(
        mode="origin", slope=float(slope), intercept=0.0, cutoff=float(prediction_cutoff), fit_cutoff=float(fit_cutoff),
        low_slope=float(slope), sample_count=int(len(x)), high_sample_count=high_count,
        r2=float(r2) if np.isfinite(r2) else np.nan,
    )

def _fit_identity(group: pd.DataFrame, fit_cutoff: float, prediction_cutoff: float | None = None) -> SpeedSortModel:
    """Published final sparse fallback: 45-degree line through origin (y=x)."""
    x, _ = _sorted_pairs(group)
    fit_cutoff = max(0.0, float(fit_cutoff))
    prediction_cutoff = fit_cutoff if prediction_cutoff is None else max(0.0, float(prediction_cutoff))
    high_count = int(np.sum(x >= fit_cutoff))
    r2 = _r2_linear(group, 1.0, 0.0)
    return SpeedSortModel(
        mode="identity", slope=1.0, intercept=0.0, cutoff=float(prediction_cutoff), fit_cutoff=float(fit_cutoff),
        low_slope=1.0, sample_count=int(len(x)), high_sample_count=high_count,
        r2=float(r2) if np.isfinite(r2) else np.nan,
    )


@dataclass
class DeterministicSpeedSort:
    global_model: SpeedSortModel
    local_models: Dict[Tuple[int, int], SpeedSortModel]
    global_cutoff: float
    concurrent_reference_mean: float
    sector_cutoffs: Dict[int, float]
    sector_fit_cutoffs: Dict[int, float]
    sector_concurrent_means: Dict[int, float]
    group_counts: Dict[Tuple[int, int], int]
    group_low_speed_randomized_counts: Dict[Tuple[int, int], int]
    group_correlations: Dict[Tuple[int, int], float]
    group_critical_samples: Dict[Tuple[int, int], int]
    group_reference_means: Dict[Tuple[int, int], float]
    random_seed: int
    low_speed_randomized_count: int
    longterm_low_speed_randomized_count: int
    group_consensus_modes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    group_consensus_votes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    group_selected_seeds: Dict[Tuple[int, int], int] = field(default_factory=dict)
    consensus_seed_count: int = 1

    @property
    def longterm_reference_mean(self) -> float:
        """Backward-compatible attribute name used by older diagnostics."""
        return float(self.concurrent_reference_mean)

    @property
    def sector_longterm_means(self) -> Dict[int, float]:
        """Backward-compatible attribute name for long-term sector means."""
        return self.sector_concurrent_means

    @property
    def local_cutoffs(self) -> Dict[Tuple[int, int], float]:
        return {key: float(model.cutoff) for key, model in self.local_models.items()}

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        local_time = pd.to_datetime(work["target_local_time"], errors="coerce")
        quarters = local_time.dt.quarter.to_numpy()
        # Windographer randomizes <1 m/s sector membership only in the SHORT-TERM
        # concurrent fitting sample. Long-term prediction uses actual reference direction.
        sectors = sector_index(work["ref_dir"])
        prediction = self.global_model.predict(work["ref_ws"])
        for (quarter, sector), model in self.local_models.items():
            mask = (quarters == quarter) & (sectors == sector)
            if mask.any():
                prediction[mask] = model.predict(work.loc[mask, "ref_ws"])
        return prediction

    def diagnostics(self) -> pd.DataFrame:
        rows = [{
            "scope": "GLOBAL",
            "yearly_division": "All",
            "quarter": np.nan,
            "sector": np.nan,
            "sector_range": "All",
            "time_steps": int(self.global_model.sample_count),
            "high_speed_steps": int(self.global_model.high_sample_count),
            "cutoff_mps": float(self.global_model.cutoff),
            "fit_cutoff_mps": float(self.global_model.fit_cutoff),
            "reference_mean_for_cutoff_mps": float(self.concurrent_reference_mean),
            "ss_random_seed": int(self.random_seed),
            "low_ref_lt1_randomized_steps": int(self.low_speed_randomized_count),
            "longterm_low_ref_lt1_randomized_steps": int(self.longterm_low_speed_randomized_count),
            "correlation": np.nan,
            "critical_samples": np.nan,
            "intercept_mps": float(self.global_model.intercept),
            "slope": float(self.global_model.slope),
            "r2": float(self.global_model.r2) if np.isfinite(self.global_model.r2) else np.nan,
            "low_segment_slope_k": float(self.global_model.low_slope),
            "fallback": "",
            "cutoff_source": "LONGTERM_GLOBAL",
            "branch_consensus": "GLOBAL",
            "branch_votes": "",
            "selected_branch_seed": int(self.random_seed),
            "branch_vote_seed_count": int(self.consensus_seed_count),
        }]
        for sector in range(N_SECTORS):
            rows.append({
                "scope": "SECTOR_CUTOFF",
                "yearly_division": "All",
                "quarter": np.nan,
                "sector": int(sector),
                "sector_range": sector_label(sector),
                "time_steps": int(sum(v for (q, s), v in self.group_counts.items() if int(s) == sector)),
                "high_speed_steps": np.nan,
                "cutoff_mps": float(self.sector_cutoffs.get(sector, self.global_cutoff)),
                "fit_cutoff_mps": float(self.sector_fit_cutoffs.get(sector, self.global_cutoff)),
                "reference_mean_for_cutoff_mps": float(self.sector_concurrent_means.get(sector, self.concurrent_reference_mean)),
                "ss_random_seed": int(self.random_seed),
                "low_ref_lt1_randomized_steps": int(sum(v for (q, s), v in self.group_low_speed_randomized_counts.items() if int(s) == sector)),
                "correlation": np.nan,
                "critical_samples": np.nan,
                "intercept_mps": np.nan,
                "slope": np.nan,
                "r2": np.nan,
                "low_segment_slope_k": np.nan,
                "fallback": "",
                "cutoff_source": "LONGTERM_ORIGINAL_REF_DIR",
                "branch_consensus": "",
                "branch_votes": "",
                "selected_branch_seed": np.nan,
                "branch_vote_seed_count": int(self.consensus_seed_count),
            })
        for quarter in range(1, 5):
            for sector in range(N_SECTORS):
                key = (quarter, sector)
                model = self.local_models.get(key)
                n = int(self.group_counts.get(key, 0))
                if model is None:
                    gm = self.global_model
                    rows.append({
                        "scope": "QxS",
                        "yearly_division": quarter_label(quarter),
                        "quarter": int(quarter),
                        "sector": int(sector),
                        "sector_range": sector_label(sector),
                        "time_steps": n,
                        "high_speed_steps": 0 if n == 0 else np.nan,
                        "cutoff_mps": float(gm.cutoff),
                        "fit_cutoff_mps": float(gm.fit_cutoff),
                        "reference_mean_for_cutoff_mps": float(self.concurrent_reference_mean),
                        "ss_random_seed": int(self.random_seed),
                        "low_ref_lt1_randomized_steps": int(self.group_low_speed_randomized_counts.get(key, 0)),
                        "correlation": float(self.group_correlations.get(key, np.nan)),
                        "critical_samples": float(self.group_critical_samples.get(key, np.nan)),
                        "intercept_mps": float(gm.intercept),
                        "slope": float(gm.slope),
                        "r2": float(gm.r2) if np.isfinite(gm.r2) else np.nan,
                        "low_segment_slope_k": float(gm.low_slope),
                        "fallback": "GLOBAL",
                        "cutoff_source": "LONGTERM_GLOBAL",
                        "branch_consensus": self.group_consensus_modes.get(key, "GLOBAL"),
                        "branch_votes": self.group_consensus_votes.get(key, ""),
                        "selected_branch_seed": float(self.group_selected_seeds.get(key, self.random_seed)),
                        "branch_vote_seed_count": int(self.consensus_seed_count),
                    })
                else:
                    rows.append({
                        "scope": "QxS",
                        "yearly_division": quarter_label(quarter),
                        "quarter": int(quarter),
                        "sector": int(sector),
                        "sector_range": sector_label(sector),
                        "time_steps": int(model.sample_count),
                        "high_speed_steps": int(model.high_sample_count),
                        "cutoff_mps": float(model.cutoff),
                        "fit_cutoff_mps": float(model.fit_cutoff),
                        "reference_mean_for_cutoff_mps": float(self.group_reference_means.get(key, self.concurrent_reference_mean)),
                        "ss_random_seed": int(self.random_seed),
                        "low_ref_lt1_randomized_steps": int(self.group_low_speed_randomized_counts.get(key, 0)),
                        "correlation": float(self.group_correlations.get(key, np.nan)),
                        "critical_samples": float(self.group_critical_samples.get(key, np.nan)),
                        "intercept_mps": float(model.intercept),
                        "slope": float(model.slope),
                        "r2": float(model.r2) if np.isfinite(model.r2) else np.nan,
                        "low_segment_slope_k": float(model.low_slope),
                        "fallback": "ORIGIN_ORTHOGONAL" if model.mode == "origin" else "",
                        "cutoff_source": "QXS_CONCURRENT_RANDOMIZED_REF",
                        "branch_consensus": self.group_consensus_modes.get(key, "ORIGIN" if model.mode == "origin" else "FULL"),
                        "branch_votes": self.group_consensus_votes.get(key, ""),
                        "selected_branch_seed": float(self.group_selected_seeds.get(key, self.random_seed)),
                        "branch_vote_seed_count": int(self.consensus_seed_count),
                    })
        return pd.DataFrame(rows)


def fit_deterministic_speedsort(
    concurrent: pd.DataFrame,
    longterm_reference: pd.DataFrame | None = None,
    random_seed: int = DEFAULT_SS_RANDOM_SEED,
) -> tuple[DeterministicSpeedSort, pd.DataFrame]:
    """Fit global + Q4xS16 SpeedSort on strict 10-minute concurrent samples.

    ``longterm_reference`` is the shifted long-term reference series and is used
    for the GLOBAL fallback dog-leg threshold. Installed QxS local models use the
    concurrent subdivision mean reference speed / 2 (capped at 4 m/s), matching
    Windographer 4.2.25 Regression Parameters.
    """
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for column in ("target_ws", "ref_ws", "ref_dir"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"]
    )
    work = work[(work["target_ws"] >= 0.0) & (work["ref_ws"] >= 0.0)].copy()
    if len(work) < 2:
        raise ValueError("SpeedSort有效同期样本少于2个")
    work["quarter"] = work["target_local_time"].dt.quarter.astype(int)
    work["sector"] = randomized_sector_index(
        work["ref_ws"], work["ref_dir"], work["target_local_time"], int(random_seed)
    ).astype(int)
    work["low_ref_lt1_randomized"] = work["ref_ws"].lt(LOW_SPEED_RANDOM_SECTOR_THRESHOLD)

    concurrent_mean = float(work["ref_ws"].mean())

    # Long-term reference drives both the GLOBAL and sector dog-leg thresholds.
    # Sector means use original long-term reference direction; <1 m/s random
    # allocation is confined to the short-term concurrent fitting sample.
    lt = None
    if longterm_reference is not None and len(longterm_reference):
        lt = longterm_reference.copy()
        for c in ("ref_ws", "ref_dir"):
            if c in lt.columns:
                lt[c] = pd.to_numeric(lt[c], errors="coerce")
        if "ref_ws" in lt.columns and "ref_dir" in lt.columns:
            lt = lt.replace([np.inf, -np.inf], np.nan).dropna(subset=["ref_ws", "ref_dir"])
            lt = lt[lt["ref_ws"] >= 0.0].copy()
            if len(lt):
                # Long-term cutoff construction always uses the actual reference
                # direction, including reference speeds below 1 m/s.
                lt["sector"] = sector_index(lt["ref_dir"]).astype(int)
                lt["low_ref_lt1_randomized"] = False
            else:
                lt = None
        else:
            lt = None

    if lt is not None and len(lt):
        longterm_mean = float(lt["ref_ws"].mean())
    else:
        longterm_mean = concurrent_mean
    if not np.isfinite(longterm_mean) or longterm_mean < 0.0:
        raise ValueError("SpeedSort长期参考平均风速无效")

    global_cutoff = float(min(4.0, max(0.0, longterm_mean / 2.0)))
    global_model = _fit_full(work, global_cutoff)

    # Count strict concurrent samples first; QxS counts drive the 4.2.25
    # LOCAL/GLOBAL minimum-sample decision.
    group_counts: Dict[Tuple[int, int], int] = {
        (int(q), int(s)): int(len(g)) for (q, s), g in work.groupby(["quarter", "sector"], sort=True)
    }
    group_low_speed_randomized_counts: Dict[Tuple[int, int], int] = {
        (int(q), int(s)): int(g["low_ref_lt1_randomized"].sum())
        for (q, s), g in work.groupby(["quarter", "sector"], sort=True)
    }

    # Windographer 4.2.25 / published SpeedSort sector cutoffs are based on the LONG-TERM
    # reference distribution in each ORIGINAL reference-direction sector.
    # Do not randomize <1 m/s long-term points here; that randomization belongs
    # only to the short-term concurrent fitting sample.
    sector_concurrent_means: Dict[int, float] = {}  # kept name for API compatibility
    sector_cutoffs: Dict[int, float] = {}
    sector_fit_cutoffs: Dict[int, float] = {}
    for sector in range(N_SECTORS):
        if lt is not None and len(lt):
            lvals = pd.to_numeric(lt.loc[lt["sector"].eq(sector), "ref_ws"], errors="coerce")
            lvals = lvals[np.isfinite(lvals) & (lvals >= 0.0)]
        else:
            # Defensive fallback only when no long-term reference was supplied.
            original_sector = sector_index(work["ref_dir"])
            mask = np.isfinite(original_sector) & (original_sector.astype(float) == float(sector))
            lvals = pd.to_numeric(work.loc[mask, "ref_ws"], errors="coerce")
            lvals = lvals[np.isfinite(lvals) & (lvals >= 0.0)]
        mean_for_cutoff = float(lvals.mean()) if len(lvals) else longterm_mean
        if not np.isfinite(mean_for_cutoff) or mean_for_cutoff < 0.0:
            mean_for_cutoff = longterm_mean
        sector_concurrent_means[sector] = mean_for_cutoff
        fit_cutoff = float(min(4.0, max(0.0, mean_for_cutoff / 2.0)))
        sector_fit_cutoffs[sector] = fit_cutoff
        # Windographer 4.2.25 keeps the sector dog-leg cutoff even if another
        # yearly division in the same direction sector is empty. The legacy
        # 4.0.27 zero-cutoff propagation rule is intentionally not applied.
        prediction_cutoff = fit_cutoff
        sector_cutoffs[sector] = prediction_cutoff

    local_models: Dict[Tuple[int, int], SpeedSortModel] = {}
    group_correlations: Dict[Tuple[int, int], float] = {}
    group_critical_samples: Dict[Tuple[int, int], int] = {}
    group_reference_means: Dict[Tuple[int, int], float] = {}
    for (quarter, sector), group in work.groupby(["quarter", "sector"], sort=True):
        key = (int(quarter), int(sector))
        n = int(len(group))

        # WG4.2.25: Transition is subdivision-specific, not one fixed annual
        # value shared by all four quarters in a direction sector. Controlled
        # tests gave exactly mean(QxS reference)/2 (capped at 4 m/s), and the
        # supplied 79071 Regression Parameters independently confirm that the
        # same direction sector has different Transition values by quarter.
        qxs_ref = pd.to_numeric(group["ref_ws"], errors="coerce")
        qxs_ref = qxs_ref[np.isfinite(qxs_ref) & (qxs_ref >= 0.0)]
        qxs_mean = float(qxs_ref.mean()) if len(qxs_ref) else concurrent_mean
        if not np.isfinite(qxs_mean) or qxs_mean < 0.0:
            qxs_mean = concurrent_mean
        group_reference_means[key] = qxs_mean
        fit_cutoff = float(min(4.0, max(0.0, qxs_mean / 2.0)))
        prediction_cutoff = fit_cutoff

        corr = _sorted_pair_correlation(group)
        critical_n = _critical_sample_count_from_correlation(corr)
        group_correlations[key] = float(corr) if np.isfinite(corr) else np.nan
        group_critical_samples[key] = int(critical_n)

        # WG4.2.25 critical-value test is against TOTAL QxS Time Steps. The
        # cutoff-above count controls the actual regression geometry only; it is
        # not the published/observed critical-count operand. A degenerate high
        # segment can still make the full fit singular, in which case Origin is
        # the next valid local branch.
        sparse_calm_global = (fit_cutoff < LOW_SPEED_RANDOM_SECTOR_THRESHOLD) and (n < 50)
        if sparse_calm_global:
            # 79117 regression table: three sparse/calm cells with candidate
            # Transition <1 m/s and N<50 fall back to the GLOBAL line.
            continue

        # Real-table compatibility guard for the remaining sparse-calm regime.
        # 79117 Q1-S5 and Q3-S15 are Origin in WG4.2.25 despite seed-sensitive
        # counts/correlations; nearby Q3-S14 remains Full. Across all supplied
        # tables the separating pattern is: Transition<1, 50<=N<100, sorted
        # correlation<0.90 -> Origin. 79071 has no cells satisfying this guard.
        sparse_calm_origin = (
            fit_cutoff < LOW_SPEED_RANDOM_SECTOR_THRESHOLD
            and 50 <= n < 100
            and np.isfinite(corr)
            and abs(float(corr)) < 0.90
        )
        if sparse_calm_origin:
            local_models[key] = _fit_origin_orthogonal(group, fit_cutoff, prediction_cutoff)
            continue

        if n >= critical_n:
            try:
                local_models[key] = _fit_full(group, fit_cutoff, prediction_cutoff)
            except ValueError:
                if n >= MIN_LOCAL_SAMPLES:
                    local_models[key] = _fit_origin_orthogonal(group, fit_cutoff, prediction_cutoff)
        elif n >= MIN_LOCAL_SAMPLES:
            local_models[key] = _fit_origin_orthogonal(group, fit_cutoff, prediction_cutoff)
        # n < 24: do not install a local model; predict_frame/diagnostics fall
        # back to GLOBAL, matching the supplied Windographer 4.2.25 boundary tests.

    single_modes: Dict[Tuple[int, int], str] = {}
    single_votes: Dict[Tuple[int, int], str] = {}
    single_selected_seeds: Dict[Tuple[int, int], int] = {}
    for q in range(1, 5):
        for sec in range(N_SECTORS):
            key = (q, sec)
            lm = local_models.get(key)
            mode = "GLOBAL" if lm is None else ("ORIGIN" if lm.mode == "origin" else "FULL")
            single_modes[key] = mode
            single_votes[key] = f"{mode}=1"
            single_selected_seeds[key] = int(random_seed)

    model = DeterministicSpeedSort(
        global_model=global_model,
        local_models=local_models,
        global_cutoff=global_cutoff,
        concurrent_reference_mean=longterm_mean,
        sector_cutoffs=sector_cutoffs,
        sector_fit_cutoffs=sector_fit_cutoffs,
        sector_concurrent_means=sector_concurrent_means,
        group_counts=group_counts,
        group_low_speed_randomized_counts=group_low_speed_randomized_counts,
        group_correlations=group_correlations,
        group_critical_samples=group_critical_samples,
        group_reference_means=group_reference_means,
        random_seed=int(random_seed),
        low_speed_randomized_count=int(work["low_ref_lt1_randomized"].sum()),
        longterm_low_speed_randomized_count=int(lt["low_ref_lt1_randomized"].sum()) if lt is not None and "low_ref_lt1_randomized" in lt.columns else 0,
        group_consensus_modes=single_modes,
        group_consensus_votes=single_votes,
        group_selected_seeds=single_selected_seeds,
        consensus_seed_count=1,
    )
    return model, work

def _branch_name(model: DeterministicSpeedSort, key: Tuple[int, int]) -> str:
    local = model.local_models.get(key)
    if local is None:
        return "GLOBAL"
    return "ORIGIN" if local.mode == "origin" else "FULL"


def build_branch_consensus_speedsort(
    seeded_models: list[tuple[int, DeterministicSpeedSort]],
    master_seed: int,
) -> DeterministicSpeedSort:
    """Build a per-QxS branch-stable model from real seed realizations.

    Only the categorical branch (GLOBAL/ORIGIN/FULL) is voted. Parameters are
    never averaged: for each winning local branch one actual seed realization
    nearest the median (cutoff, slope, intercept) of winning candidates is
    selected. This keeps a physically coherent SpeedSort curve while preventing
    one unlucky <1 m/s random-sector allocation from flipping a sparse cell.
    """
    if not seeded_models:
        raise ValueError("SpeedSort consensus requires at least one seeded model")
    ordered = [(int(seed), model) for seed, model in seeded_models]
    master = next((m for s, m in ordered if s == int(master_seed)), ordered[0][1])
    if len(ordered) == 1:
        return master

    local_models: Dict[Tuple[int, int], SpeedSortModel] = {}
    counts = dict(master.group_counts)
    low_counts = dict(master.group_low_speed_randomized_counts)
    corrs = dict(master.group_correlations)
    crits = dict(master.group_critical_samples)
    means = dict(master.group_reference_means)
    consensus_modes: Dict[Tuple[int, int], str] = {}
    consensus_votes: Dict[Tuple[int, int], str] = {}
    selected_seeds: Dict[Tuple[int, int], int] = {}

    for q in range(1, 5):
        for sec in range(N_SECTORS):
            key = (q, sec)
            modes = [_branch_name(m, key) for _, m in ordered]
            vote_counts = {name: modes.count(name) for name in ("FULL", "ORIGIN", "GLOBAL")}
            max_votes = max(vote_counts.values())
            winners = [name for name in ("FULL", "ORIGIN", "GLOBAL") if vote_counts[name] == max_votes]
            master_mode = _branch_name(master, key)
            winner = master_mode if master_mode in winners else winners[0]
            consensus_modes[key] = winner
            consensus_votes[key] = ";".join(f"{name}={vote_counts[name]}" for name in ("FULL", "ORIGIN", "GLOBAL"))

            candidates: list[tuple[int, DeterministicSpeedSort, SpeedSortModel | None]] = []
            for seed, mdl in ordered:
                if _branch_name(mdl, key) == winner:
                    candidates.append((seed, mdl, mdl.local_models.get(key)))
            if not candidates:
                selected_seeds[key] = int(master_seed)
                continue

            if winner == "GLOBAL":
                chosen_seed, chosen_source, _ = candidates[0]
                selected_seeds[key] = int(chosen_seed)
            else:
                vec = np.array([[float(c[2].cutoff), float(c[2].slope), float(c[2].intercept)] for c in candidates], dtype=float)
                med = np.nanmedian(vec, axis=0)
                scale = np.nanmedian(np.abs(vec - med), axis=0)
                scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
                dist = np.nansum(((vec - med) / scale) ** 2, axis=1)
                idx = int(np.nanargmin(dist)) if np.isfinite(dist).any() else 0
                chosen_seed, chosen_source, chosen_local = candidates[idx]
                local_models[key] = chosen_local
                selected_seeds[key] = int(chosen_seed)

            counts[key] = int(chosen_source.group_counts.get(key, counts.get(key, 0)))
            low_counts[key] = int(chosen_source.group_low_speed_randomized_counts.get(key, low_counts.get(key, 0)))
            corrs[key] = float(chosen_source.group_correlations.get(key, corrs.get(key, np.nan)))
            crits[key] = int(chosen_source.group_critical_samples.get(key, crits.get(key, 10**9)))
            means[key] = float(chosen_source.group_reference_means.get(key, means.get(key, master.concurrent_reference_mean)))

    return DeterministicSpeedSort(
        global_model=master.global_model,
        local_models=local_models,
        global_cutoff=master.global_cutoff,
        concurrent_reference_mean=master.concurrent_reference_mean,
        sector_cutoffs=dict(master.sector_cutoffs),
        sector_fit_cutoffs=dict(master.sector_fit_cutoffs),
        sector_concurrent_means=dict(master.sector_concurrent_means),
        group_counts=counts,
        group_low_speed_randomized_counts=low_counts,
        group_correlations=corrs,
        group_critical_samples=crits,
        group_reference_means=means,
        random_seed=int(master_seed),
        low_speed_randomized_count=master.low_speed_randomized_count,
        longterm_low_speed_randomized_count=master.longterm_low_speed_randomized_count,
        group_consensus_modes=consensus_modes,
        group_consensus_votes=consensus_votes,
        group_selected_seeds=selected_seeds,
        consensus_seed_count=len(ordered),
    )

