from __future__ import annotations

"""Adaptive WBL v2.

This module intentionally exposes only ONE adaptive WBL mode to the GUI.
Internally it compares several Weibull estimators, but they are candidate
engines rather than separate user-facing MCP methods.

V1.0.22 changes:
- no repeated train/test resampling;
- every candidate is fitted once on ALL available concurrent samples in the
  scope being evaluated;
- candidate selection uses final-fit error + distribution fidelity;
- local/quarter models must beat a broader fallback or be demonstrably no worse;
- explosive tails are rejected by model selection, never clipped afterwards.

The final prediction remains the ordinary WBL power law
    target = scale * reference ** exponent
with scale = c_target / c_reference**exponent and
exponent = k_reference / k_target.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .weibull_fit import METHODS, fit_weibull
from .windographer_wbl import sector_index


MIN_QUARTER = 120
MIN_LOCAL = 60
FULL_LOCAL = 240
SMALL_LOCAL_IMPROVEMENT = 0.05
LARGE_LOCAL_TOLERANCE = 0.01
QUARTER_IMPROVEMENT = 0.03
QUARTER_LARGE_TOLERANCE = 0.02


def _quarter_label(q: int) -> str:
    return {1: "Jan - Mar", 2: "Apr - Jun", 3: "Jul - Sep", 4: "Oct - Dec"}[int(q)]


def _sector_label(s: int) -> str:
    lo = (float(s) * 22.5 - 11.25) % 360.0
    hi = (float(s) * 22.5 + 11.25) % 360.0
    return "348.75 - 11.25" if s == 0 else f"{lo:.2f} - {hi:.2f}"


def _ks_distance(a, b) -> float:
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    fx = np.searchsorted(x, grid, side="right") / len(x)
    fy = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def _fit_metrics(obs, pred) -> dict:
    y = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    keys = (
        "mae", "rmse", "bias", "nmae", "nrmse", "nbias", "wpd_bias",
        "p50_bias", "p90_bias", "p95_bias", "ks", "score",
    )
    if len(y) < 2:
        return {k: float("nan") for k in keys}
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    scale = max(float(np.mean(np.abs(y))), 1.0)
    nmae = mae / scale
    nrmse = rmse / scale
    nbias = abs(bias) / scale

    y3 = float(np.mean(np.maximum(y, 0.0) ** 3))
    p3 = float(np.mean(np.maximum(p, 0.0) ** 3))
    wpd_bias = abs(p3 - y3) / max(y3, 1.0)

    def qbias(q: float) -> float:
        yq = float(np.quantile(y, q))
        pq = float(np.quantile(p, q))
        return abs(pq - yq) / max(abs(yq), 1.0)

    p50_bias = qbias(0.50)
    p90_bias = qbias(0.90)
    p95_bias = qbias(0.95)
    ks = _ks_distance(y, p)

    # WBL is fundamentally a distribution mapping.  V1.0.22 therefore gives
    # more weight to WPD/tail/CDF agreement than the old CV-only score did.
    score = (
        0.20 * nmae
        + 0.15 * nrmse
        + 0.10 * nbias
        + 0.20 * wpd_bias
        + 0.05 * p50_bias
        + 0.10 * p90_bias
        + 0.10 * p95_bias
        + 0.10 * ks
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "nmae": nmae,
        "nrmse": nrmse,
        "nbias": nbias,
        "wpd_bias": wpd_bias,
        "p50_bias": p50_bias,
        "p90_bias": p90_bias,
        "p95_bias": p95_bias,
        "ks": ks,
        "score": float(score),
    }


@dataclass
class WBLPowerModel:
    method: str
    target_k: float
    target_c: float
    reference_k: float
    reference_c: float
    scale_a: float
    exponent_b: float
    training_n: int
    target_zero_count: int
    reference_zero_count: int

    def predict(self, reference_speed) -> np.ndarray:
        x = pd.to_numeric(pd.Series(reference_speed), errors="coerce").to_numpy(float)
        y = np.full(len(x), np.nan, dtype=float)
        ok = np.isfinite(x) & (x >= 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            y[ok] = self.scale_a * np.power(x[ok], self.exponent_b)
        y[~np.isfinite(y)] = np.nan
        return np.maximum(y, 0.0)


def _fit_power(group: pd.DataFrame, method: str) -> WBLPowerModel:
    work = group[["ref_ws", "target_ws"]].copy()
    work["ref_ws"] = pd.to_numeric(work["ref_ws"], errors="coerce")
    work["target_ws"] = pd.to_numeric(work["target_ws"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["ref_ws"] >= 0.0) & (work["target_ws"] >= 0.0)]
    if len(work) < 3:
        raise RuntimeError("Adaptive WBL失败：有效并发样本少于3个。")
    tf = fit_weibull(work["target_ws"].to_numpy(float), method)
    rf = fit_weibull(work["ref_ws"].to_numpy(float), method)
    b = float(rf.k / tf.k)
    a = float(tf.c / (rf.c ** b))
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        raise RuntimeError("Adaptive WBL失败：幂律参数无效。")
    return WBLPowerModel(
        method=method,
        target_k=float(tf.k),
        target_c=float(tf.c),
        reference_k=float(rf.k),
        reference_c=float(rf.c),
        scale_a=a,
        exponent_b=b,
        training_n=int(len(work)),
        target_zero_count=int((work["target_ws"] == 0).sum()),
        reference_zero_count=int((work["ref_ws"] == 0).sum()),
    )


def _stability(model: WBLPowerModel, group: pd.DataFrame, fitm: dict) -> tuple[bool, str]:
    b = float(model.exponent_b)
    if not np.isfinite(b) or b <= 0.15 or b >= 4.5:
        return False, "exponent_outside_adaptive_guard"

    x = pd.to_numeric(group["ref_ws"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group["target_ws"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x, y = x[ok], y[ok]
    if len(x) >= 10:
        p = model.predict(x)
        y99 = float(np.quantile(y, 0.99))
        p99 = float(np.nanquantile(p, 0.99))
        ymax = float(np.max(y))
        pmax = float(np.nanmax(p))
        if p99 > max(1.6 * max(y99, 0.1), y99 + 5.0):
            return False, "p99_tail_explosive"
        if pmax > max(2.0 * max(ymax, 0.1), ymax + 10.0):
            return False, "max_tail_explosive"

    if np.isfinite(fitm.get("wpd_bias", np.nan)) and fitm["wpd_bias"] > 0.80:
        return False, "wpd_bias_extreme"
    if np.isfinite(fitm.get("p95_bias", np.nan)) and fitm["p95_bias"] > 0.60:
        return False, "p95_bias_extreme"
    return True, "ok"


def _evaluate_candidate(group: pd.DataFrame, method: str) -> dict:
    model = _fit_power(group, method)
    pred = model.predict(group["ref_ws"])
    fitm = _fit_metrics(group["target_ws"], pred)
    stable, reason = _stability(model, group, fitm)
    return {"model": model, "fit": fitm, "stable": bool(stable), "stability_reason": reason}


def _score_fixed_model(group: pd.DataFrame, model: WBLPowerModel) -> dict:
    return _fit_metrics(group["target_ws"], model.predict(group["ref_ws"]))


@dataclass
class Selection:
    level: str
    quarter: int | None
    sector: int | None
    selected_method: str
    model: WBLPowerModel
    fit_score: float
    selection_reason: str
    source: str


@dataclass
class AdaptiveWBLModel:
    global_selection: Selection
    quarter_selections: Dict[int, Selection]
    local_selections: Dict[Tuple[int, int], Selection]
    candidate_rows: list[dict]
    selection_rows: list[dict]

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        t = pd.to_datetime(work["target_local_time"], errors="coerce")
        q = t.dt.quarter.to_numpy()
        s = sector_index(work.get("ref_dir", pd.Series(np.nan, index=work.index)))
        pred = self.global_selection.model.predict(work["ref_ws"])
        for qq, sel in self.quarter_selections.items():
            mask = q == qq
            if mask.any():
                pred[mask] = sel.model.predict(work.loc[mask, "ref_ws"])
        for (qq, ss), sel in self.local_selections.items():
            mask = (q == qq) & (s == ss)
            if mask.any():
                pred[mask] = sel.model.predict(work.loc[mask, "ref_ws"])
        return pred

    def candidate_diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(self.candidate_rows)

    def selection_diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(self.selection_rows)


def _candidate_rows_for_group(group: pd.DataFrame, level: str, q: int | None, s: int | None) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    results: dict[str, dict] = {}
    for method in METHODS:
        base = {
            "Level": level,
            "Year division": "All" if q is None else _quarter_label(q),
            "Sector": "All" if s is None else _sector_label(s),
            "Samples": int(len(group)),
            "Internal estimator": method,
        }
        try:
            res = _evaluate_candidate(group, method)
            m = res["model"]
            ft = res["fit"]
            results[method] = res
            rows.append({
                **base,
                "Target k": m.target_k,
                "Target c (m/s)": m.target_c,
                "Reference k": m.reference_k,
                "Reference c (m/s)": m.reference_c,
                "Scale": m.scale_a,
                "Exponent": m.exponent_b,
                "Full_MAE": ft["mae"],
                "Full_RMSE": ft["rmse"],
                "Full_Bias": ft["bias"],
                "Full_WPD_Bias": ft["wpd_bias"],
                "Full_P50_Bias": ft["p50_bias"],
                "Full_P90_Bias": ft["p90_bias"],
                "Full_P95_Bias": ft["p95_bias"],
                "Full_KS": ft["ks"],
                "Full_Score": ft["score"],
                "Stable": res["stable"],
                "Stability reason": res["stability_reason"],
                "Target zero count": m.target_zero_count,
                "Reference zero count": m.reference_zero_count,
            })
        except Exception as exc:
            rows.append({
                **base,
                "Stable": False,
                "Stability reason": f"fit_failed: {exc}",
                "Full_Score": np.nan,
            })
    return rows, results


def _pick_best(results: dict[str, dict]) -> tuple[str, dict]:
    stable = [
        (name, r)
        for name, r in results.items()
        if r.get("stable") and np.isfinite(r["fit"].get("score", np.nan))
    ]
    pool = stable if stable else [
        (name, r)
        for name, r in results.items()
        if np.isfinite(r["fit"].get("score", np.nan))
    ]
    if not pool:
        raise RuntimeError("Adaptive WBL：内部候选均无法完成全样本拟合评分。")
    return min(pool, key=lambda z: z[1]["fit"]["score"])


def fit_adaptive_wbl(concurrent: pd.DataFrame) -> AdaptiveWBLModel:
    work = concurrent.copy()
    work["target_local_time"] = pd.to_datetime(work["target_local_time"], errors="coerce")
    for c in ("target_ws", "ref_ws", "ref_dir"):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_local_time", "target_ws", "ref_ws", "ref_dir"]
    )
    work = work[(work["target_ws"] >= 0.0) & (work["ref_ws"] >= 0.0)].copy()
    if len(work) < 50:
        raise RuntimeError("Adaptive WBL：全局并发样本过少。")
    work["quarter"] = work["target_local_time"].dt.quarter.astype(int)
    work["sector"] = sector_index(work["ref_dir"]).astype(int)

    candidate_rows: list[dict] = []
    selection_rows: list[dict] = []

    rows, res = _candidate_rows_for_group(work, "GLOBAL", None, None)
    candidate_rows.extend(rows)
    gm, gr = _pick_best(res)
    global_sel = Selection(
        "GLOBAL", None, None, gm, gr["model"], float(gr["fit"]["score"]),
        "all concurrent samples; best stable internal estimator", "GLOBAL",
    )
    selection_rows.append(_selection_row(global_sel, len(work)))

    quarter_selections: Dict[int, Selection] = {}
    for q in range(1, 5):
        g = work[work["quarter"] == q].copy()
        n = len(g)
        if n < MIN_QUARTER:
            continue
        rows, res = _candidate_rows_for_group(g, "QUARTER", q, None)
        candidate_rows.extend(rows)
        try:
            mm, rr = _pick_best(res)
        except Exception:
            continue
        local_score = float(rr["fit"]["score"])
        fallback_score = float(_score_fixed_model(g, global_sel.model)["score"])
        stable = bool(rr.get("stable", False))
        if n >= 500:
            use_q = stable and (not np.isfinite(fallback_score) or local_score <= (1.0 + QUARTER_LARGE_TOLERANCE) * fallback_score)
            reason = "quarter n>=500; stable and within 2% of global on this quarter"
        else:
            use_q = stable and np.isfinite(fallback_score) and local_score <= (1.0 - QUARTER_IMPROVEMENT) * fallback_score
            reason = "120<=quarter n<500; requires >=3% improvement over global"
        if use_q:
            sel = Selection("QUARTER", q, None, mm, rr["model"], local_score, reason, "QUARTER")
            quarter_selections[q] = sel
            selection_rows.append(_selection_row(sel, n, fallback_score=fallback_score))
        else:
            selection_rows.append(_quarter_fallback_row(q, n, global_sel, reason, mm, local_score, fallback_score))

    local_selections: Dict[Tuple[int, int], Selection] = {}
    for q in range(1, 5):
        quarter_fallback = quarter_selections.get(q, global_sel)
        for s in range(16):
            g = work[(work["quarter"] == q) & (work["sector"] == s)].copy()
            n = len(g)
            if n < MIN_LOCAL:
                selection_rows.append(_fallback_row(q, s, n, quarter_fallback, "n<60 -> quarter/global"))
                continue
            rows, res = _candidate_rows_for_group(g, "LOCAL", q, s)
            candidate_rows.extend(rows)
            try:
                mm, rr = _pick_best(res)
            except Exception:
                selection_rows.append(_fallback_row(q, s, n, quarter_fallback, "local fit failed -> quarter/global"))
                continue
            local_score = float(rr["fit"]["score"])
            fallback_score = float(_score_fixed_model(g, quarter_fallback.model)["score"])
            stable = bool(rr.get("stable", False))
            if n >= FULL_LOCAL:
                use_local = stable and (not np.isfinite(fallback_score) or local_score <= (1.0 + LARGE_LOCAL_TOLERANCE) * fallback_score)
                reason = "n>=240; stable local can be up to 1% worse than fallback to preserve local distribution"
            else:
                use_local = stable and np.isfinite(fallback_score) and local_score <= (1.0 - SMALL_LOCAL_IMPROVEMENT) * fallback_score
                reason = "60<=n<240; local must improve fallback by >=5%"
            if use_local:
                sel = Selection("LOCAL", q, s, mm, rr["model"], local_score, reason, "LOCAL")
                local_selections[(q, s)] = sel
                selection_rows.append(_selection_row(sel, n, fallback_score=fallback_score))
            else:
                selection_rows.append(
                    _fallback_row(
                        q, s, n, quarter_fallback, reason,
                        local_candidate=mm, local_score=local_score,
                        fallback_score=fallback_score,
                    )
                )

    return AdaptiveWBLModel(global_sel, quarter_selections, local_selections, candidate_rows, selection_rows)


def _selection_row(sel: Selection, n: int, fallback_score: float | None = None) -> dict:
    m = sel.model
    return {
        "Year division": "All" if sel.quarter is None else _quarter_label(sel.quarter),
        "Sector": "All" if sel.sector is None else _sector_label(sel.sector),
        "Samples": int(n),
        "Selected source": sel.source,
        "Selected internal estimator": sel.selected_method,
        "Target k": m.target_k,
        "Target c (m/s)": m.target_c,
        "Reference k": m.reference_k,
        "Reference c (m/s)": m.reference_c,
        "Scale": m.scale_a,
        "Exponent": m.exponent_b,
        "Full_Score": sel.fit_score,
        "Fallback score on scope": fallback_score,
        "Selection reason": sel.selection_reason,
    }


def _quarter_fallback_row(q: int, n: int, fallback: Selection, reason: str, candidate: str, local_score: float, fallback_score: float) -> dict:
    m = fallback.model
    return {
        "Year division": _quarter_label(q),
        "Sector": "All",
        "Samples": int(n),
        "Selected source": fallback.source,
        "Selected internal estimator": fallback.selected_method,
        "Target k": m.target_k,
        "Target c (m/s)": m.target_c,
        "Reference k": m.reference_k,
        "Reference c (m/s)": m.reference_c,
        "Scale": m.scale_a,
        "Exponent": m.exponent_b,
        "Full_Score": fallback.fit_score,
        "Quarter candidate": candidate,
        "Quarter candidate score": local_score,
        "Fallback score on scope": fallback_score,
        "Selection reason": reason + " -> GLOBAL",
    }


def _fallback_row(
    q: int,
    s: int,
    n: int,
    fallback: Selection,
    reason: str,
    local_candidate: str = "",
    local_score: float = np.nan,
    fallback_score: float = np.nan,
) -> dict:
    m = fallback.model
    return {
        "Year division": _quarter_label(q),
        "Sector": _sector_label(s),
        "Samples": int(n),
        "Selected source": fallback.source,
        "Selected internal estimator": fallback.selected_method,
        "Target k": m.target_k,
        "Target c (m/s)": m.target_c,
        "Reference k": m.reference_k,
        "Reference c (m/s)": m.reference_c,
        "Scale": m.scale_a,
        "Exponent": m.exponent_b,
        "Full_Score": fallback.fit_score,
        "Local candidate": local_candidate,
        "Local candidate score": local_score,
        "Fallback score on scope": fallback_score,
        "Selection reason": reason,
    }
