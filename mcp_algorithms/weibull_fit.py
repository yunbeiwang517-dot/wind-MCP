from __future__ import annotations

"""Deterministic 2-parameter Weibull fitting utilities for WBL MCP.

Log-based methods (ML and LS) fit the strictly positive wind-speed component.
Moment-based methods (WAsP and Openwind) can include exact zero/calm samples in
their empirical moments, which is one reason they can behave differently in
low-wind sectors.

Implemented methods:
- ML: Stevens-Smulders fixed-point maximum likelihood.
- LS: linear least squares on a Weibull probability plot using median ranks.
- WAsP: preserve third moment (wind power density proxy) and the observed
  probability of exceeding the observed mean wind speed.
- Openwind: preserve the observed mean wind speed and third moment.
"""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class WeibullFit:
    method: str
    k: float
    c: float
    positive_n: int


def _positive(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x) & (x > 0.0)]


def stevens_smulders_mle(values, initial_k: float = 2.0, tol: float = 1e-12, max_iter: int = 200) -> WeibullFit:
    x = _positive(values)
    if len(x) < 2:
        raise RuntimeError("Weibull ML失败：正风速样本少于2个。")
    logx = np.log(x)
    k = float(initial_k) if np.isfinite(initial_k) and initial_k > 0 else 2.0
    for _ in range(int(max_iter)):
        z = k * logx
        zmax = float(np.max(z))
        w = np.exp(z - zmax)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0:
            raise RuntimeError("Weibull ML失败：迭代权重无效。")
        wl = float(np.sum(w * logx) / sw)
        denom = wl - float(np.mean(logx))
        if not np.isfinite(denom) or denom <= 0:
            raise RuntimeError("Weibull ML失败：Stevens-Smulders迭代无效。")
        knew = float(1.0 / denom)
        if not np.isfinite(knew) or knew <= 0:
            raise RuntimeError("Weibull ML失败：k无效。")
        if abs(knew - k) / max(abs(k), 1e-12) < tol:
            k = knew
            break
        k = knew
    z = k * logx
    zmax = float(np.max(z))
    log_mean_power = zmax + math.log(float(np.mean(np.exp(z - zmax))))
    c = float(math.exp(log_mean_power / k))
    if not np.isfinite(c) or c <= 0:
        raise RuntimeError("Weibull ML失败：c无效。")
    return WeibullFit("Maximum Likelihood", float(k), c, int(len(x)))


def least_squares_weibull(values) -> WeibullFit:
    x = np.sort(_positive(values))
    n = len(x)
    if n < 3:
        raise RuntimeError("Weibull LS失败：正风速样本少于3个。")
    # Benard/median-rank plotting position; deterministic and well behaved at
    # both tails.  The adaptive WBL does not claim Windographer LS identity.
    ranks = np.arange(1, n + 1, dtype=float)
    F = (ranks - 0.3) / (n + 0.4)
    F = np.clip(F, 1e-9, 1.0 - 1e-9)
    xx = np.log(x)
    yy = np.log(-np.log(1.0 - F))
    A = np.vstack([xx, np.ones_like(xx)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    k = float(slope)
    if not np.isfinite(k) or k <= 0:
        raise RuntimeError("Weibull LS失败：k无效。")
    c = float(math.exp(-float(intercept) / k))
    if not np.isfinite(c) or c <= 0:
        raise RuntimeError("Weibull LS失败：c无效。")
    return WeibullFit("Least Squares", k, c, int(n))


def _bisect_root(func: Callable[[float], float], lo: float = 0.15, hi: float = 50.0, max_iter: int = 200, tol: float = 1e-10) -> float:
    # Try a log grid first so we can find a valid sign-changing interval even
    # when the physically useful root is far from k=2.
    grid = np.geomspace(lo, hi, 240)
    prev_x = None
    prev_f = None
    for x in grid:
        try:
            fx = float(func(float(x)))
        except Exception:
            continue
        if not np.isfinite(fx):
            continue
        if abs(fx) < tol:
            return float(x)
        if prev_f is not None and fx * prev_f < 0:
            a, b = float(prev_x), float(x)
            fa, fb = float(prev_f), float(fx)
            for _ in range(max_iter):
                m = 0.5 * (a + b)
                fm = float(func(m))
                if not np.isfinite(fm):
                    raise RuntimeError("Weibull根求解失败：函数值无效。")
                if abs(fm) < tol or abs(b - a) < tol * max(1.0, abs(m)):
                    return float(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return float(0.5 * (a + b))
        prev_x, prev_f = float(x), float(fx)
    raise RuntimeError("Weibull根求解失败：没有找到有效根区间。")


def openwind_weibull(values) -> WeibullFit:
    allx = np.asarray(values, dtype=float)
    x = allx[np.isfinite(allx) & (allx >= 0.0)]
    positive_n = int(np.sum(x > 0.0))
    if len(x) < 3 or positive_n < 2:
        raise RuntimeError("Openwind Weibull失败：有效样本不足。")
    mean_v = float(np.mean(x))
    mean_v3 = float(np.mean(x ** 3))
    if mean_v <= 0 or mean_v3 <= 0:
        raise RuntimeError("Openwind Weibull失败：矩无效。")
    target_log_ratio = math.log(mean_v3 / (mean_v ** 3))

    def f(k: float) -> float:
        return math.lgamma(1.0 + 3.0 / k) - 3.0 * math.lgamma(1.0 + 1.0 / k) - target_log_ratio

    k = _bisect_root(f)
    c = mean_v / math.exp(math.lgamma(1.0 + 1.0 / k))
    if not np.isfinite(c) or c <= 0:
        raise RuntimeError("Openwind Weibull失败：c无效。")
    return WeibullFit("Openwind", float(k), float(c), positive_n)


def wasp_weibull(values) -> WeibullFit:
    allx = np.asarray(values, dtype=float)
    x = allx[np.isfinite(allx) & (allx >= 0.0)]
    positive_n = int(np.sum(x > 0.0))
    if len(x) < 3 or positive_n < 2:
        raise RuntimeError("WAsP Weibull失败：有效样本不足。")
    mean_v = float(np.mean(x))
    mean_v3 = float(np.mean(x ** 3))
    p_above = float(np.mean(x > mean_v))
    if mean_v <= 0 or mean_v3 <= 0 or not (0.0 < p_above < 1.0):
        raise RuntimeError("WAsP Weibull失败：样本统计量无效。")
    L = -math.log(p_above)
    log_target_m3 = math.log(mean_v3)

    def f(k: float) -> float:
        # A = mean / (-ln P(V>mean))^(1/k)
        log_a = math.log(mean_v) - math.log(L) / k
        return 3.0 * log_a + math.lgamma(1.0 + 3.0 / k) - log_target_m3

    k = _bisect_root(f)
    c = mean_v / (L ** (1.0 / k))
    if not np.isfinite(c) or c <= 0:
        raise RuntimeError("WAsP Weibull失败：c无效。")
    return WeibullFit("WAsP", float(k), float(c), positive_n)


def fit_weibull(values, method: str) -> WeibullFit:
    key = str(method).strip().lower().replace("_", " ")
    if key in {"ml", "maximum likelihood", "maximum-likelihood", "stevens smulders", "stevens-smulders"}:
        return stevens_smulders_mle(values)
    if key in {"ls", "least squares", "least-squares"}:
        return least_squares_weibull(values)
    if key in {"wasp", "w as p"}:
        return wasp_weibull(values)
    if key in {"openwind", "open wind"}:
        return openwind_weibull(values)
    raise ValueError(f"未知Weibull拟合方法：{method}")


METHODS = ("Maximum Likelihood", "Least Squares", "WAsP", "Openwind")
