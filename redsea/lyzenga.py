"""Lyzenga depth-invariant indices.

Shallow-water reflectance is a mixture of two things: what is on the seabed and
how much water sits above it. Lyzenga (1978, 1981) showed that for a pair of
bands with different attenuation coefficients, a log-ratio combination cancels
the depth term and leaves a quantity that depends only on bottom type.

That is what makes change detection possible at all: without it, a change in
tide or a change in turbidity reads exactly like a change in the seabed.

The maths here is deliberately plain NumPy - no Earth Engine, no rasterio - so
it can be tested against data whose answer is known in advance.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "deep_water_offset",
    "attenuation_ratio",
    "depth_invariant_index",
]


def deep_water_offset(band: np.ndarray, quantile: float = 0.01) -> float:
    """Estimate the deep-water radiance to subtract from a band.

    Over optically deep water the signal that reaches the sensor is atmosphere
    and water column only - no bottom. Taking a low quantile of the scene is a
    robust stand-in for "deep water" when no deep-water mask is available, and
    it tolerates the handful of dark pixels that every scene carries.
    """
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        raise ValueError("band contains no finite values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    return float(np.quantile(finite, quantile))


def attenuation_ratio(
    x_i: np.ndarray,
    x_j: np.ndarray,
) -> float:
    """Ratio of attenuation coefficients ki/kj for a band pair.

    ``x_i`` and ``x_j`` are the log-transformed, deep-water-corrected values of
    a *uniform bottom type* at varying depth - classically a stretch of sand.
    The ratio is derived from the variances and covariance of that sample:

        a = (var_i - var_j) / (2 * cov_ij)
        ki/kj = a + sqrt(a^2 + 1)

    Choosing the training sample well matters more than any other decision in
    the pipeline. A "sand" polygon that quietly includes seagrass produces a
    ratio that is wrong everywhere.
    """
    x_i = np.asarray(x_i, dtype=float).ravel()
    x_j = np.asarray(x_j, dtype=float).ravel()
    if x_i.shape != x_j.shape:
        raise ValueError("training samples must have the same shape")

    good = np.isfinite(x_i) & np.isfinite(x_j)
    if good.sum() < 3:
        raise ValueError("need at least 3 finite training pixels")

    x_i = x_i[good]
    x_j = x_j[good]

    var_i = float(np.var(x_i, ddof=1))
    var_j = float(np.var(x_j, ddof=1))
    cov_ij = float(np.cov(x_i, x_j, ddof=1)[0, 1])

    if np.isclose(cov_ij, 0.0):
        raise ValueError(
            "the training sample has no covariance between the two bands; "
            "it probably does not span a range of depths"
        )

    a = (var_i - var_j) / (2.0 * cov_ij)
    return float(a + np.sqrt(a * a + 1.0))


def _safe_log(values: np.ndarray) -> np.ndarray:
    """log() that turns non-positive input into NaN instead of -inf.

    After subtracting the deep-water offset a few pixels always land at or
    below zero. Letting them become -inf poisons every statistic downstream;
    NaN keeps them out of the way and countable.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0
    out[positive] = np.log(values[positive])
    return out


def depth_invariant_index(
    band_i: np.ndarray,
    band_j: np.ndarray,
    *,
    deep_i: float | None = None,
    deep_j: float | None = None,
    ratio: float | None = None,
    training_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Depth-invariant index for one band pair.

        DII = ln(Li - Li_deep) - (ki/kj) * ln(Lj - Lj_deep)

    If ``ratio`` is not given it is estimated from ``training_mask``, which
    should select a single bottom type across a range of depths.
    """
    band_i = np.asarray(band_i, dtype=float)
    band_j = np.asarray(band_j, dtype=float)
    if band_i.shape != band_j.shape:
        raise ValueError("the two bands must have the same shape")

    if deep_i is None:
        deep_i = deep_water_offset(band_i)
    if deep_j is None:
        deep_j = deep_water_offset(band_j)

    x_i = _safe_log(band_i - deep_i)
    x_j = _safe_log(band_j - deep_j)

    if ratio is None:
        if training_mask is None:
            raise ValueError("pass either `ratio` or a `training_mask`")
        training_mask = np.asarray(training_mask, dtype=bool)
        if training_mask.shape != band_i.shape:
            raise ValueError("training_mask must match the band shape")
        ratio = attenuation_ratio(x_i[training_mask], x_j[training_mask])

    return x_i - ratio * x_j
