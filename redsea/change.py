"""Turning a difference image into a claim about change.

There are two ways to threshold a difference image and only one of them
measures anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ChangeResult",
    "difference",
    "percentile_threshold",
    "noise_scaled_threshold",
    "null_calibrated_threshold",
]


@dataclass
class ChangeResult:
    """A change mask plus the numbers needed to judge it."""

    mask: np.ndarray
    threshold_low: float
    threshold_high: float
    method: str
    changed_pixels: int
    valid_pixels: int

    @property
    def changed_fraction(self) -> float:
        if self.valid_pixels == 0:
            return 0.0
        return self.changed_pixels / self.valid_pixels

    def area(self, pixel_area_m2: float) -> float:
        """Changed area in square kilometres."""
        return self.changed_pixels * pixel_area_m2 / 1_000_000.0

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"{self.method}: {self.changed_pixels} / {self.valid_pixels} px "
            f"({self.changed_fraction:.2%}), "
            f"thresholds [{self.threshold_low:.4f}, {self.threshold_high:.4f}]"
        )


def difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """after - before, with NaN wherever either side is missing."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError("the two epochs must have the same shape")
    out = after - before
    out[~np.isfinite(before) | ~np.isfinite(after)] = np.nan
    return out


def percentile_threshold(diff: np.ndarray, low: float = 5.0, high: float = 95.0) -> ChangeResult:
    """Flag the tails of the difference distribution.

    This is the threshold most tutorials reach for, and it cannot measure
    extent: a p5/p95 cut flags almost exactly 10% of valid pixels whatever the
    data contains. Run it on a pair of images with no change at all and it
    still returns 10%. The number it produces restates the method, not the
    result.

    It is kept here because it is a useful *ranking* - it tells you where the
    largest differences are - and because the tests assert that it behaves
    exactly this way.
    """
    diff = np.asarray(diff, dtype=float)
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        raise ValueError("difference image has no valid pixels")

    lo = float(np.percentile(diff[valid], low))
    hi = float(np.percentile(diff[valid], high))

    mask = valid & ((diff <= lo) | (diff >= hi))
    return ChangeResult(
        mask=mask,
        threshold_low=lo,
        threshold_high=hi,
        method=f"percentile p{low:g}/p{high:g}",
        changed_pixels=int(mask.sum()),
        valid_pixels=int(valid.sum()),
    )


def noise_scaled_threshold(
    diff: np.ndarray,
    noise_sigma: float,
    k: float = 3.0,
) -> ChangeResult:
    """Flag pixels that exceed k times the measured noise of the sensor chain.

    ``noise_sigma`` comes from a null model - two epochs between which no real
    change can have happened (see :mod:`redsea.stats`). Everything below that
    level is drift in the composites, not seabed.

    Unlike a percentile cut, this threshold is anchored to something outside
    the image, so "3.2 km2 changed" is a measurement rather than a restatement
    of the percentile you picked.
    """
    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    diff = np.asarray(diff, dtype=float)
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        raise ValueError("difference image has no valid pixels")

    limit = k * noise_sigma
    mask = valid & (np.abs(diff) >= limit)
    return ChangeResult(
        mask=mask,
        threshold_low=-limit,
        threshold_high=limit,
        method=f"noise-scaled k={k:g}",
        changed_pixels=int(mask.sum()),
        valid_pixels=int(valid.sum()),
    )


def null_calibrated_threshold(
    diff: np.ndarray,
    null_diff: np.ndarray,
    false_positive_rate: float = 0.01,
) -> ChangeResult:
    """Threshold read straight off the null model.

    Take the difference image the pipeline produces when nothing can have
    changed, and pick the level that only ``false_positive_rate`` of its pixels
    exceed. Apply that same level to the real pair.

    The calibration is the whole point: the null pair now flags exactly 1% of
    pixels *by construction*, so anything the real pair flags above that is the
    part the pipeline cannot explain as drift. It turns "7.7% of pixels changed"
    into "7.7% against a 1% floor", which is a claim that can be defended.

    Unlike :func:`noise_scaled_threshold` this makes no assumption that the
    noise is normal - and in log-ratio imagery it is not, because taking the
    log of a dark pixel amplifies its noise far more than a bright one.
    """
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be between 0 and 1")

    null_values = np.asarray(null_diff, dtype=float)
    null_values = np.abs(null_values[np.isfinite(null_values)])
    if null_values.size < 10:
        raise ValueError("need at least 10 valid pixels in the null difference")

    limit = float(np.percentile(null_values, 100.0 * (1.0 - false_positive_rate)))

    diff = np.asarray(diff, dtype=float)
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        raise ValueError("difference image has no valid pixels")

    mask = valid & (np.abs(diff) >= limit)
    return ChangeResult(
        mask=mask,
        threshold_low=-limit,
        threshold_high=limit,
        method=f"null-calibrated fpr={false_positive_rate:.1%}",
        changed_pixels=int(mask.sum()),
        valid_pixels=int(valid.sum()),
    )
