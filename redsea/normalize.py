"""Relative radiometric normalisation.

Two composites of the same reef, three years apart, are never on the same
radiometric scale. Sun angle, sea state and whatever the atmospheric
correction failed to remove leave a systematic offset between them.

That offset is not noise. Dividing by a standard deviation does not remove it -
it shifts every pixel in the scene the same way, so it survives any threshold
that is symmetric about zero, and it shows up as "change" everywhere.

The fix is to put the second epoch on the first one's scale before differencing
anything, by fitting a gain and an offset over pixels that are assumed not to
have changed (pseudo-invariant features). The fit is robust, so the handful of
pixels that *did* change do not drag it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Normalisation", "fit_normalisation", "relative_normalize"]


@dataclass
class Normalisation:
    """target ~ gain * reference + offset, in reflectance units."""

    gain: float
    offset: float
    n_pixels: int

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Put ``target`` back onto the reference scale."""
        return (np.asarray(target, dtype=float) - self.offset) / self.gain

    def __str__(self) -> str:  # pragma: no cover
        return f"gain={self.gain:.4f} offset={self.offset:+.5f} ({self.n_pixels} px)"


def _trimmed_fit(x: np.ndarray, y: np.ndarray, trim: float) -> tuple[float, float]:
    """Least squares, then refit on the best-fitting ``1 - trim`` of points.

    One pass of trimming is enough here and keeps the function dependency-free:
    real changed pixels are outliers to the relationship, and dropping the worst
    residuals removes them without needing to know where they are.
    """
    gain, offset = np.polyfit(x, y, 1)
    if trim <= 0:
        return float(gain), float(offset)

    residuals = np.abs(y - (gain * x + offset))
    keep = residuals <= np.quantile(residuals, 1.0 - trim)
    if keep.sum() < 3:
        return float(gain), float(offset)

    gain, offset = np.polyfit(x[keep], y[keep], 1)
    return float(gain), float(offset)


def fit_normalisation(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    trim: float = 0.15,
) -> Normalisation:
    """Fit the gain and offset that map ``target`` onto ``reference``.

    ``mask`` restricts the fit to pixels believed to be invariant - deep water,
    bare rock, an area known not to have been touched. With no mask the whole
    scene is used and the trimming does the work, which is fine as long as the
    changed fraction is small.
    """
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    if reference.shape != target.shape:
        raise ValueError("reference and target must have the same shape")

    good = np.isfinite(reference) & np.isfinite(target)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    if good.sum() < 10:
        raise ValueError("not enough valid pixels to fit a normalisation")

    gain, offset = _trimmed_fit(reference[good], target[good], trim)
    if not np.isfinite(gain) or np.isclose(gain, 0.0):
        raise ValueError("degenerate normalisation fit")

    return Normalisation(gain=gain, offset=offset, n_pixels=int(good.sum()))


def relative_normalize(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    trim: float = 0.15,
) -> tuple[np.ndarray, Normalisation]:
    """Convenience wrapper: fit, then apply."""
    fit = fit_normalisation(reference, target, mask=mask, trim=trim)
    return fit.apply(target), fit
