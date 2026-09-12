"""The null model - the part that decides whether a result is a result.

A change-detection pipeline will always return a number. The question is
whether that number is larger than the number the same pipeline returns when
it is given two images between which nothing can have changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseFloor", "estimate_noise_floor", "detectable"]


@dataclass
class NoiseFloor:
    """What the pipeline reports when there is nothing to report."""

    sigma: float          # robust standard deviation of the null difference
    p95: float            # 95th percentile of |null difference|
    n_pixels: int

    def __str__(self) -> str:  # pragma: no cover
        return f"noise floor: sigma={self.sigma:.4f}, p95|d|={self.p95:.4f} over {self.n_pixels} px"


def estimate_noise_floor(null_difference: np.ndarray) -> NoiseFloor:
    """Measure the noise from a difference that should contain no change.

    In practice the null pair is two composites from the same season of the
    same year - close enough in time that real benthic change is negligible,
    far enough apart that they are genuinely different acquisitions and carry
    the full radiometric, atmospheric and sea-state variation of the archive.

    The spread is estimated with the median absolute deviation rather than the
    plain standard deviation: a handful of cloud-edge pixels would otherwise
    set the noise floor on their own.
    """
    diff = np.asarray(null_difference, dtype=float)
    values = diff[np.isfinite(diff)]
    if values.size < 10:
        raise ValueError("need at least 10 valid pixels to estimate a noise floor")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad  # MAD -> sigma for a normal distribution

    if sigma == 0.0:
        sigma = float(np.std(values))

    return NoiseFloor(
        sigma=sigma,
        p95=float(np.percentile(np.abs(values), 95)),
        n_pixels=int(values.size),
    )


def detectable(observed_area_km2: float, null_area_km2: float, margin: float = 1.0) -> bool:
    """Is the observed extent above what the null model already produces?

    ``margin`` is how many times larger the observed area has to be before the
    result is worth reporting. 1.0 means "larger at all", which is the weakest
    defensible bar; 2.0 is a reasonable working choice.

    This is the function that should have been called before publishing a
    number. In the Red Sea run that motivated this repository, the six-year
    difference produced 62 km2 and the two-month null model produced 183 km2 -
    three times more "change" in two months than in six years. The 62 km2 was
    an artefact, and this call returns False for it.
    """
    if null_area_km2 < 0 or observed_area_km2 < 0:
        raise ValueError("areas cannot be negative")
    return observed_area_km2 > margin * null_area_km2
