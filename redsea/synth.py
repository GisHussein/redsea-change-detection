"""A synthetic reef scene whose answer is known in advance.

Earth Engine needs an account and an internet connection; a demonstration that
the *method* is sound should need neither. This module builds a small
shallow-water scene with:

  * a depth gradient running offshore,
  * three bottom types (sand, seagrass, coral),
  * a patch that really does change between two epochs,
  * per-epoch radiometric drift - the sun angle, sea state and residual
    atmospheric correction that differ between two composites, and
  * sensor noise.

Because the true changed area is set by the caller, any threshold can be
scored against it instead of argued about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Scene", "make_scene"]

# Bottom albedo per class, for a green and a blue band.
_BOTTOM = {
    "sand": (0.48, 0.42),
    "seagrass": (0.16, 0.14),
    "coral": (0.30, 0.24),
}

# Diffuse attenuation. Blue penetrates further than green - which is exactly
# what makes the Lyzenga pair work.
K_GREEN = 0.19
K_BLUE = 0.09

DEEP_GREEN = 0.021
DEEP_BLUE = 0.034


@dataclass
class Scene:
    """One epoch: two bands plus the truth used to build it."""

    green: np.ndarray
    blue: np.ndarray
    depth: np.ndarray
    bottom: np.ndarray          # integer class map
    sand_mask: np.ndarray       # training sample for the attenuation ratio
    changed_mask: np.ndarray    # truth, empty for a null epoch

    @property
    def shape(self) -> tuple[int, int]:
        return self.green.shape


def _bottom_map(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """0 = sand, 1 = seagrass, 2 = coral, in coherent patches."""
    bottom = np.zeros((height, width), dtype=int)

    yy, xx = np.mgrid[0:height, 0:width]

    # A seagrass meadow in the middle of the shelf.
    meadow = ((yy - height * 0.45) ** 2 / (height * 0.18) ** 2
              + (xx - width * 0.4) ** 2 / (width * 0.28) ** 2) < 1
    bottom[meadow] = 1

    # A reef ribbon along the shallow edge.
    ribbon = np.abs(yy - (height * 0.18 + 6 * np.sin(xx / 9.0))) < 4
    bottom[ribbon] = 2

    # Speckle, so the classes are not perfectly clean.
    speckle = rng.random((height, width)) < 0.02
    bottom[speckle] = rng.integers(0, 3, size=int(speckle.sum()))
    return bottom


def make_scene(
    height: int = 160,
    width: int = 220,
    *,
    changed_box: tuple[int, int, int, int] | None = None,
    drift_green: float = 0.0,
    drift_blue: float = 0.0,
    noise: float = 0.0010,
    seed: int = 0,
    bottom_seed: int = 1234,
) -> Scene:
    """Build one epoch.

    Parameters
    ----------
    changed_box:
        ``(row0, col0, row1, col1)`` of a patch that becomes sand in this
        epoch - the real seabed change. ``None`` leaves the seabed untouched.
    drift_green, drift_blue:
        Additive radiometric offset applied to the whole scene, standing in for
        the difference between two composites that no atmospheric correction
        fully removes. This is the term the null model is designed to expose.
    noise:
        Standard deviation of per-pixel sensor noise.
    bottom_seed:
        Seed for the seabed itself. It is deliberately separate from ``seed``
        and shared across epochs: the seabed is the thing being measured, so it
        must be identical in every epoch except where ``changed_box`` says
        otherwise. Varying it per epoch would inject change into the null model
        and quietly invalidate the whole experiment.
    """
    rng = np.random.default_rng(seed)
    bottom_rng = np.random.default_rng(bottom_seed)

    # Depth: 0.5 m inshore to about 14 m offshore, with mild relief.
    rows = np.linspace(0.5, 14.0, height)[:, None]
    relief = 0.45 * np.sin(np.linspace(0, 5.0, width))[None, :]
    depth = np.clip(rows + relief, 0.3, None) * np.ones((1, width))

    bottom = _bottom_map(height, width, bottom_rng)

    changed = np.zeros((height, width), dtype=bool)
    if changed_box is not None:
        r0, c0, r1, c1 = changed_box
        changed[r0:r1, c0:c1] = True
        bottom = bottom.copy()
        bottom[changed] = 0  # dredged / buried: whatever was there is sand now

    albedo_green = np.zeros((height, width))
    albedo_blue = np.zeros((height, width))
    for index, name in enumerate(["sand", "seagrass", "coral"]):
        g, b = _BOTTOM[name]
        albedo_green[bottom == index] = g
        albedo_blue[bottom == index] = b

    green = DEEP_GREEN + albedo_green * np.exp(-2.0 * K_GREEN * depth) + drift_green
    blue = DEEP_BLUE + albedo_blue * np.exp(-2.0 * K_BLUE * depth) + drift_blue

    if noise:
        green = green + rng.normal(0.0, noise, size=green.shape)
        blue = blue + rng.normal(0.0, noise, size=blue.shape)

    # Training sample: sand only, across the depth range, away from the
    # changed patch so the ratio is not fitted on the thing being measured.
    sand_mask = (bottom == 0) & ~changed

    return Scene(
        green=green,
        blue=blue,
        depth=depth,
        bottom=bottom,
        sand_mask=sand_mask,
        changed_mask=changed,
    )
