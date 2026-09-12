"""Shallow-water change detection, and the test that decides whether to believe it."""

from .change import (
    ChangeResult,
    difference,
    noise_scaled_threshold,
    null_calibrated_threshold,
    percentile_threshold,
)
from .lyzenga import attenuation_ratio, deep_water_offset, depth_invariant_index
from .normalize import Normalisation, fit_normalisation, relative_normalize
from .stats import NoiseFloor, detectable, estimate_noise_floor
from .synth import Scene, make_scene

__version__ = "0.1.0"

__all__ = [
    "attenuation_ratio",
    "deep_water_offset",
    "depth_invariant_index",
    "difference",
    "percentile_threshold",
    "noise_scaled_threshold",
    "null_calibrated_threshold",
    "ChangeResult",
    "estimate_noise_floor",
    "detectable",
    "NoiseFloor",
    "make_scene",
    "Scene",
    "relative_normalize",
    "fit_normalisation",
    "Normalisation",
]
