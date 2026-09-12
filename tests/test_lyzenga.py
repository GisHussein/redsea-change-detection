import numpy as np
import pytest

from redsea import make_scene
from redsea.lyzenga import (
    _safe_log,
    attenuation_ratio,
    deep_water_offset,
    depth_invariant_index,
)


def test_deep_water_offset_is_a_low_quantile():
    band = np.linspace(0.02, 0.5, 1000)
    assert deep_water_offset(band, quantile=0.01) == pytest.approx(0.0248, abs=1e-3)


def test_deep_water_offset_rejects_an_all_nan_band():
    with pytest.raises(ValueError):
        deep_water_offset(np.full(10, np.nan))


def test_safe_log_turns_non_positive_into_nan():
    out = _safe_log(np.array([-1.0, 0.0, 1.0, np.e]))
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(1.0)


def test_attenuation_ratio_recovers_the_known_ratio():
    # Build a sand-only column: x_i and x_j are linear in depth with slopes
    # -2*k_i and -2*k_j, so the ratio the estimator returns must be k_i/k_j.
    depth = np.linspace(0.5, 12.0, 500)
    k_i, k_j = 0.19, 0.09
    x_i = np.log(0.48) - 2 * k_i * depth
    x_j = np.log(0.42) - 2 * k_j * depth
    assert attenuation_ratio(x_i, x_j) == pytest.approx(k_i / k_j, rel=0.02)


def test_attenuation_ratio_needs_a_depth_range():
    flat = np.full(50, 0.3)
    with pytest.raises(ValueError):
        attenuation_ratio(flat, flat)


def test_attenuation_ratio_rejects_mismatched_samples():
    with pytest.raises(ValueError):
        attenuation_ratio(np.zeros(10), np.zeros(11))


def test_index_is_flat_over_one_bottom_type_at_varying_depth():
    """The whole point: depth must drop out.

    Over sand only, the index should vary far less with depth than either raw
    band does. That is the property the rest of the pipeline depends on.
    """
    scene = make_scene(noise=0.0, seed=3)
    index = depth_invariant_index(
        scene.green, scene.blue, training_mask=scene.sand_mask
    )
    sand = scene.sand_mask & np.isfinite(index)

    index_spread = float(np.nanstd(index[sand]))
    band_spread = float(np.nanstd(np.log(scene.green[sand] - deep_water_offset(scene.green))))

    assert index_spread < band_spread / 5


def test_index_separates_bottom_types():
    scene = make_scene(noise=0.0, seed=3)
    index = depth_invariant_index(
        scene.green, scene.blue, training_mask=scene.sand_mask
    )
    sand = np.nanmedian(index[(scene.bottom == 0) & np.isfinite(index)])
    seagrass = np.nanmedian(index[(scene.bottom == 1) & np.isfinite(index)])
    assert abs(sand - seagrass) > 0.3


def test_index_requires_a_ratio_or_a_training_mask():
    scene = make_scene(seed=1)
    with pytest.raises(ValueError):
        depth_invariant_index(scene.green, scene.blue)


def test_index_rejects_mismatched_bands():
    with pytest.raises(ValueError):
        depth_invariant_index(np.zeros((4, 4)), np.zeros((4, 5)), ratio=1.5)
