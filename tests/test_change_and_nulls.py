import numpy as np
import pytest

from redsea import (
    depth_invariant_index,
    detectable,
    difference,
    estimate_noise_floor,
    make_scene,
    noise_scaled_threshold,
    null_calibrated_threshold,
    percentile_threshold,
)
from redsea.lyzenga import _safe_log, attenuation_ratio, deep_water_offset

CHANGED_BOX = (34, 40, 64, 110)


# --- difference -----------------------------------------------------------


def test_difference_propagates_missing_values():
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([2.0, np.nan, 3.0])
    out = difference(a, b)
    assert out[0] == 1.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_difference_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        difference(np.zeros(3), np.zeros(4))


# --- the percentile trap --------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_percentile_threshold_always_flags_ten_percent(seed):
    """This is the failure mode the repository exists to demonstrate.

    Whatever the data - pure noise, a huge change, no change at all - a p5/p95
    cut returns 10% of valid pixels. Any 'extent' derived from it is a
    restatement of the percentile, not a measurement.
    """
    rng = np.random.default_rng(seed)
    for data in (
        rng.normal(size=5000),                       # pure noise
        np.zeros(5000) + rng.normal(scale=1e-9, size=5000),  # nothing happening
        rng.normal(size=5000) + 10 * (rng.random(5000) < 0.3),  # a real signal
    ):
        result = percentile_threshold(data)
        assert result.changed_fraction == pytest.approx(0.10, abs=0.005)


def test_percentile_threshold_needs_valid_pixels():
    with pytest.raises(ValueError):
        percentile_threshold(np.full(10, np.nan))


# --- noise floor ----------------------------------------------------------


def test_noise_floor_is_robust_to_outliers():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1.0, size=10000)
    dirty = clean.copy()
    dirty[:50] = 500.0  # cloud edges
    assert estimate_noise_floor(dirty).sigma == pytest.approx(
        estimate_noise_floor(clean).sigma, rel=0.05
    )


def test_noise_floor_needs_enough_pixels():
    with pytest.raises(ValueError):
        estimate_noise_floor(np.zeros(5))


# --- null-calibrated threshold -------------------------------------------


def test_null_calibrated_threshold_hits_its_false_positive_rate():
    rng = np.random.default_rng(0)
    null = rng.normal(size=100_000)
    result = null_calibrated_threshold(null, null, false_positive_rate=0.01)
    assert result.changed_fraction == pytest.approx(0.01, abs=0.002)


def test_null_calibrated_threshold_validates_its_rate():
    with pytest.raises(ValueError):
        null_calibrated_threshold(np.zeros(100), np.zeros(100), false_positive_rate=1.5)


def test_noise_scaled_threshold_rejects_bad_sigma():
    with pytest.raises(ValueError):
        noise_scaled_threshold(np.zeros(100), noise_sigma=0.0)


# --- detectable -----------------------------------------------------------


def test_detectable_rejects_a_result_below_its_own_null():
    # The real Red Sea numbers: 62 km2 over six years, 183 km2 over two months.
    assert detectable(62.0, 183.0) is False
    assert detectable(200.0, 50.0, margin=2.0) is True
    assert detectable(80.0, 50.0, margin=2.0) is False


def test_detectable_rejects_negative_areas():
    with pytest.raises(ValueError):
        detectable(-1.0, 10.0)


# --- the end-to-end claim -------------------------------------------------


def _pipeline(max_depth=8.0, drift=0.004, seed=7):
    a = make_scene(seed=seed)
    b = make_scene(seed=seed + 1, changed_box=CHANGED_BOX,
                   drift_green=drift, drift_blue=drift * 0.6)
    na = make_scene(seed=seed + 10)
    nb = make_scene(seed=seed + 11, drift_green=drift, drift_blue=drift * 0.6)

    x_g = _safe_log(a.green - deep_water_offset(a.green))
    x_b = _safe_log(a.blue - deep_water_offset(a.blue))
    ratio = attenuation_ratio(x_g[a.sand_mask], x_b[a.sand_mask])

    def dii(scene):
        return depth_invariant_index(scene.green, scene.blue, ratio=ratio)

    shallow = a.depth <= max_depth
    real = np.where(shallow, difference(dii(a), dii(b)), np.nan)
    null = np.where(shallow, difference(dii(na), dii(nb)), np.nan)
    return real, null, b.changed_mask & shallow


def test_null_calibration_recovers_the_patch_and_the_percentile_does_not():
    real, null, truth = _pipeline()

    pct = percentile_threshold(real)
    nc = null_calibrated_threshold(real, null, 0.01)

    def precision(mask):
        tp = int((mask & truth).sum())
        fp = int((mask & ~truth).sum())
        return tp / (tp + fp) if tp + fp else 0.0

    # The calibrated threshold is far cleaner on the same difference image.
    assert precision(nc.mask) > 0.8
    assert precision(pct.mask) < 0.6

    # And it is honest about what it misses: the deep third of the patch sits
    # below the floor, so recall is good but not perfect.
    recall = int((nc.mask & truth).sum()) / int(truth.sum())
    assert 0.5 < recall < 0.95


def test_the_null_pair_stays_at_the_calibrated_rate():
    real, null, _ = _pipeline()
    nc_null = null_calibrated_threshold(null, null, 0.01)
    nc_real = null_calibrated_threshold(real, null, 0.01)
    assert nc_null.changed_fraction == pytest.approx(0.01, abs=0.002)
    # A real change has to stand well clear of that.
    assert nc_real.changed_fraction > 5 * nc_null.changed_fraction


def test_no_change_means_no_detection():
    """Feed the pipeline two epochs with no change and it must report none."""
    real, null, _ = _pipeline()
    nc = null_calibrated_threshold(null, null, 0.01)
    assert not detectable(nc.area(100.0), nc.area(100.0), margin=2.0)
