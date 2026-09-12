import numpy as np
import pytest

from redsea.normalize import fit_normalisation, relative_normalize


def test_recovers_a_known_gain_and_offset():
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.02, 0.5, size=5000)
    target = 1.08 * reference + 0.004

    fit = fit_normalisation(reference, target)
    assert fit.gain == pytest.approx(1.08, rel=1e-3)
    assert fit.offset == pytest.approx(0.004, abs=1e-4)


def test_applying_the_fit_puts_the_target_back_on_scale():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.02, 0.5, size=5000)
    target = 1.08 * reference + 0.004

    corrected, _ = relative_normalize(reference, target)
    assert np.allclose(corrected, reference, atol=1e-6)


def test_changed_pixels_do_not_drag_the_fit():
    """Trimming is what makes this usable on a scene that contains change."""
    rng = np.random.default_rng(2)
    reference = rng.uniform(0.02, 0.5, size=5000)
    target = 1.08 * reference + 0.004
    target[:400] += 0.25  # 8% of the scene really did change

    fit = fit_normalisation(reference, target, trim=0.15)
    assert fit.gain == pytest.approx(1.08, rel=0.02)
    assert fit.offset == pytest.approx(0.004, abs=5e-4)


def test_an_invariant_mask_is_respected():
    rng = np.random.default_rng(3)
    reference = rng.uniform(0.02, 0.5, size=1000)
    target = 1.05 * reference + 0.002
    target[:200] = 0.9  # nonsense, but masked out

    mask = np.ones(1000, dtype=bool)
    mask[:200] = False

    fit = fit_normalisation(reference, target, mask=mask)
    assert fit.gain == pytest.approx(1.05, rel=1e-3)
    assert fit.n_pixels == 800


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fit_normalisation(np.zeros(10), np.zeros(11))


def test_rejects_too_few_pixels():
    with pytest.raises(ValueError):
        fit_normalisation(np.zeros(4), np.zeros(4))
