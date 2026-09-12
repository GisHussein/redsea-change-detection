"""The whole argument, in one runnable script.

    python run_demo.py

No Earth Engine account, no downloads. It builds a shallow-water scene whose
changed area is known in advance, runs the two thresholds anyone would reach
for, and shows which one measures the seabed and which one only restates its
own parameters.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from redsea import (
    depth_invariant_index,
    detectable,
    difference,
    estimate_noise_floor,
    make_scene,
    null_calibrated_threshold,
    percentile_threshold,
)
from redsea.lyzenga import _safe_log, attenuation_ratio, deep_water_offset

PIXEL_AREA_M2 = 100.0             # 10 m Sentinel-2 pixel
CHANGED_BOX = (34, 40, 64, 110)   # 30 x 70 px = 0.21 km2, in 3-6 m of water


def dii(scene, ratio):
    return depth_invariant_index(scene.green, scene.blue, ratio=ratio)


def score(mask: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    fn = int((~mask & truth).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return recall, precision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--drift",
        type=float,
        default=0.004,
        help="radiometric offset between composites (default: 0.004 reflectance)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=8.0,
        help="only analyse water shallower than this, in metres (default: 8)",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=0.01,
        help="false-positive rate to calibrate the threshold at (default: 0.01)",
    )
    parser.add_argument("--figure", help="write a PNG summary to this path")
    args = parser.parse_args(argv)

    # --- the pair we care about ------------------------------------------
    epoch_2019 = make_scene(seed=args.seed)
    epoch_2025 = make_scene(
        seed=args.seed + 1,
        changed_box=CHANGED_BOX,
        drift_green=args.drift,
        drift_blue=args.drift * 0.6,
    )

    # --- the null pair ----------------------------------------------------
    # Same seabed, same season, two different acquisitions. Any "change" this
    # pair produces is the pipeline talking to itself.
    null_a = make_scene(seed=args.seed + 10)
    null_b = make_scene(
        seed=args.seed + 11,
        drift_green=args.drift,
        drift_blue=args.drift * 0.6,
    )

    # One attenuation ratio, fitted once on sand in the first epoch and reused
    # everywhere. Re-fitting it per epoch would quietly absorb part of the
    # change into the calibration.
    x_g = _safe_log(epoch_2019.green - deep_water_offset(epoch_2019.green))
    x_b = _safe_log(epoch_2019.blue - deep_water_offset(epoch_2019.blue))
    ratio = attenuation_ratio(x_g[epoch_2019.sand_mask], x_b[epoch_2019.sand_mask])

    # Optically deep water carries no bottom signal, so a difference taken
    # there is pure amplified noise. Masking it is not cosmetic.
    shallow = epoch_2019.depth <= args.max_depth
    real_diff = np.where(shallow, difference(dii(epoch_2019, ratio), dii(epoch_2025, ratio)), np.nan)
    null_diff = np.where(shallow, difference(dii(null_a, ratio), dii(null_b, ratio)), np.nan)

    truth = epoch_2025.changed_mask & shallow
    valid_px = int(np.isfinite(real_diff).sum())

    print(f"attenuation ratio ki/kj : {ratio:.4f}")
    print(f"analysis mask           : depth <= {args.max_depth:g} m  ({valid_px} px)")
    print(f"true changed area       : {truth.sum() * PIXEL_AREA_M2 / 1e6:.3f} km2  "
          f"({truth.sum() / valid_px:.1%} of the analysed area)")
    print()

    # --- what the null model says the floor is ---------------------------
    floor = estimate_noise_floor(null_diff)
    print(f"null model              : {floor}")
    print()

    # --- threshold 1: the percentile cut ---------------------------------
    pct_real = percentile_threshold(real_diff)
    pct_null = percentile_threshold(null_diff)
    recall, precision = score(pct_real.mask, truth)
    print("threshold A - percentile p5/p95")
    print(f"  real pair : {pct_real.area(PIXEL_AREA_M2):6.3f} km2   {pct_real.changed_fraction:6.2%}")
    print(f"  null pair : {pct_null.area(PIXEL_AREA_M2):6.3f} km2   {pct_null.changed_fraction:6.2%}")
    print(f"  recall {recall:.2f}   precision {precision:.2f}")
    print("  -> identical on both pairs. A p5/p95 cut flags 10% of pixels by")
    print("     construction, whatever the data contains. It cannot measure extent.")
    print()

    # --- threshold 2: calibrated on the null model -----------------------
    nc_real = null_calibrated_threshold(real_diff, null_diff, args.fpr)
    nc_null = null_calibrated_threshold(null_diff, null_diff, args.fpr)
    recall2, precision2 = score(nc_real.mask, truth)
    print(f"threshold B - calibrated on the null pair at fpr={args.fpr:.0%}")
    print(f"  threshold : |d| >= {nc_real.threshold_high:.3f}")
    print(f"  real pair : {nc_real.area(PIXEL_AREA_M2):6.3f} km2   {nc_real.changed_fraction:6.2%}")
    print(f"  null pair : {nc_null.area(PIXEL_AREA_M2):6.3f} km2   {nc_null.changed_fraction:6.2%}")
    print(f"  recall {recall2:.2f}   precision {precision2:.2f}")
    print()

    ratio_to_null = (
        nc_real.changed_fraction / nc_null.changed_fraction
        if nc_null.changed_fraction
        else float("inf")
    )
    verdict = detectable(
        nc_real.area(PIXEL_AREA_M2), nc_null.area(PIXEL_AREA_M2), margin=2.0
    )
    print(f"real / null             : {ratio_to_null:.1f}x")
    print(f"detectable(margin=2.0)  : {verdict}")
    if verdict:
        print("  the signal clears its own floor - the extent is worth reporting,")
        print(f"  with the {1 - recall2:.0%} of the patch that sits deeper than the")
        print("  method can see reported as a limit, not hidden.")
    else:
        print("  the result does not clear its own noise floor; nothing to report")

    if args.figure:
        _write_figure(
            pathlib.Path(args.figure),
            real_diff,
            null_diff,
            truth,
            pct_real.mask,
            nc_real.mask,
        )
        print(f"\nfigure written to {args.figure}")

    return 0


def _write_figure(path, real_diff, null_diff, truth, pct_mask, nc_mask):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    limit = float(np.nanpercentile(np.abs(real_diff), 98))

    panels = [
        (real_diff, "2019 - 2025 difference", "RdBu_r"),
        (null_diff, "null pair (no real change)", "RdBu_r"),
        (truth, "truth", "Greys"),
        (pct_mask, "A: p5/p95 threshold", "Greys"),
        (nc_mask, "B: null-calibrated threshold", "Greys"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            ax.imshow(data, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
