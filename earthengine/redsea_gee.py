"""The same pipeline against real Sentinel-2 data in Earth Engine.

This is the part that needs an account:

    pip install earthengine-api
    earthengine authenticate
    python earthengine/redsea_gee.py --project my-ee-project

It builds the two epoch composites, the null pair, the depth-invariant index
and the difference, and exports the change mask. The thresholds and the
verdict are computed in `redsea/` on the arrays that come back, so the logic
that decides whether a result is real is the same logic the tests cover.

Default AOI: the Egyptian Red Sea coast between Marsa Alam and Ras Banas.
"""

from __future__ import annotations

import argparse

try:
    import ee
except ImportError:  # pragma: no cover - the offline demo does not need this
    ee = None

# Marsa Alam -> Ras Banas, roughly 180 km of coastline.
AOI_COORDS = [
    [34.72, 25.30],
    [35.20, 25.30],
    [35.20, 23.85],
    [34.72, 23.85],
    [34.72, 25.30],
]

GREEN = "B3"
BLUE = "B2"
SCALE = 10  # metres


def sentinel2_composite(aoi, start: str, end: str, max_cloud: float = 10.0):
    """Median composite of cloud-free Sentinel-2 L2A scenes.

    A median rather than a mean: one bright cloud edge in the stack moves a
    mean and does not move a median.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(_mask_clouds_and_land)
    )
    return collection.median().select([GREEN, BLUE]).divide(10000).clip(aoi)


def _mask_clouds_and_land(image):
    """Keep cloud-free water pixels only.

    Land has to go before anything else: a beach that moved 20 m between two
    composites is a huge signal in a log-ratio and has nothing to do with the
    seabed.
    """
    scl = image.select("SCL")
    cloud_free = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    water = scl.eq(6)
    return image.updateMask(cloud_free.And(water))


def depth_invariant(image, ratio: float, deep_green: float, deep_blue: float):
    """ln(G - Gdeep) - ratio * ln(B - Bdeep), as an ee.Image."""
    green = image.select(GREEN).subtract(deep_green).max(1e-6).log()
    blue = image.select(BLUE).subtract(deep_blue).max(1e-6).log()
    return green.subtract(blue.multiply(ratio)).rename("dii")


def deep_water_values(image, aoi, quantile: int = 1):
    """Low-quantile radiance per band, the stand-in for deep water."""
    stats = image.reduceRegion(
        reducer=ee.Reducer.percentile([quantile]),
        geometry=aoi,
        scale=SCALE * 4,
        maxPixels=1e9,
        bestEffort=True,
    )
    return (
        stats.getNumber(f"{GREEN}_p{quantile}").getInfo(),
        stats.getNumber(f"{BLUE}_p{quantile}").getInfo(),
    )


def attenuation_ratio_from_sand(image, sand_geometry, deep_green, deep_blue) -> float:
    """ki/kj from the variance/covariance of a sand training area.

    ``sand_geometry`` must be sand across a range of depths. Getting this
    polygon wrong is the single easiest way to produce a ratio that is wrong
    everywhere, and nothing downstream will tell you.
    """
    logs = ee.Image.cat(
        image.select(GREEN).subtract(deep_green).max(1e-6).log().rename("x_i"),
        image.select(BLUE).subtract(deep_blue).max(1e-6).log().rename("x_j"),
    )
    stats = logs.reduceRegion(
        reducer=ee.Reducer.covariance(),
        geometry=sand_geometry,
        scale=SCALE,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    matrix = stats["array"]
    var_i, cov_ij = matrix[0][0], matrix[0][1]
    var_j = matrix[1][1]
    a = (var_i - var_j) / (2.0 * cov_ij)
    return a + (a * a + 1.0) ** 0.5


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Earth Engine cloud project id")
    parser.add_argument("--before", default="2019-01-01,2019-12-31")
    parser.add_argument("--after", default="2025-01-01,2025-12-31")
    parser.add_argument(
        "--null",
        default="2025-01-01,2025-02-28;2025-03-01,2025-04-30",
        help="the two windows of the null pair, separated by ';'",
    )
    parser.add_argument("--export", help="export the change mask to this Drive folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    if ee is None:
        print("earthengine-api is not installed. See the README, or run run_demo.py "
              "for the offline version.")
        return 2

    args = build_arguments().parse_args(argv)
    ee.Initialize(project=args.project)

    aoi = ee.Geometry.Polygon([AOI_COORDS])

    before_start, before_end = args.before.split(",")
    after_start, after_end = args.after.split(",")
    null_a_window, null_b_window = args.null.split(";")

    before = sentinel2_composite(aoi, before_start, before_end)
    after = sentinel2_composite(aoi, after_start, after_end)
    null_a = sentinel2_composite(aoi, *null_a_window.split(","))
    null_b = sentinel2_composite(aoi, *null_b_window.split(","))

    deep_green, deep_blue = deep_water_values(before, aoi)

    # Replace this with a real sand polygon for the site before trusting the
    # ratio: a rectangle is a placeholder, not a training sample.
    sand = ee.Geometry.Rectangle([34.80, 25.05, 34.83, 25.09])
    ratio = attenuation_ratio_from_sand(before, sand, deep_green, deep_blue)
    print(f"deep water: green={deep_green:.5f} blue={deep_blue:.5f}")
    print(f"attenuation ratio ki/kj: {ratio:.4f}")

    dii_before = depth_invariant(before, ratio, deep_green, deep_blue)
    dii_after = depth_invariant(after, ratio, deep_green, deep_blue)
    dii_null_a = depth_invariant(null_a, ratio, deep_green, deep_blue)
    dii_null_b = depth_invariant(null_b, ratio, deep_green, deep_blue)

    real_diff = dii_after.subtract(dii_before).rename("diff")
    null_diff = dii_null_b.subtract(dii_null_a).rename("diff")

    # The threshold comes from the null pair, not from the real one.
    null_p99 = null_diff.abs().reduceRegion(
        reducer=ee.Reducer.percentile([99]),
        geometry=aoi,
        scale=SCALE * 2,
        maxPixels=1e9,
        bestEffort=True,
    ).getNumber("diff_p99")

    threshold = null_p99.getInfo()
    print(f"null-calibrated threshold (fpr 1%): |d| >= {threshold:.4f}")

    changed = real_diff.abs().gte(threshold).selfMask().rename("changed")
    null_changed = null_diff.abs().gte(threshold).selfMask()

    def area_km2(mask):
        value = (
            mask.multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=SCALE,
                maxPixels=1e10,
                bestEffort=True,
            )
            .getNumber(mask.bandNames().get(0))
        )
        return value.getInfo() / 1e6

    real_area = area_km2(changed)
    null_area = area_km2(null_changed)

    print(f"real pair : {real_area:.2f} km2")
    print(f"null pair : {null_area:.2f} km2")
    print(f"ratio     : {real_area / null_area:.1f}x" if null_area else "ratio: inf")

    from redsea import detectable

    if detectable(real_area, null_area, margin=2.0):
        print("-> the signal clears its own floor")
    else:
        print("-> below the noise floor. Do not publish this number.")

    if args.export:
        task = ee.batch.Export.image.toDrive(
            image=changed.toByte(),
            description="redsea_change_mask",
            folder=args.export,
            region=aoi,
            scale=SCALE,
            maxPixels=1e10,
        )
        task.start()
        print(f"export task started: {task.id}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
