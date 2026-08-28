"""
solar_extraction.py

Pipeline for Challenge 1 (solar output prediction) - solar data prep step:

1. Geocode Kaggle weather-dataset city names to lat/lon (with caching).
2. Build a metric-CRS buffer polygon around each city.
3. Clip the Global Solar Atlas GeoTIFF to that polygon and average the pixels.
4. Sanity-check the result visually for a couple of cities before running
   the full batch.

Usage:
    python solar_extraction.py

"""

import configparser
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Point, Polygon, mapping
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')
_se_cfg = cfg["solar_extraction"]

TIF_PATH = _se_cfg["tif_path"]
COORDS_CACHE_PATH = _se_cfg["coords_cache_path"]
OUTPUT_PATH = _se_cfg["output_path"]

METRIC_CRS = _se_cfg["metric_crs"]      # Australian Albers Equal Area (meters)
WGS84 = _se_cfg["wgs84_crs"]            # standard lat/lon
BUFFER_RADIUS_M = int(_se_cfg["buffer_radius_m"])   # 15 km — documented assumption, adjust/justify as needed

CITY_NAMES = [c.strip() for c in _se_cfg["city_names"].split(",")]


# ---------------------------------------------------------------------------
# Step 1: Geocode city names -> lat/lon (cached to CSV)
# ---------------------------------------------------------------------------

# Rough bounding box for mainland Australia + Tasmania, used to sanity-check
# geocoding results. Geocoders can silently return a match outside Australia
# for ambiguous city names (e.g. a "Richmond" or "Perth" elsewhere in the
# world) — anything outside this box is almost certainly a bad match.
AUSTRALIA_BOUNDS = {"lat_min": -44.0, "lat_max": -10.0, "lon_min": 112.0, "lon_max": 168.0}


def geocode_cities(city_names, cache_path=COORDS_CACHE_PATH, country="Australia"):
    """Geocode a list of city names, caching results to CSV so repeated runs
    don't re-hit the geocoding API. Returns a DataFrame with city, lat, lon.
    Rows that failed to geocode, or that geocoded to a point outside
    Australia's bounding box (a likely wrong match), will have NaN lat/lon —
    check and fix these manually (e.g. station names like 'MelbourneAirport'
    may need cleanup, or an ambiguous name needs a more specific query)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached coordinates from {cache_path}")
        return pd.read_csv(cache_path)

    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="solar_case_study")
    records = []
    for city in city_names:
        try:
            # country_codes restricts results to Australia; exactly_one avoids
            # ambiguity. This alone doesn't guarantee a correct match for
            # small/ambiguous names, hence the bounding-box check below.
            location = geolocator.geocode(
                f"{city}, {country}", country_codes="au", exactly_one=True
            )
            if location and _in_australia(location.latitude, location.longitude):
                records.append({"city": city, "lat": location.latitude, "lon": location.longitude})
            elif location:
                print(f"  SUSPECT match for {city}: ({location.latitude}, {location.longitude}) "
                      f"is outside Australia's bounding box — flagging for manual fix")
                records.append({"city": city, "lat": np.nan, "lon": np.nan})
            else:
                print(f"  No result for: {city}")
                records.append({"city": city, "lat": np.nan, "lon": np.nan})
        except Exception as e:
            print(f"  Failed for {city}: {e}")
            records.append({"city": city, "lat": np.nan, "lon": np.nan})
        time.sleep(1)  # respect Nominatim's 1 request/second usage policy

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"Saved coordinates to {cache_path}")
    n_missing = df["lat"].isna().sum()
    if n_missing:
        print(f"\n{n_missing} cities need manual coordinate fixes in {cache_path} "
              f"before running extraction — edit the CSV directly, or re-run geocoding "
              f"with a more specific query for just those rows.")
    return df


def _in_australia(lat, lon, bounds=AUSTRALIA_BOUNDS):
    return (bounds["lat_min"] <= lat <= bounds["lat_max"]
            and bounds["lon_min"] <= lon <= bounds["lon_max"])


# ---------------------------------------------------------------------------
# Step 2: Build a metric-CRS buffer polygon per city, in the raster's CRS
# ---------------------------------------------------------------------------

def city_polygon_in_raster_crs(lon, lat, radius_m, raster_crs,
                                metric_crs=METRIC_CRS, wgs84=WGS84):
    """Build a circular buffer of `radius_m` meters around (lon, lat),
    correctly in a metric CRS, then reproject the polygon boundary into
    the raster's CRS so it can be used to clip that raster."""
    to_metric = Transformer.from_crs(wgs84, metric_crs, always_xy=True)
    to_raster_crs = Transformer.from_crs(metric_crs, raster_crs, always_xy=True)

    x, y = to_metric.transform(lon, lat)
    circle_metric = Point(x, y).buffer(radius_m)
    coords = list(circle_metric.exterior.coords)
    reproj_coords = [to_raster_crs.transform(px, py) for px, py in coords]
    return Polygon(reproj_coords)


# ---------------------------------------------------------------------------
# Step 3: Clip raster to polygon and average valid pixels
# ---------------------------------------------------------------------------

def average_solar_value(tif_path, polygon):
    """Clip the raster to `polygon` and return the mean of valid pixels.
    Returns (value, status) where status is 'ok', 'no_overlap' (polygon
    doesn't intersect the raster at all — usually a bad geocode or a point
    genuinely outside the raster's coverage), or 'nodata_only' (overlaps,
    but every pixel inside is masked/nodata, e.g. open ocean)."""
    with rasterio.open(tif_path) as src:
        geom = [mapping(polygon)]
        try:
            out_image, _ = mask(src, geom, crop=True, nodata=src.nodata)
        except ValueError:
            # "Input shapes do not overlap raster" — the polygon is entirely
            # outside the raster's extent. Don't crash the batch; flag it.
            return np.nan, "no_overlap"

        data = out_image[0]  # first band
        valid = data[data != src.nodata] if src.nodata is not None else data.flatten()
        valid = valid[~np.isnan(valid)]
        if valid.size == 0:
            return np.nan, "nodata_only"
        return float(valid.mean()), "ok"


def extract_all_cities(tif_path, cities_df, radius_m=BUFFER_RADIUS_M):
    """Run the polygon-build + clip-and-average steps for every city in
    cities_df (expects columns: city, lat, lon). Skips rows with missing
    coordinates. Returns a DataFrame with city, solar_output, status —
    check the status column to see *why* any NaNs occurred."""
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
    print(f"  Raster CRS: {raster_crs}")
    print(f"  Raster bounds: {raster_bounds}")

    results = []
    for _, row in cities_df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            print(f"  Skipping {row['city']} — no coordinates")
            results.append({"city": row["city"], "solar_output": np.nan, "status": "no_coords"})
            continue
        poly = city_polygon_in_raster_crs(row["lon"], row["lat"], radius_m, raster_crs)
        val, status = average_solar_value(tif_path, poly)
        if status != "ok":
            print(f"  {row['city']}: {status} (lat={row['lat']}, lon={row['lon']})")
        results.append({"city": row["city"], "solar_output": val, "status": status})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 4: Visual sanity check for a handful of cities before trusting the batch
# ---------------------------------------------------------------------------

def sanity_check_plot(tif_path, lon, lat, city_name, radius_m=BUFFER_RADIUS_M,
                       save_dir="../output/sanity_checks"):
    """Plot the city's buffer polygon over the full raster, and a zoomed
    view of the clipped region, so you can visually confirm the geometry
    lines up before running the extraction over all cities."""
    import matplotlib.pyplot as plt
    import geopandas as gpd

    Path(save_dir).mkdir(exist_ok=True)

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        polygon = city_polygon_in_raster_crs(lon, lat, radius_m, raster_crs)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        show(src, ax=axes[0], cmap="viridis")
        gpd.GeoSeries([polygon]).boundary.plot(ax=axes[0], edgecolor="red", linewidth=2)
        axes[0].set_title(f"{city_name} — location on full raster")

        geom = [mapping(polygon)]
        out_image, _ = mask(src, geom, crop=True, nodata=src.nodata)
        axes[1].imshow(out_image[0], cmap="viridis")
        axes[1].set_title(f"{city_name} — clipped region (zoomed)")

        plt.tight_layout()
        out_path = Path(save_dir) / f"sanity_check_{city_name}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 1: Geocoding cities...")
    cities_df = geocode_cities(CITY_NAMES)

    missing = cities_df[cities_df["lat"].isna()]
    if not missing.empty:
        print(f"\nWARNING: {len(missing)} cities failed to geocode — fix manually in "
              f"{COORDS_CACHE_PATH} before continuing:")
        print(missing["city"].tolist())

    print("\nStep 2: Running sanity checks on a couple of cities...")
    sample = cities_df.dropna(subset=["lat", "lon"]).head(2)
    for _, row in sample.iterrows():
        sanity_check_plot(TIF_PATH, row["lon"], row["lat"], row["city"])

    print("\nStep 3: Extracting average solar value for all cities...")
    solar_df = extract_all_cities(TIF_PATH, cities_df)
    
    nan_rows = solar_df[solar_df["solar_output"].isna()]
    if not nan_rows.empty:
        print(f"\nWARNING: {len(nan_rows)} cities returned NaN — see status column for why:")
        print(nan_rows[["city", "status"]])
        print("\n  'no_overlap'   -> polygon entirely outside raster; check the city's "
              "geocoded lat/lon in city_coords.csv (likely a wrong/ambiguous geocode)")
        print("  'nodata_only'  -> polygon overlaps the raster but every pixel inside is "
              "masked (e.g. coastal city with buffer mostly over ocean); try a smaller radius")
        print("  'no_coords'    -> geocoding failed or was flagged as outside Australia; "
              "fix manually in city_coords.csv")

    solar_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved final table to {OUTPUT_PATH}")
    print(solar_df)


if __name__ == "__main__":
    main()
