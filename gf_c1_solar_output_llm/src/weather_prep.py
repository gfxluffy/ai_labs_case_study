"""
weather_prep.py

Pipeline for Challenge 1 (solar output prediction) - weather-side data prep.

NOTE: This dataset variant has no Date column (just a row-number index),
so there's no way to filter to a specific year or build a day-by-day
table - the data points are simply observations spread across the
~10-year span with no per-row date attached. Per the case study brief,
using an aggregate form of the weather data as regressors is an explicitly
valid option, so this script builds ONLY the aggregated (one row per city)
feature table.

Steps:
1. Load the Kaggle Australia weather CSV.
2. Drop any city with zero non-null values for one or more weather
   features - nothing city-specific to impute from, so the location is
   dropped rather than filled from the global mean.
3. Impute missing daily values using each city's OWN mean for that column.
   Imputing at the city level, before aggregation, matters here: imputing
   with the global mean first would pull every city's stats toward the
   country-wide average, flattening exactly the city-to-city variation
   the model needs to learn from - a real risk with only ~49 rows to
   train on.
4. Build aggregated features per city: mean/min/max of each weather
   variable across all available rows for that city.
5. Join to city_solar_output.csv (produced by solar_extraction.py) on city.
6. Save a final modeling-ready table.

Usage:
    python weather_prep.py

"""

from pathlib import Path

import numpy as np
import pandas as pd
import configparser

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_wp_cfg = cfg["weather_prep"]

WEATHER_CSV_PATH = _wp_cfg["weather_csv_path"]
SOLAR_CSV_PATH = _wp_cfg["solar_csv_path"]
CITY_COLUMN = _wp_cfg["city_column"]

WEATHER_FEATURE_COLUMNS = [c.strip() for c in _wp_cfg["weather_feature_columns"].split(",")]
SELECT_FEATURE_COLUMNS = [c.strip() for c in _wp_cfg["select_feature_columns"].split(",")]

AGGREGATED_OUTPUT_PATH = _wp_cfg["aggregated_output_path"]


# ---------------------------------------------------------------------------
# Step 1: Load weather CSV
# ---------------------------------------------------------------------------

def load_weather(csv_path, city_col=CITY_COLUMN):
    """Load the weather CSV and print basic coverage diagnostics per city
    (row counts vary by city since there's no shared date range to filter
    to - some stations simply have more/fewer observations)."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows across {df[city_col].nunique()} cities")

    coverage = df.groupby(city_col).size().sort_values()
    print("\nRow count per city (lowest 5 shown - check these aren't too thin):")
    print(coverage.head())

    return df


# ---------------------------------------------------------------------------
# Step 2: Impute missing values, per-city first
# ---------------------------------------------------------------------------

def check_missingness(weather_df, feature_columns=WEATHER_FEATURE_COLUMNS):
    """Print missingness % per feature so you can decide, BEFORE imputing,
    whether a feature is sparse enough that imputing it would mostly be
    inventing data rather than filling small gaps (e.g. this dataset's
    Sunshine/Evaporation columns are often missing 40%+ of rows). If a
    feature you're relying on (see FEATURES in model_training.py) is
    missing heavily, consider dropping it instead of imputing it."""
    missing_pct = weather_df[feature_columns].isna().mean().sort_values(ascending=False) * 100
    print("Missingness per feature (%):")
    print(missing_pct.round(1))
    return missing_pct


def drop_zero_coverage_locations(weather_df, feature_columns=SELECT_FEATURE_COLUMNS,
                                  city_col=CITY_COLUMN):
    """Drop any city that has zero non-null values for one or more feature
    columns. A city with no observations at all for a feature has nothing
    city-specific to impute from - filling it via global mean would be
    inventing that city's value outright rather than filling a gap, so we
    drop the location instead of imputing it."""
    non_null_counts = weather_df.groupby(city_col)[feature_columns].count()
    zero_coverage = non_null_counts.eq(0)
    bad_cities = zero_coverage[zero_coverage.any(axis=1)]

    if len(bad_cities):
        for city, row in bad_cities.iterrows():
            missing_cols = row[row].index.tolist()
            print(f"  Dropping {city}: zero non-null values for {missing_cols}")
        weather_df = weather_df[~weather_df[city_col].isin(bad_cities.index)].copy()
        print(f"  Dropped {len(bad_cities)} city(ies) with zero coverage; "
              f"{weather_df[city_col].nunique()} cities remain")
    else:
        print("  No cities with zero non-null values for any feature - none dropped")

    return weather_df


def impute_features(weather_df, feature_columns=WEATHER_FEATURE_COLUMNS, city_col=CITY_COLUMN):
    """Fill missing values using each city's OWN mean for that column first.
    Falls back to the global mean only for the rare case where a city has
    zero non-null values for a column (nothing city-specific to average
    from) - this fallback should be unreachable if drop_zero_coverage_locations()
    ran first, but is kept as a safety net. Doing this per-city, before
    aggregation, preserves the city-to-city variation the model is trying
    to learn - global-mean imputation would flatten it."""
    df = weather_df.copy()
    for col in feature_columns:
        n_before = df[col].isna().sum()
        if n_before == 0:
            continue
        df[col] = df.groupby(city_col)[col].transform(lambda s: s.fillna(s.mean()))
        n_after_city_fill = df[col].isna().sum()
        if n_after_city_fill:
            print(f"  {col}: {n_after_city_fill} rows had no city-level data to impute from - "
                  f"using global mean as fallback")
            df[col] = df[col].fillna(df[col].mean())
        print(f"  {col}: imputed {n_before} missing values")
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregated features - one row per city
# ---------------------------------------------------------------------------

def build_aggregated_features(weather_df, feature_columns=WEATHER_FEATURE_COLUMNS,
                               city_col=CITY_COLUMN):
    """Collapse all observations into one row per city: mean, min, max for
    each feature column. Assumes weather_df has already been through
    impute_features() - if not, missing values are ignored per-column
    (pandas default) and may produce NaN aggregates for sparse cities."""
    missing_cols = [c for c in feature_columns if c not in weather_df.columns]
    if missing_cols:
        raise ValueError(
            f"These columns aren't in the CSV: {missing_cols}. "
            f"Run df.columns.tolist() and update WEATHER_FEATURE_COLUMNS to match."
        )

    agg_funcs = ["mean", "min", "max"]
    grouped = weather_df.groupby(city_col)[feature_columns].agg(agg_funcs)

    # Flatten multi-level columns: ('MinTemp', 'mean') -> 'MinTemp_mean'
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    n_missing = grouped.isna().sum().sum()
    if n_missing:
        print(f"\nWARNING: aggregated table still has {n_missing} missing values - this "
              f"shouldn't happen if impute_features() ran first. Check that this function "
              f"was called on an already-imputed weather_df.")

    return grouped


# ---------------------------------------------------------------------------
# Step 3: Join with solar output table
# ---------------------------------------------------------------------------

def join_with_solar(weather_features_df, solar_csv_path=SOLAR_CSV_PATH,
                     city_col=CITY_COLUMN):
    """Join aggregated weather features to the per-city solar output table."""
    solar_df = pd.read_csv(solar_csv_path)

    if "status" in solar_df.columns:
        n_before = len(solar_df)
        solar_df = solar_df[solar_df["status"] == "ok"]
        n_dropped = n_before - len(solar_df)
        if n_dropped:
            print(f"\nDropping {n_dropped} cities from the join - solar extraction "
                  f"did not succeed for them (see status column in {solar_csv_path}).")

    solar_df = solar_df[["city", "solar_output"]].rename(columns={"city": city_col})
    solar_df[city_col] = solar_df[city_col].replace(r'\s+', '', regex=True)
    print(solar_df[city_col].head(10))
    
    merged = weather_features_df.merge(solar_df, on=city_col, how="inner")

    n_weather_cities = weather_features_df[city_col].nunique()
    n_merged_cities = merged[city_col].nunique()
    if n_merged_cities < n_weather_cities:
        dropped_cities = set(weather_features_df[city_col].unique()) - set(merged[city_col].unique())
        print(f"\nNote: {n_weather_cities - n_merged_cities} weather-dataset cities had no "
              f"matching solar value and were dropped: {dropped_cities}")
        print("Check for name mismatches between the weather CSV's city names and the "
              "'city' column in city_solar_output.csv (e.g. 'MelbourneAirport' vs 'Melbourne').")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 1: Loading weather data...")
    weather_df = load_weather(WEATHER_CSV_PATH)

    print("\nStep 2: Checking missingness...")
    check_missingness(weather_df)

    print("\nStep 3: Dropping cities with zero non-null values for any feature...")
    weather_df = drop_zero_coverage_locations(weather_df)

    print("\nStep 4: Imputing missing values (per-city mean, global fallback)...")
    weather_df = impute_features(weather_df)

    print("\nStep 5: Building aggregated (one row per city) feature table...")
    agg_features = build_aggregated_features(weather_df)

    print("\nStep 6: Joining with solar output data...")
    final_df = join_with_solar(agg_features)

    final_df.to_csv(AGGREGATED_OUTPUT_PATH, index=False)
    print(f"\nSaved modeling table ({final_df.shape[0]} rows, {final_df.shape[1]} columns) "
          f"to {AGGREGATED_OUTPUT_PATH}")
    print(final_df.head())


if __name__ == "__main__":
    main()
