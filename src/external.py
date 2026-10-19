"""External climate data: ERA5 / ERA5-Land daily reanalysis via the Open-Meteo Historical Weather API.

Source: Copernicus Climate Change Service ERA5 / ERA5-Land, served by Open-Meteo
(https://open-meteo.com/en/docs/historical-weather-api), data licence CC BY 4.0. No API key needed.

Every feature is a backward-looking window ending at (or before) the death date, so no future information is used.
"""

import time
from pathlib import Path

import pandas as pd
import requests

START, END = "2006-09-01", "2022-12-31"  # 12+ weeks of history before the earliest death

ERA5_VARS = [
    "precipitation_sum",            # mm
    "relative_humidity_2m_mean",    # %
    "dew_point_2m_mean",            # degC
    "soil_moisture_0_to_7cm_mean",  # m3/m3
    "vapour_pressure_deficit_max",  # kPa
]

# Windows as (days before death the window ends, window length in days)
# (0, 28) = the 4 weeks up to the death date, (56, 28) = 8 to 12 weeks before death
RAIN_WINDOWS = {"0_4w": (0, 28), "4_8w": (28, 28), "8_12w": (56, 28)}
MEAN_WINDOWS = {"0_4w": (0, 28), "8_12w": (56, 28)}
RECENT_WINDOW = {"0_4w": (0, 28)}

# (daily column, how to aggregate, windows, feature name prefix)
WINDOW_SPECS = [
    ("precipitation_sum", "sum", RAIN_WINDOWS, "era5_rain"),
    ("relative_humidity_2m_mean", "mean", MEAN_WINDOWS, "rh"),
    ("dew_point_2m_mean", "mean", RECENT_WINDOW, "dew"),
    ("soil_moisture_0_to_7cm_mean", "mean", MEAN_WINDOWS, "soil"),
    ("vapour_pressure_deficit_max", "mean", RECENT_WINDOW, "vpd"),
]


def _add_keys(df):
    """Merge keys: the ERA5-Land grid cell (coordinates rounded to 0.1 degrees) and the death date.

    ERA5-Land is a 0.1-degree grid, so villages in the same cell get the same weather. Fetching per cell
    instead of per village cuts the number of API requests from about 55 to 15.
    """
    df = df.copy()
    df["cell_lat"] = df["latitude"].round(1)
    df["cell_lon"] = df["longitude"].round(1)
    df["date"] = pd.to_datetime(df["deathdate"]).dt.normalize()
    return df


def fetch_cell(lat, lon, variables=ERA5_VARS, retries=10):
    """Daily ERA5 values for one grid cell over the whole date range. Waits and retries if rate-limited."""
    for _ in range(retries):
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": START, "end_date": END,
                "daily": ",".join(variables),
                "timezone": "Africa/Kampala",
                "models": "era5_seamless",  # consistent ERA5 / ERA5-Land across all years
            },
            timeout=120,
        )
        data = response.json()
        if not data.get("error"):
            break
        if "limit" not in data["reason"].lower():
            raise ValueError(data["reason"])
        print(f"\nRate limit hit, waiting a minute: {data['reason']}")
        time.sleep(65)
    else:
        raise RuntimeError("Still rate-limited after several retries. Try again later; finished cells are cached.")

    daily = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["cell_lat"], daily["cell_lon"] = lat, lon
    return daily


def load_era5_daily(train, test, cache_dir="../data/external", pause=1.0, variables=ERA5_VARS, subdir="era5_daily"):
    """Daily ERA5 data for every grid cell used by train/test. Downloads once, then reads from the cache.

    Each cell is cached in its own file, so an interrupted download picks up where it stopped.
    """
    cache_dir = Path(cache_dir) / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Every grid cell used by train or test, once
    all_rows = pd.concat([_add_keys(train), _add_keys(test)])
    cells = all_rows[["cell_lat", "cell_lon"]].drop_duplicates()

    frames = []
    for i, (lat, lon) in enumerate(cells.itertuples(index=False), start=1):
        path = cache_dir / f"cell_{lat}_{lon}.csv"
        if not path.exists():
            daily_cell = fetch_cell(lat, lon, variables)
            daily_cell.to_csv(path, index=False)
            time.sleep(pause)  # be polite to the free API
        frames.append(pd.read_csv(path, parse_dates=["date"]))
        print(f"ERA5 grid cells: {i}/{len(cells)}", end="\r")
    print()
    return pd.concat(frames, ignore_index=True)


def backward_window(values, length, lag, how):
    """Rolling sum or mean over `length` days, shifted so the window ends `lag` days before each date."""
    rolling = values.rolling(length, min_periods=int(length * 0.8))
    if how == "sum":
        result = rolling.sum()
    else:
        result = rolling.mean()
    return result.shift(lag)


def add_window_column(daily, column, how, lag, length, feature):
    """Compute one backward-looking window feature separately for each grid cell (daily must be sorted by date)."""
    feature_values = pd.Series(index=daily.index, dtype=float)
    for _, cell_rows in daily.groupby(["cell_lat", "cell_lon"]):
        feature_values.loc[cell_rows.index] = backward_window(cell_rows[column], length, lag, how)
    daily[feature] = feature_values


def merge_window_features(df, windows):
    """Add window features to train or test by grid cell and death date."""
    keys = ["cell_lat", "cell_lon", "date"]
    merged = _add_keys(df).merge(windows, on=keys, how="left", validate="many_to_one")
    return merged.drop(columns=keys)


def build_window_features(daily):
    """Turn daily values into backward-looking window features (one row per grid cell and day)."""
    daily = daily.sort_values(["cell_lat", "cell_lon", "date"])
    daily = daily.reset_index(drop=True)

    feature_names = []
    for column, how, windows, prefix in WINDOW_SPECS:
        for window_name, (lag, length) in windows.items():
            feature = f"{prefix}_{window_name}"
            add_window_column(daily, column, how, lag, length, feature)
            feature_names.append(feature)
    return daily[["cell_lat", "cell_lon", "date"] + feature_names], feature_names


def add_era5_features(train, test, cache_dir="../data/external"):
    """Add ERA5 window features to train and test. Returns (train, test, list of new feature names)."""
    daily = load_era5_daily(train, test, cache_dir)
    windows, feature_names = build_window_features(daily)
    train = merge_window_features(train, windows)
    test = merge_window_features(test, windows)
    return train, test, feature_names


# ---------- Lagged temperature (5 to 11 weeks before death) ----------

TEMPERATURE_VARS = ["temperature_2m_max", "temperature_2m_mean"]
LAG_START, LAG_END = 35, 77           # days before death: 5 to 11 weeks
MALARIA_TMAX_RANGE = (25.2, 29.9)     # daily max temperature range linked to child malaria deaths at this site


def add_temperature_lag_features(train, test, cache_dir="../data/external"):
    """Add temperature over the window 5 to 11 weeks (days 35-77) before each death.

    - tmax_lag_5_11w:  mean daily max temperature in the window
    - tmean_lag_5_11w: mean daily mean temperature in the window
    - malaria_temp_days_5_11w: number of days in the window with max temperature between 25.2 and 29.9 degC

    Returns (train, test, list of new feature names).
    """
    daily = load_era5_daily(train, test, cache_dir, variables=TEMPERATURE_VARS, subdir="era5_temperature")
    daily = daily.sort_values(["cell_lat", "cell_lon", "date"])
    daily = daily.reset_index(drop=True)

    low, high = MALARIA_TMAX_RANGE
    in_malaria_range = daily["temperature_2m_max"].between(low, high)
    daily["malaria_temp_day"] = in_malaria_range.astype(float)

    # A 43-day window, shifted so it ends 35 days before the death date
    length = LAG_END - LAG_START + 1
    add_window_column(daily, "temperature_2m_max", "mean", LAG_START, length, "tmax_lag_5_11w")
    add_window_column(daily, "temperature_2m_mean", "mean", LAG_START, length, "tmean_lag_5_11w")
    add_window_column(daily, "malaria_temp_day", "sum", LAG_START, length, "malaria_temp_days_5_11w")

    feature_names = ["tmax_lag_5_11w", "tmean_lag_5_11w", "malaria_temp_days_5_11w"]
    windows = daily[["cell_lat", "cell_lon", "date"] + feature_names]
    train = merge_window_features(train, windows)
    test = merge_window_features(test, windows)
    return train, test, feature_names
