import math
import numpy as np
import pandas as pd
import xarray as xr

from pycontrails import MetDataset



def get_iagos_measurements(
    iagos_file_path: str,
    sensing_time,
) -> dict | None:
    """
    Extract interpolated IAGOS measurements at a given sensing time.

    The function loads an IAGOS NetCDF file,
    and interpolates all data variables to the requested sensing time.
    Coordinate variables (latitude, longitude, barometric altitude) are
    included if available.

    Parameters
    ----------
    iagos_file_path : str
        Path to the IAGOS NetCDF file.
    sensing_time : str or pandas.Timestamp or datetime-like
        Target sensing time at which measurements are interpolated.
        Timezone-aware timestamps are converted to naive UTC.

    Returns
    -------
    dict or None
        Dictionary mapping variable names to interpolated scalar values.
        Missing values are returned as None.
        Returns None if the file is invalid or interpolation fails.
    """
    try:
        ds = xr.open_dataset(iagos_file_path)
    except Exception:
        return None

    if "UTC_time" not in ds.coords:
        return None

    # Convert sensing time to pandas.Timestamp
    sensing_time = pd.Timestamp(sensing_time)

    # Remove timezone information (assume UTC)
    if sensing_time.tzinfo is not None:
        sensing_time = sensing_time.tz_convert(None)

    target_time = np.datetime64(sensing_time)

    # Remove duplicate UTC_time entries (required for interpolation)
    utc_times = ds["UTC_time"].values
    _, unique_indices = np.unique(utc_times, return_index=True)
    ds = ds.isel(UTC_time=np.sort(unique_indices))

    try:
        interpolated = ds.interp(UTC_time=target_time)
    except Exception:
        return None

    result: dict[str, float | None] = {}

    # Add interpolated data variables
    for var in interpolated.data_vars:
        value = interpolated[var].item()
        result[var] = None if _is_nan(value) else value

    # Add selected coordinate variables if present
    for coord in ("lat", "lon", "baro_alt_AC"):
        if coord in interpolated.coords:
            value = interpolated[coord].item()
            result[coord] = None if _is_nan(value) else value

    return result


def get_iagos_measurements_at_location(
    iagos_file_path: str,
    longitude: float,
    latitude: float
) -> dict | None:
    """
    Extract IAGOS measurements at the point closest to a given longitude and latitude.

    Parameters
    ----------
    iagos_file_path : str
        Path to the IAGOS NetCDF file.
    longitude : float
        Target longitude in degrees (WGS84).
    latitude : float
        Target latitude in degrees (WGS84).

    Returns
    -------
    dict or None
        Dictionary mapping variable names to values at the closest point.
        Missing values are returned as None.
        Returns None if the file is invalid or coordinates are missing.
    """
    try:
        ds = xr.open_dataset(iagos_file_path)
    except Exception:
        return None

    # Must have coordinates
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None

    # Compute distance squared to all points
    lons = ds["lon"].values
    lats = ds["lat"].values
    dist2 = (lons - longitude)**2 + (lats - latitude)**2

    # Find the index of the closest point
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)

    # Extract values at that index
    result: dict[str, float | None] = {}

    for var in ds.data_vars:
        value = ds[var].isel(**{dim: idx_dim for dim, idx_dim in zip(ds[var].dims, idx)})
        # Convert to Python scalar
        try:
            value = value.item()
        except Exception:
            value = None
        result[var] = None if _is_nan(value) else value

    # Include coordinates
    for coord in ("lat", "lon", "baro_alt_AC"):
        if coord in ds.coords:
            value = ds[coord].isel(**{dim: idx_dim for dim, idx_dim in zip(ds[coord].dims, idx)})
            try:
                value = value.item()
            except Exception:
                value = None
            result[coord] = None if _is_nan(value) else value

    return result


def _is_nan(value) -> bool:
    """Return True if value is a NaN float."""
    return isinstance(value, float) and math.isnan(value)


# Broadcast 1D time series over a 3D grid
def broadcast_to_grid(series, lon_grid, lat_grid):
    return np.tile(series.values, (len(lon_grid), len(lat_grid), 1)).reshape(
        len(lon_grid), len(lat_grid), len(series))


def create_met_from_iagos(iagos_file_path: str) -> MetDataset:
    # Load IAGOS NetCDF dataset
    ds = xr.open_dataset(iagos_file_path)

    # Convert xarray dataset to DataFrame and flatten index
    df_raw = ds.to_dataframe().reset_index()

    # if "air_temp_P1" in df_raw.columns:
    #     df_raw = df_raw.rename(columns={"air_temp_P1": "air_temperature"})
    # elif "air_temp_AC" in df_raw.columns:
    #     df_raw = df_raw.rename(columns={"air_temp_AC": "air_temperature"})
    # else:
    #     raise ValueError("No air temperature column found in IAGOS data")

    df_raw = df_raw.rename(columns={"air_temp_AC": "air_temperature"})

    # Standardize column names
    df_raw = df_raw.rename(columns={
        "UTC_time": "time",
        "lon": "longitude",
        "lat": "latitude",
        "gps_alt_AC": "altitude",  # altitude in meters
        "air_press_AC": "pressure",
        "zon_wind_AC": "eastward_wind",
        "mer_wind_AC": "northward_wind"
    })

    # Keep only the required columns and drop rows with any missing values
    df_raw = df_raw[[
        "time", "longitude", "latitude", "altitude",
        "air_temperature", "pressure", "eastward_wind", "northward_wind"
    ]].dropna()

    # Prepare meteorological DataFrame for gridding
    df_met = df_raw.copy()
    df_met["time"] = pd.to_datetime(df_met["time"])

    df_met = df_met.set_index("time").resample("1min").mean()
    
    # Define coarse spatial grid
    lon_grid = np.arange(-180, 180, 5.0)
    lat_grid = np.arange(-90, 90, 5.0)
    time_axis = df_met.index.to_numpy()

    # Create the grids with met values
    temp_grid = broadcast_to_grid(df_met["air_temperature"], lon_grid, lat_grid)
    u_wind_grid = broadcast_to_grid(df_met["eastward_wind"], lon_grid, lat_grid)
    v_wind_grid = broadcast_to_grid(df_met["northward_wind"], lon_grid, lat_grid)
    
    # Define pressure levels. Currently just 2 with the exact same data
    levels = np.array([1000, 100])

    # Duplicate the data along the new level axis
    temp_grid_4d = np.stack([temp_grid, temp_grid], axis=0)  # shape: (level, lon, lat, time)
    u_wind_grid_4d = np.stack([u_wind_grid, u_wind_grid], axis=0)
    v_wind_grid_4d = np.stack([v_wind_grid, v_wind_grid], axis=0)

    # Wrap variables into DataArrays
    da_temp = xr.DataArray(
        data=temp_grid_4d,
        dims=["level", "longitude", "latitude", "time"],
        coords={"level": levels, "longitude": lon_grid, "latitude": lat_grid, "time": time_axis},
        name="air_temperature"
    )

    da_u = xr.DataArray(
        data=u_wind_grid_4d,
        dims=["level", "longitude", "latitude", "time"],
        coords={"level": levels, "longitude": lon_grid, "latitude": lat_grid, "time": time_axis},
        name="eastward_wind"
    )

    da_v = xr.DataArray(
        data=v_wind_grid_4d,
        dims=["level", "longitude", "latitude", "time"],
        coords={"level": levels, "longitude": lon_grid, "latitude": lat_grid, "time": time_axis},
        name="northward_wind"
    )
    # Combine all meteorological fields into a dataset
    ds_met = xr.Dataset({
        "air_temperature": da_temp,
        "eastward_wind": da_u,
        "northward_wind": da_v
    })

    # Wrap into MetDataset object
    return MetDataset(ds_met)


if __name__ == "__main__":
    # Example usage
    iagos_file = "/Users/twoldhuis1/Documents/GitHub/iagos-sentinel2-dataset/data/intersects_sentinel/2019/2019011101212214_L1C_T50PRC_A009653_20190111T023626/IAGOS_timeseries_2019011101212214_L2_3.1.2.nc4"
    sensing_time= "2019-01-11 02:43:18.500000+00:00"
    measurements = get_iagos_measurements(iagos_file, sensing_time)
    print(measurements)