from datetime import datetime, timedelta
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails import MetDataset
from pycontrails.datalib.ecmwf import ERA5ARCO

from comop.meteo.isa import get_p_ISA

from iagos_toolkit.weather.thermo import calc_rh_from_specific_humidity, calc_rhi_from_specific_humidity


def download_arco_era5_data(
    air_pressure: float,
    sensing_time,
    output_file: str,
    variables=("t", "q", "u", "v"),
) -> xr.Dataset:
    """
    Download ARCO ERA5 pressure-level data around a sensing time and altitude.

    Parameters
    ----------
    altitude_m : float
        Aircraft altitude in meters.
    sensing_time : datetime-like
        Target sensing time.
    output_file : str
        Path to output NetCDF file.
    variables : tuple[str]
        ERA5 variables to download.

    Returns
    -------
    xarray.Dataset
        Loaded ERA5 dataset.
    """
    sensing_time = pd.Timestamp(sensing_time).tz_convert(None)

    hour_before = sensing_time.replace(minute=0, second=0, microsecond=0)
    hour_after = hour_before + timedelta(hours=1)

    # Convert altitude → pressure → pressure level
    target_pressure_pa = get_p_ISA(air_pressure)
    target_pressure_hpa = round(target_pressure_pa / 100)

    # Load ERA5 model level table
    model_level_path = (
        Path(pycontrails.__file__).parent
        / "datalib/ecmwf/static/model_level_dataframe_v20240418.csv"
    )
    model_level_df = pd.read_csv(model_level_path)

    above = model_level_df[model_level_df["pf [hPa]"] >= target_pressure_hpa].iloc[0]
    below = model_level_df[model_level_df["pf [hPa]"] < target_pressure_hpa].iloc[-1]

    pressure_levels = [
        round(above["pf [hPa]"]),
        round(below["pf [hPa]"]),
    ]

    era5 = ERA5ARCO(
        time=[hour_before, hour_after],
        variables=list(variables),
        pressure_levels=pressure_levels,
        cachestore=None,
    )

    met = era5.open_metdataset()
    met.data.load()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    met.data.to_netcdf(output_file, format="NETCDF4")

    return met.data


def get_arco_era5_data(
    air_pressure: float,
    longitude: float,
    latitude: float,
    sensing_time,
    local_file: str | None = None,
) -> dict:
    """
    Retrieve and interpolate ARCO ERA5 data to a single space-time point.

    If a local ERA5 file exists it is reused, otherwise the data is downloaded.

    Parameters
    ----------
    altitude_m : float
        Aircraft altitude in meters.
    longitude : float
        Longitude in degrees.
    latitude : float
        Latitude in degrees.
    sensing_time : datetime-like
        Target sensing time.
    local_file : str or None
        Optional ERA5 NetCDF file path.

    Returns
    -------
    dict
        Dictionary with interpolated ERA5 variables and metadata.
    """
    sensing_time = pd.Timestamp(sensing_time).tz_convert(None)
    target_time = np.datetime64(sensing_time)

    # Determine local file
    if local_file is not None and Path(local_file).exists():
        ds = xr.open_dataset(local_file)
        era5_filename = Path(local_file).name
    else:
        if local_file is None:
            raise ValueError("local_file must be provided if download is required")

        ds = download_arco_era5_data(
            air_pressure=air_pressure,
            sensing_time=sensing_time,
            output_file=local_file,
        )
        era5_filename = Path(local_file).name

    # Compute pressure level
    target_pressure_pa = air_pressure
    target_pressure_hpa = target_pressure_pa / 100

    # Compute relative humidity fields
    ds["relative_humidity_liquid"] = xr.apply_ufunc(
        calc_rh_from_specific_humidity,
        ds["specific_humidity"],
        ds["air_temperature"],
        ds["air_pressure"],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    ds["relative_humidity_ice"] = xr.apply_ufunc(
        calc_rhi_from_specific_humidity,
        ds["specific_humidity"],
        ds["air_temperature"],
        ds["air_pressure"],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Interpolate
    interp = ds.interp(
        longitude=longitude,
        latitude=latitude,
        time=target_time,
        level=target_pressure_hpa,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Convert to dictionary
    result = {
        var: interp[var].item()
        for var in interp.data_vars
    }
    result["air_pressure"] = interp["air_pressure"].item()
    result["era5_filename"] = era5_filename

    # Replace NaN with None
    for k, v in result.items():
        if isinstance(v, float) and math.isnan(v):
            result[k] = None

    return result


def create_met_from_era5(era5_path: str, max_altitude: float = None, min_altitude: float = None, extrapolate_time: float = 0) -> MetDataset:
    # Load ERA5 meteorological dataset
    era5_data = xr.open_dataset(era5_path)

    if min_altitude is not None and max_altitude is not None:
        # Compute min and max pressure (in hPa) from flight altitudes using ISA model
        min_pressure = round(get_p_ISA(max_altitude) / 100) - 1
        max_pressure = round(get_p_ISA(min_altitude) / 100) + 1

        # Get existing pressure levels in the dataset
        existing_levels = era5_data.level.values

        # Append min and max pressure if they are outside the existing range
        new_levels = np.unique(np.append(existing_levels, [min_pressure, max_pressure]))
        new_levels.sort()

        # Interpolate/extrapolate dataset along the level axis
        era5_data = era5_data.interp(level=new_levels, method="linear", kwargs={"fill_value": "extrapolate"})

    if extrapolate_time != 0:
        # Convert to timedelta
        delta = timedelta(hours=abs(extrapolate_time))
        times = era5_data.time.values
        if extrapolate_time < 0:
            new_time = np.append([times[0] - np.timedelta64(delta)], times)
        elif extrapolate_time > 0:
            new_time = np.append(times, [times[-1] + np.timedelta64(delta)])

        new_time = np.sort(np.unique(new_time))

        # Interpolate along time axis
        era5_data = era5_data.interp(time=new_time, method="linear", kwargs={"fill_value": "extrapolate"})

    # Wrap ERA5 data in a MetDataset object
    return MetDataset(era5_data)


if __name__ == "__main__":
    # Example usage
    res = get_arco_era5_data(
        air_pressure=18750.0,
        longitude=120.28216250000001,
        latitude=16.182412499999998,
        sensing_time="2019-01-11 02:43:18.500000+00:00",
        local_file="/Users/twoldhuis1/Documents/GitHub/iagos-sentinel2-dataset/data/intersects_sentinel/2019/2019011101212214_L1C_T50PRC_A009653_20190111T023626/arco-era5.nc4",
    )

    print(res)

    #     "era5_data": {
    #     "air_temperature": 216.8190776479005,
    #     "specific_humidity": 4.231288111692186e-05,
    #     "eastward_wind": 8.192067389594978,
    #     "northward_wind": 13.329193219918627,
    #     "relative_humidity_liquid": 0.41529307911849045,
    #     "relative_humidity_ice": 0.7576245420866066,
    #     "air_pressure": 18750.0,
    #     "filename": "arco-era5.nc4"
    # },