import xarray as xr
import pandas as pd
import numpy as np

from pycontrails import Flight
from pycontrails.models.ps_model import PSFlight
from pycontrails.ext.bada import BADAFlight

from iagos_toolkit.weather.iagos import create_met_from_iagos
from iagos_toolkit.flight.iagos_fleet import AIRCRAFT_PARS

BADA4_PATH = "/Users/twoldhuis1/Documents/BADA 4.2 pycontrails"
BADA3_PATH = "/Users/twoldhuis1/Documents/bada_315_cfc23ebb2306e5a61b91"


def create_flight_from_adsb(
    adsb_file_path: str,
    aircraft_type: str,
    engine_uid: str | None = None,
    resample: bool = False,
) -> Flight:
    """
    Create a pycontrails Flight object from ADS-B CSV data.

    The function:
    - loads ADS-B position data from a CSV file
    - converts timestamps to pandas datetime
    - converts altitude from feet to meters
    - filters invalid altitude spikes
    - optionally resamples and cleans the trajectory

    Parameters
    ----------
    adsb_file_path : str
        Path to ADS-B CSV file.
    aircraft_type : str
        ICAO aircraft type designator.
    engine_uid : str, optional
        Engine UID used by pycontrails (default: None).
    resample : bool, optional
        If True, clean and resample the flight trajectory
        using `Flight.clean_and_resample()`.

    Returns
    -------
    Flight
        pycontrails Flight object.
    """
    df = pd.read_csv(adsb_file_path)

    # Require valid position
    df = df.dropna(subset=["longitude", "latitude"]).copy()

    # Time handling
    df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Altitude: feet → meters
    altitude_m = 0.3048 * df["geoaltitude"]
    altitude_m = altitude_m.where(altitude_m <= 15_000)
    altitude_m = altitude_m.ffill()

    df["altitude"] = altitude_m
    df = df.drop(columns=["geoaltitude"])

    # Drop rows with invalid time or altitude
    df = df.dropna(subset=["time", "altitude"])

    fl = Flight(
        data=df,
        attrs={
            "flight_id": "1",
            "aircraft_type": aircraft_type,
            "engine_uid": engine_uid,
        },
    )

    return fl.clean_and_resample() if resample else fl


def create_flight_from_iagos(
    iagos_file_path: str,
    aircraft_type: str,
    engine_uid: str | None = None,
    resample: bool = False,
) -> Flight:
    """
    Create a pycontrails Flight object from an IAGOS NetCDF file.

    The function:
    - loads IAGOS trajectory data from NetCDF
    - prefers GPS altitude (`gps_alt_AC`) if available
    - falls back to barometric altitude (`baro_alt_AC`)
    - standardizes coordinates for pycontrails
    - optionally resamples and cleans the trajectory

    Parameters
    ----------
    iagos_file_path : str
        Path to IAGOS NetCDF file.
    aircraft_type : str
        ICAO aircraft type designator.
    engine_uid : str, optional
        Engine UID used by pycontrails (default: None).
    resample : bool, optional
        If True, clean and resample the flight trajectory
        using `Flight.clean_and_resample()`.

    Returns
    -------
    Flight
        pycontrails Flight object.

    Raises
    ------
    ValueError
        If neither GPS nor barometric altitude is present.
    """
    ds = xr.open_dataset(iagos_file_path)
    df = ds.to_dataframe().reset_index()

    # Select altitude source
    if "gps_alt_AC" in df.columns:
        altitude_col = "gps_alt_AC"
    elif "baro_alt_AC" in df.columns:
        altitude_col = "baro_alt_AC"
    else:
        raise ValueError(
            "IAGOS dataset must contain either 'gps_alt_AC' or 'baro_alt_AC'."
        )

    df = df.rename(
        columns={
            "UTC_time": "time",
            "lon": "longitude",
            "lat": "latitude",
            altitude_col: "altitude",
        }
    )

    df = df[["time", "longitude", "latitude", "altitude"]].dropna()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    fl = Flight(
        data=df,
        attrs={
            "flight_id": "1",
            "aircraft_type": aircraft_type,
            "engine_uid": engine_uid,
        },
    )

    return fl.clean_and_resample() if resample else fl



def compute_aircraft_efficiency(
    flight,
    met=None,
    apm: str = "PS",
    sensing_time=None,
):
    """
    Compute aircraft engine efficiency along a flight trajectory.

    The function evaluates an Aircraft Performance Model (APM) using
    pycontrails and returns either:
    - the full engine efficiency time series, or
    - the engine efficiency interpolated to a specific sensing time.

    Parameters
    ----------
    flight : Flight
        pycontrails Flight object.
    met : MetDataset, optional
        Meteorological dataset. If None, ISA conditions are assumed.
    apm : {"PS", "BADA4"}, optional
        Aircraft performance model to use:
        - "PS"    : Poll-Schumann model
        - "BADA4" : EUROCONTROL BADA v4
    sensing_time : datetime-like or str, optional
        Timestamp at which to interpolate engine efficiency.
        If None, the full efficiency time series is returned.

    Returns
    -------
    pandas.Series or float
        - pandas.Series indexed by time if sensing_time is None
        - float with interpolated engine efficiency otherwise

    Raises
    ------
    ValueError
        If an unsupported APM is requested.
    """
    if apm == "PS":
        if met is None: # use ISA conditions
            apm_model = PSFlight(
                fill_low_altitude_with_isa_temperature=True,
                fill_low_altitude_with_zero_wind=True,
            )
        else:  # use given met object
            apm_model = PSFlight(
                met=met,
                fill_low_altitude_with_isa_temperature=True,
                fill_low_altitude_with_zero_wind=True,
            )
    
    elif apm == "BADA4":
        if met is None: # use ISA conditions
            apm_model = BADAFlight(
                fill_low_altitude_with_isa_temperature=True,
                fill_low_altitude_with_zero_wind=True,
                bada3_path=BADA3_PATH,
                bada4_path=BADA4_PATH,
                bada_priority=4
            )
        else:  # use given met object
            apm_model = BADAFlight(
                met=met,
                fill_low_altitude_with_isa_temperature=False,
                fill_low_altitude_with_zero_wind=False,
                bada3_path=BADA3_PATH,
                bada4_path=BADA4_PATH,
                bada_priority=4
                )
    else:
        raise ValueError("APM must be either 'BADA4' or 'PS'")

    output = apm_model.eval(flight)
    time = pd.to_datetime(output.data["time"])
    efficiency = pd.Series(
        output.data["engine_efficiency"],
        index=time,
        name="engine_efficiency",
    )
    
    if sensing_time is None:
        return efficiency

    sensing_time = pd.Timestamp(sensing_time)
    if sensing_time.tzinfo is not None:
        sensing_time = sensing_time.tz_convert(None)

    # Ensure monotonic index (required by time interpolation)
    efficiency = efficiency.sort_index()

    interpolated = (
        efficiency
        .reindex(efficiency.index.union([sensing_time]))
        .interpolate(method="time")
        .loc[sensing_time]
    )

    return float(interpolated)

if __name__ == "__main__":
    # Example Usage
    iagos_file = "/Users/twoldhuis1/Documents/GitHub/iagos-sentinel2-dataset/data/intersects_sentinel/2019/2019011101212214_L1C_T50PRC_A009653_20190111T023626/IAGOS_timeseries_2019011101212214_L2_3.1.2.nc4"
    sensing_time= "2019-01-11 02:43:18.500000+00:00"
    
    icao24 = "8991BE"
    icao_code = AIRCRAFT_PARS.get(icao24, {}).get("icao_code", "A333")
    aircraft_type = AIRCRAFT_PARS.get(icao24, {}).get("airframe_type", "A343")
    engine_uid = AIRCRAFT_PARS.get(icao24, {}).get("engine_uid", None)

    met = create_met_from_iagos(iagos_file)
    flight = create_flight_from_iagos(iagos_file, aircraft_type=icao_code, engine_uid=engine_uid, resample=True)
    
    eff_ps = compute_aircraft_efficiency(flight, met=met, apm="PS")
    eff_bada = compute_aircraft_efficiency(flight, met=met, apm="BADA4")

    eff_ps_val = compute_aircraft_efficiency(flight, met=met, apm="PS", sensing_time=sensing_time)
    eff_bada_val = compute_aircraft_efficiency(flight, met=met, apm="BADA4", sensing_time=sensing_time)
    import matplotlib.pyplot as plt
    plt.plot(eff_ps.index, eff_ps.values, label="PS", color="blue")
    plt.plot(eff_bada.index, eff_bada.values, label="BADA4", color="orange")
    plt.vlines(pd.to_datetime(sensing_time), ymin=0, ymax=0.5, color="green", linestyles="dashed", label="Sensing time")
    plt.scatter(pd.to_datetime(sensing_time), eff_ps_val, color="blue", marker="o", s=100, label="PS @ sensing time")
    plt.scatter(pd.to_datetime(sensing_time), eff_bada_val, color="orange", marker="o", s=100, label="BADA4 @ sensing time")
    plt.ylim(-0.05, 0.5)
    plt.xlabel("Time")
    plt.ylabel("Engine Efficiency")
    plt.legend()
    plt.show()

