"""
Implements the ICAO Standard Atmosphere model, up to 80 km altitude.

Note: This file is copied from CoMoP (https://gitlab.tudelft.nl/ace/CoMOP/-/tree/main?ref_type=heads)

Reference:
Manual of the ICAO Standard Atmosphere - extended to 80 kilometres
 / 262,500 feet (Doc 7488)
"""
import numpy as np

import iagos_toolkit.weather.constants as constants


# Sea-level standard temperature, K, see Table A from ref
T0_ISA = 288.15  
# Sea-level standard pressure in Pascals, see Table A from ref
P0_ISA = 101325  

# Altitude boundaries for atmospheric layers (in meters)
TROPOPAUSE_ISA = 11e3
STRATOSPHERE_START_ISA = 20e3
STRATOSPHERE_KINK_ISA = 32e3
STRATOPAUSE_ISA = 47e3
MESOSPHERE_ISA = 51e3
MESOSPHERE_KINK_ISA = 71e3

# Temperature lapse rates (K/m), see Table D from ref
TROPOSPHERE_LAPSE_RATE = 6.5e-3
STRATOSPHERE_LAPSE_RATE_1 = -1e-3
STRATOSPHERE_LAPSE_RATE_2 = -2.8e-3
MESOSPHERE_LAPSE_RATE_1 = 2.8e-3
MESOSPHERE_LAPSE_RATE_2 = 2.0e-3

MAX_ALTITUDE = 80e3

def get_p_ISA_layer(z, p0, lapse_rate, bottom_alt, bottom_T):
    """
    Calculate the pressure at altitude z within an atmospheric layer.

    Parameters:
        z (float or np.ndarray): Altitude at which to compute the pressure (m)
        p0 (float): Pressure at the bottom of the layer (Pa)
        lapse_rate (float): Temperature lapse rate (K/m) in the layer
        bottom_alt (float): Altitude of the bottom of the layer (m)
        bottom_T (float): Temperature at the bottom of the layer (K)

    Returns:
        float or np.ndarray: Pressure at altitude z (Pa)
    """
    if lapse_rate == 0.0:
        return p0 * np.exp(-constants.g0 / (constants.R_air * bottom_T) * (z - bottom_alt))
    else:
        return p0 * (1 - lapse_rate * (z - bottom_alt) / bottom_T) ** (constants.g0 / (constants.R_air * lapse_rate))
    
def get_T_ISA_layer(z, T0, lapse_rate, bottom_alt):
    """
    Calculate the temperature at altitude z within an atmospheric layer.

    Parameters:
        z (float or np.ndarray): Altitude at which to compute the temperature (m)
        T0 (float): Temperature at the bottom of the layer (K)
        lapse_rate (float): Temperature lapse rate (K/m) in the layer
        bottom_alt (float): Altitude of the bottom of the layer (m)

    Returns:
        float or np.ndarray: Temperature at altitude z (K)
    """
    return T0 - lapse_rate * (z - bottom_alt)


# Derived temperature values at key altitudes
T_TROPOPAUSE = get_T_ISA_layer(TROPOPAUSE_ISA, T0_ISA, TROPOSPHERE_LAPSE_RATE,
                            0)
T_STRATOSPHERE_KINK = get_T_ISA_layer(STRATOSPHERE_KINK_ISA, T_TROPOPAUSE,
                            STRATOSPHERE_LAPSE_RATE_1, STRATOSPHERE_START_ISA)

T_STRATOPAUSE = get_T_ISA_layer(STRATOPAUSE_ISA, T_STRATOSPHERE_KINK,
                            STRATOSPHERE_LAPSE_RATE_2, STRATOSPHERE_KINK_ISA)
T_MESOSPHERE_KINK = get_T_ISA_layer(MESOSPHERE_KINK_ISA, T_STRATOPAUSE,
                            MESOSPHERE_LAPSE_RATE_1, MESOSPHERE_ISA)

# Derived pressure values at key altitudes
P_TROPOPAUSE = get_p_ISA_layer(TROPOPAUSE_ISA, P0_ISA, TROPOSPHERE_LAPSE_RATE,
                                0, T0_ISA)
P_STRATOSPHERE = get_p_ISA_layer(STRATOSPHERE_START_ISA, P_TROPOPAUSE, 0,
                                  TROPOPAUSE_ISA, T_TROPOPAUSE)
P_STRATOSPHERE_KINK = get_p_ISA_layer( STRATOSPHERE_KINK_ISA, P_STRATOSPHERE,
            STRATOSPHERE_LAPSE_RATE_1, STRATOSPHERE_START_ISA, T_TROPOPAUSE)

P_STRATOPAUSE = get_p_ISA_layer(STRATOPAUSE_ISA, P_STRATOSPHERE_KINK,
        STRATOSPHERE_LAPSE_RATE_2, STRATOSPHERE_KINK_ISA, T_STRATOSPHERE_KINK)
P_MESOSPHERE = get_p_ISA_layer(MESOSPHERE_ISA, P_STRATOPAUSE, 0,
                            STRATOPAUSE_ISA, T_STRATOPAUSE)
P_MESOSPHERE_KINK = get_p_ISA_layer(MESOSPHERE_KINK_ISA, P_MESOSPHERE,
            MESOSPHERE_LAPSE_RATE_1, MESOSPHERE_ISA, T_STRATOPAUSE)


def _get_altitude_conditions(z):
    conditions = [
        z < TROPOPAUSE_ISA,
        (z >= TROPOPAUSE_ISA) & (z < STRATOSPHERE_START_ISA),
        (z >= STRATOSPHERE_START_ISA) & (z < STRATOSPHERE_KINK_ISA),
        (z >= STRATOSPHERE_KINK_ISA) & (z < STRATOPAUSE_ISA),
        (z >= STRATOPAUSE_ISA) & (z < MESOSPHERE_ISA),
        (z >= MESOSPHERE_ISA) & (z < MESOSPHERE_KINK_ISA),
        (z >= MESOSPHERE_KINK_ISA)
    ]
    return conditions


def get_T_ISA(z):
    """
    Compute the temperature (in Kelvin) at a given altitude or array of
    altitudes based on the ISA model.

    Parameters:
        z (float or np.ndarray): Altitude(s) in meters

    Returns:
        float or np.ndarray: Temperature(s) in Kelvin

    Raises:
        NotImplementedError: If any altitude is above the maximum.
    """
    if np.any(z > MAX_ALTITUDE):
        raise NotImplementedError(f"Altitudes above {int(MAX_ALTITUDE/1000)} km are not implemented")

    conditions = _get_altitude_conditions(z)

    functions = [
        lambda z: get_T_ISA_layer(z, T0_ISA, TROPOSPHERE_LAPSE_RATE, 0),
        lambda z: get_T_ISA_layer(z, T_TROPOPAUSE, 0, TROPOPAUSE_ISA),
        lambda z: get_T_ISA_layer(z, T_TROPOPAUSE,
                            STRATOSPHERE_LAPSE_RATE_1, STRATOSPHERE_START_ISA),
        lambda z: get_T_ISA_layer(z, T_STRATOSPHERE_KINK,
                            STRATOSPHERE_LAPSE_RATE_2, STRATOSPHERE_KINK_ISA),
        lambda z: get_T_ISA_layer(z, T_STRATOPAUSE, 0, STRATOPAUSE_ISA),
        lambda z: get_T_ISA_layer(z, T_STRATOPAUSE,
                                  MESOSPHERE_LAPSE_RATE_1, MESOSPHERE_ISA),
        lambda z: get_T_ISA_layer(z, T_MESOSPHERE_KINK,
                                MESOSPHERE_LAPSE_RATE_2, MESOSPHERE_KINK_ISA)
    ]

    return np.piecewise(z, conditions, functions)

def get_p_ISA(z):
    """
    Compute the pressure (in Pascals) at a given altitude or array of altitudes
    based on the ISA model.

    Parameters:
        z (float or np.ndarray): Altitude(s) in meters

    Returns:
        float or np.ndarray: Pressure(s) in Pascals

    Raises:
        NotImplementedError: If any altitude is above the maximum.
    """
    if np.any(z > MAX_ALTITUDE):
        raise NotImplementedError(f"Altitudes above {int(MAX_ALTITUDE/1000)} km are not implemented")

    conditions = _get_altitude_conditions(z)

    functions = [
        lambda z: get_p_ISA_layer(z, P0_ISA, TROPOSPHERE_LAPSE_RATE, 0, T0_ISA),
        lambda z: get_p_ISA_layer(z, P_TROPOPAUSE, 0,
                                                TROPOPAUSE_ISA, T_TROPOPAUSE),
        lambda z: get_p_ISA_layer(z, P_STRATOSPHERE, STRATOSPHERE_LAPSE_RATE_1,
                                STRATOSPHERE_START_ISA, T_TROPOPAUSE),
        lambda z: get_p_ISA_layer(z, P_STRATOSPHERE_KINK,
        STRATOSPHERE_LAPSE_RATE_2, STRATOSPHERE_KINK_ISA, T_STRATOSPHERE_KINK),
        lambda z: get_p_ISA_layer(z, P_STRATOPAUSE, 0.0, STRATOPAUSE_ISA,
                                 T_STRATOPAUSE),
        lambda z: get_p_ISA_layer(z, P_MESOSPHERE, MESOSPHERE_LAPSE_RATE_1,
                                MESOSPHERE_ISA, T_STRATOPAUSE),
        lambda z: get_p_ISA_layer(z, P_MESOSPHERE_KINK, MESOSPHERE_LAPSE_RATE_2,
                                MESOSPHERE_KINK_ISA, T_MESOSPHERE_KINK)

    ]

    return np.piecewise(z, conditions, functions)



def get_z_isothermal(p, p0, bottom_alt, T):
    return bottom_alt - constants.R_air * T / constants.g0 * np.log(p/p0)

def get_z_with_lapse(p, p0, bottom_alt, T0, lapse_rate):
    return bottom_alt + T0/lapse_rate * (1 - (p/p0)**(constants.R_air * lapse_rate/constants.g0))


def get_z_ISA(p):
    """
    Compute the altitude (in meters) for a given pressure or array of pressures
    based on the ISA model. This is the inverse of get_p_ISA.

    Parameters
    ----------
        p : float or np.ndarray
            Pressure(s) in Pascals

    Returns
    -------
        float or np.ndarray: Altitude(s) in meters

    Raises
    ------
        NotImplementedError: If any pressure is below the minimum pressure at MAX_ALTITUDE
        ValueError: If any pressure is above sea level pressure
    """
    min_pressure = get_p_ISA(MAX_ALTITUDE)
    if np.any(p < min_pressure):
        raise NotImplementedError(f"Pressures below {min_pressure:.2f} Pa are not implemented")
    if np.any(p > P0_ISA):
        raise ValueError(f"Pressures above sea level ({P0_ISA} Pa) are invalid")

    conditions = [
        p > P_TROPOPAUSE,
        (p <= P_TROPOPAUSE) & (p > P_STRATOSPHERE),
        (p <= P_STRATOSPHERE) & (p > P_STRATOSPHERE_KINK),
        (p <= P_STRATOSPHERE_KINK) & (p > P_STRATOPAUSE),
        (p <= P_STRATOPAUSE) & (p > P_MESOSPHERE),
        (p <= P_MESOSPHERE) & (p > P_MESOSPHERE_KINK),
        p <= P_MESOSPHERE_KINK
    ]


    functions = [
        lambda p: get_z_with_lapse(p, P0_ISA, 0, T0_ISA, TROPOSPHERE_LAPSE_RATE),
        lambda p: get_z_isothermal(p, P_TROPOPAUSE, TROPOPAUSE_ISA, T_TROPOPAUSE),
        lambda p: get_z_with_lapse(p, P_STRATOSPHERE, STRATOSPHERE_START_ISA, T_TROPOPAUSE, STRATOSPHERE_LAPSE_RATE_1),
        lambda p: get_z_with_lapse(p, P_STRATOSPHERE_KINK, STRATOSPHERE_KINK_ISA, T_STRATOSPHERE_KINK, STRATOSPHERE_LAPSE_RATE_2),
        lambda p: get_z_isothermal(p, P_STRATOPAUSE, STRATOPAUSE_ISA, T_STRATOPAUSE),
        lambda p: get_z_with_lapse(p, P_MESOSPHERE, MESOSPHERE_ISA, T_STRATOPAUSE, MESOSPHERE_LAPSE_RATE_1),
        lambda p: get_z_with_lapse(p, P_MESOSPHERE_KINK, MESOSPHERE_KINK_ISA, T_MESOSPHERE_KINK, MESOSPHERE_LAPSE_RATE_2)
    ]

    return np.piecewise(p, conditions, functions)