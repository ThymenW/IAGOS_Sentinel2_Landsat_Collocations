import numpy as np

import iagos_toolkit.weather.constants as constants

# Saturation over liquid water (Buck, 1981)
COEFFS_WATER_BUCK = {
    'a1': 611.21,   # Pa
    'a3': 17.502,
    'a4': 32.19    # K
}

# Saturation over ice (Alduchov & Eskridge, 1996)
COEFFS_ICE_ALDUCHOV = {
    'a1': 611.21,   # Pa
    'a3': 22.587,
    'a4': -0.7     # K
}


def calc_rh_from_specific_humidity(q, T, P):
    """
    Calculate relative humidity from specific humidity.

    Parameters
    ----------
    q : float
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : float
        Temperature, [:math:`K`]
    P : float
        Pressure, [:math:`Pa`]

    Returns
    -------
    float
        Relative humidity, [:math:`%`]

    """
    e_sat = saturation_vapor_pressure_era5(T, COEFFS_WATER_BUCK)
    return (q * P * constants.R_v / constants.R_d) / e_sat


def calc_rhi_from_specific_humidity(q, T, P):
    """
    Calculate relative humidity w.r.t. ice from specific humidity.

    Parameters
    ----------
    q : float
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : float
        Temperature, [:math:`K`]
    P : float
        Pressure, [:math:`Pa`]

    Returns
    -------
    float
        Relative humidity, [:math:`%`]

    """
    e_sat = saturation_vapor_pressure_era5(T, COEFFS_ICE_ALDUCHOV)
    return (q * P * constants.R_v / constants.R_d) / e_sat


def saturation_vapor_pressure(T, coeffs):
    """
    Calculate the saturation vapor pressure using the formula from the problem.
    T : Temperature in Kelvin
    coeffs : Dictionary with 'b' and coefficients a_-1, a_0, a_1, a_2, a_3, a_4
    """
    b = coeffs['b']
    ln_ep = b * np.log(T)
    for j in range(-1, 5):
        aj = coeffs.get(f'a{j}', 0)
        ln_ep += aj * T**j
    return np.exp(ln_ep)  # Output in Pascals


def saturation_vapor_pressure_era5(T, coeffs):
    """
    Calculate the saturation vapor pressure using the Tetens-type formula.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in Kelvin
    coeffs : dict
        Dictionary containing:
        - 'a1' : coefficient a1 (Pa)
        - 'a3' : coefficient a3
        - 'a4' : coefficient a4 (K)

    Returns
    -------
    e_sat : float or np.ndarray
        Saturation vapor pressure in Pascals
    """
    a1 = coeffs['a1']
    a3 = coeffs['a3']
    a4 = coeffs['a4']

    return a1 * np.exp(a3 * (T - constants.T0) / (T - a4))