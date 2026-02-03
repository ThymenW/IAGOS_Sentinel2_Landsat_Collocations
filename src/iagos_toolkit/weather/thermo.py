import numpy as np


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

#: Absolute zero value :math:`[C]`
absolute_zero: float = -273.16
T0 = -1 * absolute_zero  # 273.15 K

#: Gas constant of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
R_d: float = 287.05

#: Gas constant of water vapour :math:`[J \ kg^{-1} \ K^{-1}]`
R_v: float = 461.51

#: Ratio of gas constant for dry air / gas constant for water vapor
epsilon: float = R_d / R_v

#: Isobaric heat capacity of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
c_pd: float = 1004.0  # 1005.7?

#: Isobaric heat capacity of water vapor :math:`[J \ kg^{-1} \ K^{-1}]`
c_pv: float = 1870.0

#: Isobaric heat capacity
c_pm: float = 1004


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
    return (q * P * R_v / R_d) / e_sat


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
    return (q * P * R_v / R_d) / e_sat


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

    return a1 * np.exp(a3 * (T - T0) / (T - a4))

