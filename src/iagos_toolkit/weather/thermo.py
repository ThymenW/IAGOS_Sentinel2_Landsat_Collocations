from typing import Union

import numpy as np
import xarray as xr

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


def goff_gratch_water(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates saturation vapor pressure over water.
    Source: Goff and Gratch (1946) adapted to ITS-90
    in Sonntag (1994).

    Used in MOZAIC.
    
    Parameters
    ----------
    T : float or numpy array
        Temperature in Kelvin
    
    Outputs
    -------
    ew : float or numpy array
        Saturation vapor pressure over water in Pa
    """
    exponent = (-6096.9385/T 
                + 16.635794 
                - (2.711193e-2)*T
                + (1.673952e-5)*T**2
                + 2.433502 * np.log(T))
                
    return 100*np.exp(exponent)


def goff_gratch_ice(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates saturation vapor pressure overice.
    Source: https://www.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    
    Used in MOZAIC. 
    
    Parameters
    ----------
    T : float or numpy array
        Temperature in Kelvin
    
    Outputs
    -------
    ei : float or numpy array
        Saturation vapor pressure over ice in Pa
    """
    exponent = np.log(10)*(-9.09718*(273.16/T -1)
                - 3.56664*np.log10(273.16/T)
                + 0.876739*(1-T/273.16)
                + np.log10(6.1071)
                  )

    return 100 * np.exp(exponent)


def get_T_LM(G : Union[np.ndarray, xr.DataArray]) -> np.ndarray:
    """ 
    Computes the T_LM temperature from Schumann (1996)
    which is used in the computation of the Schmidt-
    Appleman criterion.
    
    Parameters
    ----------
    G : numpy array
        Mixing line gradient in Pa/K
    
    Outputs 
    -------
    T_LM : numpy array
        Temperature in Kelvin 
    """ 
    # If G <= 0.053, the approximation does not work. 

    T_LM = 273.15-46.46 + 9.43*np.log(G-0.053) + 0.720*(np.log(G-0.053))**2

    return T_LM


def get_mixing_line_gradient_from_LHV(pressure: Union[float, np.ndarray],
    epsilon: float=0.622, EI_H2O: float=1.23, LHV: float=42e6, cp: float=constants.c_p,
             eta: float=0.4) -> Union[float, np.ndarray]:
    """
    Computes the mixing line gradient based on lower heating value (LHV) of the used fuel. See Schumann (1996) for a 
    derivation. 

    Parameters
    ----------
    pressure: float or numpy array
        Pressure in Pa
    epsilon: float 
        Ratio of molar mass of water to air 
    EI_H2O: float
        Emissions index of water for the fuel used 
    LHV: float
        Lower heating value of fuel in J/kg
    cp: float
        Specific heat of air in J/(Kg*K)
    eta: float
        Efficiency of aircraft 

    Outputs
    -------
    G: numpy array or float
        Mixing line gradient in Pa/K
    """

    return (cp*pressure/epsilon)*(EI_H2O/(LHV*(1-eta)))
