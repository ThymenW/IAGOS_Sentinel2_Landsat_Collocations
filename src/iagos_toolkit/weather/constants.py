"""
Physical constants and unit conversion factors
"""

T0 = 273.16  # Absolute zero value K

c_p = 1004.6662184201462 # Specific heat in J/kg/K
R_air = 287.04749097718457 # Gas constant of dry air in J/kg/K
P0 = 1000e2 # Reference pressure in Pa
g0 = 9.80665 # m/s^2
N_A = 602.257e21 # Avogadro's number, 1/mol

PLANCK_CONSTANT = 6.62607015e-34 # Planck's constant in J/Hz
BOLTZMANN_CONSTANT = 1.380649e-23 # Boltzmann constant in J/k

# Speed of light in m/s
SPEED_OF_LIGHT = 299792458.

# Conversion factors
NM2METER = 1852 # nautical miles to meters
KNOTS2MS = 0.514444444 # knots to meters per second

# TODO: the precisions used for the molar masses
# are a bit arbitrary and inconsistent
M_C = 12.01 # kg/kmol, molar mass of carbon
M_H = 1.008 # kg/kmol, molar mass of hydrogen
M_N = 14.01 # kg/kmol, molar mass of nitrogen
M_O = 16.00 # kg/kmol, molar mass of oxygen
M_CO2 = M_C + 2 * M_O
M_H2O = 2 * M_H + M_O
X_N2 = 0.79 # volume fraction of nitrogen in air
X_O2 = 0.21 # volume fraction of oxygen in air

EPSILON = 0.622 # Ratio of molecular weight of water to dry air
R_wv = R_air / EPSILON # Gas constant of water vapor in J/kg/K

R_d: float = 287.05 # Gas constant of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
R_v: float = 461.51 # Gas constant of water vapour :math:`[J \ kg^{-1} \ K^{-1}]`

# #: Isobaric heat capacity of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
# c_pd: float = 1004.0  # 1005.7?

# #: Isobaric heat capacity of water vapor :math:`[J \ kg^{-1} \ K^{-1}]`
# c_pv: float = 1870.0

# #: Isobaric heat capacity
# c_pm: float = 1004
