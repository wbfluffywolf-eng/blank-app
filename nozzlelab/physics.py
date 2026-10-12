"""Ideal rocket nozzle and PDAS U.S. Standard Atmosphere, 1976 calculations.

Assumes steady, choked, calorically perfect, isentropic nozzle flow.
The gas constant defaults to air as an illustrative assignment assumption.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

# PDAS Table 1: geometric altitude Z (km), pressure (Pa), density (kg/m^3).
# https://www.pdas.com/bigtables.html
# Values are printed to limited significant figures in the PDAS public table.
PDAS_KM = np.arange(0.0, 101.0, 5.0)
PDAS_PA = np.array([
    101325.0, 54048.0, 26500.0, 12112.0, 5529.3,
    2549.2, 1197.0, 574.59, 287.14, 149.10,
    79.779, 42.525, 21.959, 10.930, 5.2209,
    2.3881, 1.0525, 0.44568, 0.18435, 0.075775, 0.032012
], dtype=float)
PDAS_RHO = np.array([
    1.2250, 0.73643, 0.41351, 0.19476, 0.088910,
    0.040084, 0.018410, 0.0084634, 0.0039957, 0.0019663,
    0.0010269, 0.00056810, 0.00030968, 0.00016321, 0.000082829,
    0.000039921, 0.000018458, 0.0000082195, 0.0000034400,
    0.0000013873, 0.00000056044
], dtype=float)


def atmospheric_pressure(altitude_km):
    """Log-linear interpolation of PDAS absolute pressure. Altitudes in geometric km."""
    z = np.asarray(altitude_km, dtype=float)
    if np.any(z < 0) or np.any(z > 100):
        raise ValueError("PDAS tabulation covers geometric altitude 0–100 km in this app.")
    result = np.exp(np.interp(z, PDAS_KM, np.log(PDAS_PA)))
    return float(result) if result.ndim == 0 else result


def atmospheric_density(altitude_km):
    z = np.asarray(altitude_km, dtype=float)
    if np.any(z < 0) or np.any(z > 100):
        raise ValueError("Density tabulation covers geometric altitude 0–100 km.")
    result = np.exp(np.interp(z, PDAS_KM, np.log(PDAS_RHO)))
    return float(result) if result.ndim == 0 else result


def nozzle_area_mach(mach, gamma):
    t = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) * mach**2 / 2.0)
    return t**((gamma + 1.0) / (2.0 * (gamma - 1.0))) / mach


def compute_nozzle(throat_cm=61.5, diameter_ratio=3.14, t0=1624.,
                   p0_mpa=18., gamma=1.33, gas_R=287.):
    if not (throat_cm > 0 and diameter_ratio >= 1 and t0 > 0 and p0_mpa > 0
            and gamma > 1 and gas_R > 0):
        raise ValueError("Diameter, pressure, temperature and R must be positive; gamma > 1; diameter ratio >= 1.")
    Dstar = throat_cm / 100.0
    astar = math.pi * Dstar**2 / 4.0
    de = Dstar * diameter_ratio
    ae = math.pi * de**2 / 4.0
    area_ratio = diameter_ratio**2
    if math.isclose(area_ratio, 1.0, rel_tol=0, abs_tol=1e-12):
        mach = 1.0
    else:
        def residual(m):
            return nozzle_area_mach(m, gamma) - area_ratio
        upper = 20.
        while residual(upper) < 0 and upper < 1e6:
            upper *= 2.
        mach = brentq(residual, 1.0 + 1e-9, upper, xtol=1e-12)
    t_exit = t0 / (1.0 + (gamma - 1.0) * mach**2 / 2.0)
    p0 = p0_mpa * 1e6
    p_exit = p0 / (1.0 + (gamma - 1.0) * mach**2 / 2.0)**(gamma / (gamma - 1.0))
    exit_velocity = mach * math.sqrt(gamma * gas_R * t_exit)
    mass_flow = astar * p0 / math.sqrt(gas_R * t0) * math.sqrt(
        gamma * (2.0 / (gamma + 1.0))**((gamma + 1.0) / (gamma - 1.0)))
    return dict(throat_cm=throat_cm, diameter_ratio=diameter_ratio,
                t0=t0, p0_mpa=p0_mpa, gamma=gamma, gas_R=gas_R,
                Dstar=Dstar, Dexit=de, Astar=astar, Aexit=ae,
                area_ratio=area_ratio, Mach=mach, T_exit=t_exit,
                P_exit=p_exit, U_exit=exit_velocity, mdot=mass_flow,
                F_momentum=mass_flow * exit_velocity,
                F_vacuum=mass_flow * exit_velocity + p_exit * ae)


def thrust_at(nozzle, alt_km):
    return (nozzle["F_momentum"] +
            (nozzle["P_exit"] - atmospheric_pressure(alt_km)) * nozzle["Aexit"])


def altitude_curve(nozzle, max_km=96.0, points=600):
    if not (0 < max_km <= 100 and points >= 2):
        raise ValueError("Choose a maximum altitude from 0–100 km and at least two samples.")
    alt_km = np.linspace(0.0, max_km, int(points))
    p = atmospheric_pressure(alt_km)
    pressure_thrust = (nozzle["P_exit"] - p) * nozzle["Aexit"]
    thrust = nozzle["F_momentum"] + pressure_thrust
    integral = float(trapezoid(thrust, alt_km * 1000.0))
    return dict(alt_km=alt_km, p_ambient=p,
                density=atmospheric_density(alt_km), p_thrust=pressure_thrust,
                momentum_thrust=np.full_like(alt_km, nozzle["F_momentum"]),
                thrust=thrust, integrated_thrust=integral)
