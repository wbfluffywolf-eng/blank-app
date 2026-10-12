"""MEE 453 Problem 3 simplified Earth-Moon trajectory models.

Models:
- Passive 14-day satellite: Earth-centered 2D Earth/Moon gravity including the
  indirect lunar acceleration on the Earth-centered origin.
- Translunar flight: finite 600s (editable) tangential engine acceleration in
  an *Earth-only two-body* coast, targeting the radial Moon-orbit distance, NOT
  an exact Moon intercept.
- Extra credit: timing/phase shift relative to a moving circular Moon, holding
  burn magnitude and duration unchanged and assuming the nominal transfer is aligned.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

MU_E = 398600.4418  # km^3/s^2
MU_M = 4902.8001    # km^3/s^2
R_E = 6371.0        # mean Earth radius, km
R_MOON = 384400.0   # ideal mean lunar orbital radius, km
G0 = 9.80665
LUNAR_RATE = math.sqrt((MU_E + MU_M) / R_MOON**3)


def circular_start(period_min: float):
    if period_min < 85 or period_min > 1000:
        raise ValueError("Choose a reasonable orbital period between 85 and 1000 minutes.")
    T = period_min * 60.0
    r = (MU_E * (T / (2 * math.pi))**2)**(1.0 / 3.0)
    if r <= R_E:
        raise ValueError("Orbital radius must be larger than the Earth radius.")
    v = math.sqrt(MU_E / r)
    return r, v


def passive_orbit(period_min=93., duration_days=14., moon_phase_deg=60., points=1800):
    """Integrate rotating-Moon third-body perturbation with an Earth-only comparator."""
    r0, v0 = circular_start(period_min)
    tmax = duration_days * 86400.0
    t_eval = np.linspace(0., tmax, int(points))
    phase = math.radians(moon_phase_deg)
    def rhs(t, y):
        r = y[:2]
        d = np.linalg.norm(r)
        m = R_MOON * np.array([math.cos(LUNAR_RATE*t + phase),
                               math.sin(LUNAR_RATE*t + phase)])
        delta = m-r
        accel = -MU_E * r / d**3 + MU_M * (
            delta / np.linalg.norm(delta)**3 - m / R_MOON**3)
        return np.r_[y[2:], accel]

    sol = solve_ivp(rhs, [0., tmax], [r0,0.,0.,v0], t_eval=t_eval,
                    method="DOP853", rtol=2e-10, atol=1e-11,
                    max_step=400.)
    if not sol.success:
        raise RuntimeError(f"Orbit integration failed: {sol.message}")
    positions = sol.y[:2].T
    velocities = sol.y[2:].T
    norms = np.linalg.norm(positions, axis=1)
    v2 = (velocities**2).sum(axis=1)
    dot = (positions * velocities).sum(axis=1)
    evec = ((v2 - MU_E / norms)[:,None] * positions -
            dot[:,None]*velocities) / MU_E
    eccentricity = np.linalg.norm(evec, axis=1)
    ang = 2*math.pi*t_eval/(period_min*60.0)
    baseline = np.stack([r0*np.cos(ang), r0*np.sin(ang)], axis=1)
    positional_difference = np.linalg.norm(positions-baseline,axis=1)
    return dict(
        time_days=t_eval/86400., positions=positions,
        r_km=norms, eccentricity=eccentricity,
        baseline=baseline,
        position_offset_km=positional_difference,
        initial_radius_km=r0,
        circular_speed_km_s=v0,
        peak_e=float(np.max(eccentricity)),
        end_e=float(eccentricity[-1]),
        rms_e=float(np.sqrt(np.mean(eccentricity**2))),
        maximum_position_offset_km=float(np.max(positional_difference)))


def launch_burn(delta_v_km_s, burn_seconds=600., r0_km=None, v0_km_s=None):
    """Constant tangential acceleration (magnitude deltaV/burn_seconds)."""
    if r0_km is None or v0_km_s is None:
        r0_km,v0_km_s=circular_start(93.)
    if delta_v_km_s < 0 or burn_seconds <= 0:
        raise ValueError("Delta-v must be nonnegative and burn duration positive.")
    acceleration=delta_v_km_s/burn_seconds

    def rhs(_t,y):
        r=y[:2]; vel=y[2:]; rn=np.linalg.norm(r); vn=np.linalg.norm(vel)
        return np.r_[vel, -MU_E*r/rn**3 + acceleration*vel/max(vn,1e-12)]

    sol=solve_ivp(rhs,[0.,burn_seconds],[r0_km,0.,0.,v0_km_s],
                  rtol=1e-10,atol=1e-11,method="DOP853",
                  max_step=max(5.,burn_seconds/25.))
    if not sol.success:
        raise RuntimeError(f"Burn integration failed: {sol.message}")
    return sol.y[:,-1]


def time_to_moon_radius(delta_v_km_s, burn_seconds=600., period_min=93.,
                        include_path=False):
    """First outward crossing of radial Moon distance after finite tangential burn."""
    r0,v0=circular_start(period_min)
    state=launch_burn(delta_v_km_s,burn_seconds,r0,v0)
    r=state[:2]
    vel=state[2:]
    rn=np.linalg.norm(r)
    energy=.5*np.dot(vel,vel)-MU_E/rn
    evec=((np.dot(vel,vel)-MU_E/rn)*r-np.dot(r,vel)*vel)/MU_E
    ecc=np.linalg.norm(evec)
    if energy < 0:
        semimajor=-MU_E/(2*energy)
        apo=semimajor*(1+ecc)
        if apo < R_MOON:
            return (math.inf,None) if include_path else math.inf

    def rhs(_t,y):
        rr=y[:2]; nr=np.linalg.norm(rr)
        return np.r_[y[2:],-MU_E*rr/nr**3]

    def reached_moon(_t,y):
        return np.linalg.norm(y[:2])-R_MOON
    reached_moon.terminal=True
    reached_moon.direction=1
    # Dense output supports plotting full transfer only when specifically requested
    sol=solve_ivp(rhs,[burn_seconds, 8.0*86400.], state,
                  events=reached_moon, rtol=2e-9, atol=1e-9,
                  method="DOP853", max_step=20000.,
                  dense_output=include_path)
    if not sol.success:
        raise RuntimeError(f"Transfer coast integration failed: {sol.message}")
    if len(sol.t_events[0])==0:
        return (math.inf,None) if include_path else math.inf
    t_event=float(sol.t_events[0][0])
    if include_path:
        ts=np.linspace(burn_seconds,t_event,360)
        states=sol.sol(ts)
        return t_event,dict(t=ts,states=states,burn_end=state)
    return t_event


def delta_v_for_days(days=3.0,burn_seconds=600.,period_min=93.):
    if not 1.0<=days<=6.0:
        raise ValueError("Flight-time target must be 1–6 days.")
    target=days*86400.0
    def difference(dv):
        tof=time_to_moon_radius(float(dv),burn_seconds,period_min)
        return (tof-target) if math.isfinite(tof) else 8*86400-target
    low,high=2.0,6.5
    flo,fhi=difference(low),difference(high)
    if flo*fhi>0:
        raise ValueError("Target transfer duration cannot be bracketed with given inputs.")
    dv=brentq(difference,low,high,xtol=1e-6,rtol=1e-8,maxiter=50)
    return dv


def propellant_from_dv(dv_km_s,isp_s,delivered_kg):
    if isp_s<=0 or delivered_kg<=0:
        raise ValueError("Isp and final mass must be positive.")
    return delivered_kg*math.expm1(dv_km_s*1000./(isp_s*G0))


def timing_error_km(offset_s=11.,period_min=93.,moon_radius_km=R_MOON):
    """Signed circumferential displacement assuming nominal lunar-phase alignment.

    First-order estimate for a shift in ignition time at unchanged burn duration:
    R_Moon * (orbital angular rate - lunar orbital rate) * delta time.
    Not a full 3-body flyby targeting calculation.
    """
    n_leo=2*math.pi/(period_min*60.)
    return moon_radius_km*(n_leo-LUNAR_RATE)*offset_s
