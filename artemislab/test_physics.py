"""Numerical checks for ArtemisLab physical assumptions and geometry."""
import math
import numpy as np
from artemislab.physics import (
  circular_start, passive_orbit, delta_v_for_days, time_to_moon_radius,
  timing_error_km, propellant_from_dv
)

r0, v0 = circular_start(93)
assert abs(r0 - 6799.58) < 2, r0
assert abs(v0 - 7.65646) < 0.01, v0
p = passive_orbit(93., 14., 60., points=450)
assert 0 < p['peak_e'] < 0.0001, p['peak_e']
assert abs(p['r_km'][0] - r0) < 1e-6
assert p['maximum_position_offset_km'] > 0
print(f"14-day passive orbit peak e = {p['peak_e']:.10f}")
short = delta_v_for_days(1.5, 600, 93)
middle = delta_v_for_days(3.0, 600, 93)
long = delta_v_for_days(4.2, 600, 93)
assert 2.8 < long < middle < short < 5.0, (short,middle,long)
for target, dv in [(1.5,short), (3.0,middle), (4.2,long)]:
    tof = time_to_moon_radius(dv,600,93)
    assert abs(tof/86400-target) < .002, (target,tof/86400,dv)
    print(f"{target:.1f} days => {dv:.6f} km/s, flight={tof/86400:.6f} d")
err = timing_error_km(11, 93)
assert 4500 < err < 5000, err
assert abs(timing_error_km(-11,93)+err) < 1e-8
assert propellant_from_dv(short,450,26000) > propellant_from_dv(long,450,26000)
print(f"Extra-credit nominal +11s downrange offset = {err:.2f} km")
