import math
import numpy as np
from nozzlelab.physics import compute_nozzle, atmospheric_pressure, atmospheric_density, altitude_curve, thrust_at

n = compute_nozzle()
c = altitude_curve(n,96,601)
assert abs(n["Mach"] - 3.667) < .01, n["Mach"]
assert abs(n["mdot"] - 5268.1) < 5, n["mdot"]
assert abs(n["U_exit"] - 1609.3) < 5, n["U_exit"]
assert abs(n["P_exit"] - 161843) < 1500, n["P_exit"]
assert abs(thrust_at(n,0)/1e6 - 8.655) < .015
assert abs(thrust_at(n,96)/1e6 - 8.952) < .015
assert abs(atmospheric_pressure(96) - 0.0638) < 0.0001
assert abs(atmospheric_density(96) - 1.157e-6) < .02e-6
assert c["thrust"][-1] > c["thrust"][0]
assert 8.4e11 < c["integrated_thrust"] < 8.7e11
n2=compute_nozzle(67.65,3.611,1786.4,19.8,1.33,287)
c2=altitude_curve(n2,96,601)
assert c2["integrated_thrust"] > c["integrated_thrust"]
assert abs(n2["Mach"]-3.934) < .03
assert c2["integrated_thrust"] > 1e12
for ratio in [1.0,1.01,2.0,8.0]:
    z=compute_nozzle(61.5,ratio,1624,18)
    assert math.isfinite(z["Mach"])
print("NozzleLab physics checks passed")
