# 🚀 NozzleLab — MEE 453 Problem 1

Interactive Streamlit app for steady isentropic rocket-nozzle calculations and extra-credit altitude-integrated thrust.

## Open alongside Problem 2

Deploy the **OrbitLab** app using this GitHub repository and branch:
- Repository: `wbfluffywolf-eng/blank-app`
- Branch: `mee453-orbitlab`
- Main file: `orbitlab/streamlit_app.py`

The new Problem 1 page then appears in Streamlit's left-hand navigation automatically.

## Or deploy Problem 1 as its own standalone website

- Repository: `wbfluffywolf-eng/blank-app`
- Branch: `mee453-orbitlab`
- Main file: `nozzlelab/streamlit_app.py`

Streamlit Community Cloud: https://share.streamlit.io/

## Features

- Editable throat diameter, exit-to-throat diameter ratio, chamber temperature, chamber pressure, gamma and specific gas constant.
- Choked mass flow and supersonic isentropic area–Mach root.
- Exit conditions and thrust, momentum/pressure contributions.
- Altitude 0–96 km via log-linear interpolation of **absolute pressure in Pa** and density from PDAS U.S. Standard Atmosphere 1976, geometric altitudes.
- Extra-credit four-parameter design comparison, numerical integral of F dh (N·m) and expansion-ratio sensitivity sweep.
- Step-by-step textbook equation references and CSV exports.

Default parameters: throat diameter 61.5 cm, exit diameter ratio 3.14, chamber T0=1624 K, chamber p0=18 MPa, gamma=1.33, and R=287 J/(kg K). **R=287 is dry air only, assumed due to missing exhaust gas composition.**

Sources: Lee, *Aerospace Propulsion* Chs. 1, 8, 10; https://www.pdas.com/bigtables.html

**Limitations:** Ideal isentropic nozzle; flow separation/shocks, combustion chemistry, nozzle heating, and mass or propellant limits not modeled. J=∫F dh is the assignment metric (N·m), *not* impulse.
