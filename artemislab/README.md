# ArtemisLab — MEE 453 Problem 3

Interactive Streamlit numerical study for the third aerospace-propulsion assignment.

## Easiest deployment: add to the same OrbitLab website

The new Problem 3 page is already linked into the existing Streamlit multipage app:

- Repository: `wbfluffywolf-eng/blank-app`
- Branch: `mee453-orbitlab`
- Main file path: `orbitlab/streamlit_app.py`

After deployment on https://share.streamlit.io/, open the **Problem 3 Artemis** page in the left menu. Problems 1, 2, and 3 are in the same site.

## Standalone Problem 3 deployment

Use main file `artemislab/streamlit_app.py` with the same repository and branch.

## Features

- 93-minute initial circular Earth orbit: radius, altitude, orbital speed.
- 14-day passive LEO simulation with the Moon on a circular orbit, direct lunar gravity and indirect Earth-centered term.
- Time history of osculating eccentricity and offset from a two-body circular reference.
- Simulated finite-duration (default 600 s) tangential burn and Earth-only coast to the Moon's **orbital radius** in 1.5–4.2 days, solved with an event-based ODE/root solver.
- Comparison of Δv, propellant requirements and food/water supplies (editable crew number, Isp and dry mass).
- First-order downrange phase displacement for ±11 s ignition start-time error.
- Interactive Plotly charts, exportable CSV results and relevant equations.

## Important caveat

Reaching the Moon's orbital radius **is not the same as intercepting the moving Moon**. The simple transfer calculations ignore lunar gravity during the transfer and do not include navigation, corrections, realistic Artemis II mission maneuvers, or crew life-support/vehicle-mass constraints. The ignition-error estimate is an angular first-order estimate relative to an assumed nominal alignment, **not** a high-fidelity lunar flyby miss distance.

The 3-body model is used for the passive wobble; the 2-body model is used to compute the transfer trades. This separation is intentional and clearly documented in the app.

## Files

- `physics.py`: numerical models, constants and root solving
- `streamlit_app.py`: interactive dashboard
- `test_physics.py`: numerical validation checks

Physics references: T.-W. Lee, *Aerospace Propulsion*, Ch. 8–9, standard restricted three-body equations, and the Tsiolkovsky rocket equation.
