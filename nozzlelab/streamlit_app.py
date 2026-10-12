"""NozzleLab: MEE 453 Problem 1 interactive nozzle and extra-credit design study."""
import math
from pathlib import Path
import sys

# Supports direct Streamlit deployment and use as an OrbitLab multipage page.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nozzlelab.physics import (
    PDAS_KM, PDAS_PA, atmospheric_pressure, altitude_curve,
    compute_nozzle, nozzle_area_mach, thrust_at,
)


def readable(value, precision=3):
    return f"{value:,.{precision}f}"


def metric_row(model, curve):
    cols = st.columns(4)
    cols[0].metric("Exit Mach number", f"{model['Mach']:.4f}")
    cols[1].metric("Exit velocity", f"{model['U_exit']:,.1f} m/s")
    cols[2].metric("Exit pressure", f"{model['P_exit']/1000:,.3f} kPa")
    cols[3].metric("Propellant mass flow", f"{model['mdot']:,.1f} kg/s")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sea-level thrust", f"{curve['thrust'][0]/1e6:.4f} MN")
    c2.metric("96 km thrust", f"{thrust_at(model,96)/1e6:.4f} MN")
    c3.metric("Area expansion ratio", f"{model['area_ratio']:.4f}")
    c4.metric("Integrated thrust", f"{curve['integrated_thrust']/1e11:.4f} × 10¹¹ N·m")


def render():
    st.markdown("""
    <style>
    .block-container {max-width: 1480px; padding-top: 1.2rem;}
    [data-testid="stMetric"] {background:rgba(68,150,191,.08);
      padding:14px; border-radius:12px; border:1px solid rgba(120,157,186,.25);}
    </style>""", unsafe_allow_html=True)
    st.title("🚀 NozzleLab — MEE 453 Problem 1")
    st.caption("Ideal choked rocket nozzle · Mach and exit conditions · altitude–thrust curves · extra-credit design comparison")
    st.sidebar.header("Rocket inputs")
    throat_cm = st.sidebar.number_input("Throat diameter D* (cm)", min_value=0.1, max_value=500.0,
                                        value=61.5, step=0.5, format="%.2f")
    diam_ratio = st.sidebar.number_input("Exit/throat diameter ratio De/D*", min_value=1.0,
                                         max_value=15.0, value=3.14, step=0.05, format="%.3f")
    t0 = st.sidebar.number_input("Chamber total temperature T⁰ (K)", min_value=200.0,
                                 max_value=6000.0, value=1624.0, step=25.0)
    p0 = st.sidebar.number_input("Chamber total pressure p⁰ (MPa)", min_value=0.01,
                                 max_value=100.0, value=18.0, step=0.5, format="%.3f")
    gamma = st.sidebar.number_input("Specific heat ratio γ", min_value=1.05, max_value=1.67,
                                    value=1.33, step=0.01, format="%.3f")
    gas_R = st.sidebar.number_input("Exhaust gas constant R (J/kg·K)", min_value=50.0,
                                    max_value=2000.0, value=287.0, step=5.0)
    resolution = st.sidebar.slider("Altitude plot samples", min_value=100, max_value=3000,
                                   value=600, step=100)
    st.sidebar.caption("R = 287 is the textbook value for *air*, used here as an assumption for exhaust gas. "
                       "Change it if your instructor supplies rocket exhaust gas properties.")
    try:
        model = compute_nozzle(throat_cm, diam_ratio, t0, p0, gamma, gas_R)
        curve = altitude_curve(model, max_km=96.0, points=resolution)
    except ValueError as ex:
        st.error(str(ex))
        st.stop()
    metric_row(model, curve)
    st.info("Model assumptions: steady ideal-gas flow, constant γ and R, choked throat, "
            "isentropic supersonic nozzle expansion, fixed chamber conditions, "
            "and no losses or internal shock. Atmospheric pressure is the ABSOLUTE pressure "
            "in Pa from the PDAS U.S. Standard Atmosphere 1976, not the pressure ratio.")

    a, b, c, d = st.tabs([
        "📊 Thrust vs altitude", "⚙️ Nozzle conditions",
        "🏁 Extra credit", "📚 Equations & data"
    ])
    with a:
        st.subheader("Thrust from sea level to 96 km")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["alt_km"], y=curve["thrust"]/1e6, name="Total thrust",
            line=dict(color="#37b7af", width=4)))
        fig.add_trace(go.Scatter(
            x=curve["alt_km"], y=curve["momentum_thrust"]/1e6,
            name="Momentum contribution", line=dict(color="#5089dd",dash="dash",width=2)))
        fig.add_trace(go.Scatter(
            x=curve["alt_km"], y=curve["p_thrust"]/1e6,
            name="Pressure contribution", line=dict(color="#e7a448",width=2)))
        fig.update_layout(height=460, hovermode="x unified",
                          xaxis_title="Geometric altitude (km)",
                          yaxis_title="Thrust (MN)", legend=dict(orientation="h",y=-0.22),
                          margin=dict(t=22))
        st.plotly_chart(fig, use_container_width=True)
        probe = st.slider("Explore thrust at altitude (km)", 0.0, 96.0, 96.0, 0.5)
        pressure = atmospheric_pressure(probe)
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Atmospheric pressure", f"{pressure:.6g} Pa")
        cp2.metric("Thrust at this altitude", f"{thrust_at(model,probe)/1e6:.5f} MN")
        cp3.metric("Pressure-thrust component",
                   f"{((model['P_exit']-pressure)*model['Aexit'])/1000:,.2f} kN")
        st.latex(r"F(h)=\dot{m}_p U_e + [p_e-p_a(h)]A_e")
        st.caption("At 96 km, the pressure is interpolated from the 95 km and 100 km PDAS values. "
                   "PDAS lists geometric altitude; interpolation uses the logarithm of pressure.")
        bottom1,bottom2=st.columns(2)
        with bottom1:
            figp=go.Figure(go.Scatter(x=curve["alt_km"],
                         y=curve["p_ambient"],line=dict(color="#9b80d7",width=3)))
            figp.update_layout(height=320,title="Atmospheric pressure — PDAS",
                               yaxis_type="log",xaxis_title="Altitude (km)",yaxis_title="Pressure (Pa)")
            st.plotly_chart(figp,use_container_width=True)
        with bottom2:
            figq=go.Figure(go.Scatter(x=curve["alt_km"],
                         y=curve["density"],line=dict(color="#c5903e",width=3)))
            figq.update_layout(height=320,title="Atmospheric density — PDAS",
                               yaxis_type="log",xaxis_title="Altitude (km)",
                               yaxis_title="Density (kg/m³)")
            st.plotly_chart(figq,use_container_width=True)
        result_data=pd.DataFrame({
            "Altitude_km":curve["alt_km"],
            "Ambient_pressure_Pa":curve["p_ambient"],
            "Density_kg_m3":curve["density"],
            "Momentum_thrust_N":curve["momentum_thrust"],
            "Pressure_thrust_N":curve["p_thrust"],
            "Total_thrust_N":curve["thrust"],
        })
        st.download_button("Download altitude/thrust data (CSV)",
                           result_data.to_csv(index=False).encode("utf-8"),
                           "nozzlelab_thrust_altitude.csv","text/csv")
        with st.expander("Show numerical results table"):
            st.dataframe(result_data.iloc[::max(1,int(resolution/24))].round(5),
                         hide_index=True,use_container_width=True)

    with b:
        st.subheader("Exit conditions and geometry")
        specs=pd.DataFrame([
            ("Throat diameter D*",model["Dstar"],"m"),
            ("Throat area A*",model["Astar"],"m²"),
            ("Exit diameter De",model["Dexit"],"m"),
            ("Exit area Ae",model["Aexit"],"m²"),
            ("Area ratio Ae/A*",model["area_ratio"],"—"),
            ("Exit Mach Me",model["Mach"],"—"),
            ("Exit temperature Te",model["T_exit"],"K"),
            ("Exit pressure pe",model["P_exit"],"Pa"),
            ("Exit speed Ue",model["U_exit"],"m/s"),
            ("Mass flow ṁp",model["mdot"],"kg/s"),
            ("Vacuum thrust",model["F_vacuum"],"N")
        ],columns=["Quantity","Value","Units"])
        st.dataframe(specs.style.format({"Value":"{:,.6f}"}),hide_index=True,use_container_width=True)
        st.download_button("Download nozzle conditions CSV",specs.to_csv(index=False).encode(),
                           "nozzlelab_exit_conditions.csv","text/csv")
        ma=np.linspace(1.001,max(5.0,model["Mach"]*1.35),450)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=ma,y=[nozzle_area_mach(float(x),gamma) for x in ma],
                        name="Supersonic solution branch",line=dict(color="#308cc8",width=3)))
        fig.add_trace(go.Scatter(x=[model["Mach"]],y=[model["area_ratio"]],
                        name="Selected nozzle",mode="markers",marker=dict(size=13,color="#f0ad48")))
        fig.update_layout(height=390,xaxis_title="Exit Mach number",yaxis_title="Area ratio Ae/A*",
                          title="Area–Mach relationship: supersonic root")
        st.plotly_chart(fig,use_container_width=True)
        if model["P_exit"] < 101325:
            st.warning("The exit pressure is below sea-level atmospheric pressure. "
                       "This overexpanded condition can cause internal shocks or flow separation; "
                       "the ideal isentropic model may be unrealistic at low altitude.")

    with c:
        st.subheader("Extra credit — explore several design changes")
        st.write("Compare your original engine with a candidate using throat diameter, "
                 "expansion ratio, chamber temperature, and chamber pressure. "
                 "The goal is the assignment's altitude-integrated thrust over 0–96 km.")
        col1,col2,col3,col4=st.columns(4)
        with col1:
            cd=st.number_input("Candidate throat D* (cm)",1.0,600.0,
                               min(600.0,round(throat_cm*1.10,3)),0.5,key="cd")
        with col2:
            cr=st.number_input("Candidate De/D*",1.0,15.0,
                               min(15.0,round(diam_ratio*1.15,4)),0.05,
                               format="%.4f",key="cr")
        with col3:
            ct=st.number_input("Candidate chamber T⁰ (K)",200.0,7000.0,
                               min(7000.0,round(t0*1.10,3)),20.0,key="ct")
        with col4:
            cp=st.number_input("Candidate pressure p⁰ (MPa)",0.01,150.0,
                               min(150.0,round(p0*1.10,3)),0.5,key="cp")
        try:
            candidate=compute_nozzle(cd,cr,ct,cp,gamma,gas_R)
            alt_candidate=altitude_curve(candidate,96.0,resolution)
        except ValueError as ex:
            st.error(str(ex)); st.stop()
        j0=curve["integrated_thrust"]
        j1=alt_candidate["integrated_thrust"]
        pc=100*(j1/j0-1)
        m1,m2,m3=st.columns(3)
        m1.metric("Original integrated thrust",f"{j0:.5e} N·m")
        m2.metric("Candidate integrated thrust",f"{j1:.5e} N·m")
        m3.metric("Relative change",f"{pc:+.2f}%")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=curve["alt_km"],y=curve["thrust"]/1e6,
                      name="Original",line=dict(width=3,color="#4d91d8")))
        fig.add_trace(go.Scatter(x=alt_candidate["alt_km"],y=alt_candidate["thrust"]/1e6,
                      name="Candidate",line=dict(width=3,color="#2abb99")))
        fig.update_layout(height=420,xaxis_title="Altitude (km)",yaxis_title="Thrust (MN)",
                          title="Original vs candidate, 0–96 km")
        st.plotly_chart(fig,use_container_width=True)
        summary=pd.DataFrame({
            "Design":["Original","Candidate"],
            "Throat_cm":[throat_cm,cd],
            "Diameter_ratio":[diam_ratio,cr],
            "Chamber_T_K":[t0,ct],
            "Chamber_p_MPa":[p0,cp],
            "Exit_velocity_m_s":[model["U_exit"],candidate["U_exit"]],
            "Propellant_flow_kg_s":[model["mdot"],candidate["mdot"]],
            "Sea_level_thrust_N":[curve["thrust"][0],alt_candidate["thrust"][0]],
            "Thrust_96km_N":[curve["thrust"][-1],alt_candidate["thrust"][-1]],
            "Altitude_integrated_thrust_N_m":[j0,j1]
        })
        st.dataframe(summary.T,use_container_width=True)
        st.download_button("Download design comparison CSV",
                           summary.to_csv(index=False).encode(),"nozzlelab_extra_credit.csv","text/csv")
        st.subheader("Expansion-ratio sensitivity")
        ratios=np.linspace(1.15, min(12.0,max(5.0,diam_ratio*2)),48)
        vals=[]
        for rr in ratios:
            m=compute_nozzle(throat_cm,float(rr),t0,p0,gamma,gas_R)
            vals.append(altitude_curve(m,96.0,max(100,int(resolution/3)))["integrated_thrust"])
        best=int(np.argmax(vals))
        fig=go.Figure(go.Scatter(x=ratios,y=np.array(vals)/1e11,
                        line=dict(color="#d99d47",width=3),name="Sensitivity"))
        fig.add_trace(go.Scatter(x=[ratios[best]],y=[vals[best]/1e11],
                        mode="markers",marker=dict(color="#26b4a5",size=12),
                        name="Best sampled ratio"))
        fig.update_layout(height=320,xaxis_title="Exit/throat DIAMETER ratio De/D*",
                          yaxis_title="Integrated thrust (10¹¹ N·m)")
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"The best sampled diameter ratio in the chosen sweep is {ratios[best]:.3f}. "
                   "This is a bounded parameter sweep, not a global optimization.")
        st.warning("Changing throat diameter or chamber pressure can raise thrust simply by burning "
                   "more propellant. The assignment's objective is measured in N·m (not N·s), "
                   "and has no fuel-budget, mass, temperature-limit, or chamber-strength penalty. "
                   "A larger integral therefore does NOT prove better rocket efficiency. "
                   "Increasing chamber temperature alone does not raise ideal thrust when geometry, "
                   "chamber pressure, gamma, and R are fixed.")

    with d:
        st.subheader("Step-by-step textbook equations")
        st.markdown("**1. Geometry** — Circular throat and exit areas.")
        st.latex(r"A^*=\frac{\pi(D^*)^2}{4},\qquad A_e=\frac{\pi D_e^2}{4},\qquad \frac{A_e}{A^*}=\left(\frac{D_e}{D^*}\right)^2")
        st.write(f"Current values: A* = {model['Astar']:.6f} m², Ae = {model['Aexit']:.6f} m².")
        st.markdown("**2. Area–Mach relation (Lee Eq. 1.26)** — Select the supersonic root.")
        st.latex(r"\frac{A_e}{A^*}=\frac{1}{M_e}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M_e^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}")
        st.write(f"Mach = {model['Mach']:.6f} (supersonic branch).")
        st.markdown("**3. Exit temperature and pressure (Lee Eqs. 1.16–1.17)**")
        st.latex(r"T_e=\frac{T^0}{1+(\gamma-1)M_e^2/2},\quad p_e=\frac{p^0}{[1+(\gamma-1)M_e^2/2]^{\gamma/(\gamma-1)}}")
        st.write(f"Te = {model['T_exit']:.4f} K, pe = {model['P_exit']:.4f} Pa.")
        st.markdown("**4. Exit velocity (Lee Eq. 10.3, via local speed of sound)**")
        st.latex(r"U_e=M_e\sqrt{\gamma R T_e}")
        st.write(f"Ue = {model['U_exit']:.4f} m/s.")
        st.markdown("**5. Choked mass flow (Lee Eq. 10.6)**")
        st.latex(r"\dot{m}_p=\frac{A^*p^0}{\sqrt{RT^0}}\sqrt{\gamma\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}}")
        st.write(f"Mass flow = {model['mdot']:.4f} kg/s.")
        st.markdown("**6. Rocket thrust (Lee Eq. 8.7)**")
        st.latex(r"F(h)=\dot{m}_p U_e+[p_e-p_a(h)]A_e")
        st.markdown("**7. Extra-credit objective**")
        st.latex(r"J=\int_{0}^{96000}F(h)\,dh")
        st.write(f"Baseline integrated thrust = {j0:.5e} N·m.")
        st.divider()
        st.markdown("**PDAS Standard Atmosphere source**")
        st.markdown("[PDAS Table 1 — U.S. Standard Atmosphere 1976](https://www.pdas.com/bigtables.html)")
        st.latex(r"p(h)=p_1\left(\frac{p_2}{p_1}\right)^{(h-h_1)/(h_2-h_1)}")
        st.markdown("This is log-linear interpolation between two tabulated geometric altitudes. "
                    "**PDAS already reports absolute pressure in Pa. Do not multiply by 101,325 again.**")
        st.write(f"Using PDAS at 95 and 100 km, p(96 km) = {atmospheric_pressure(96):.7f} Pa.")
        st.dataframe(pd.DataFrame({"Geometric altitude (km)":PDAS_KM,
                                   "Absolute pressure (Pa)":PDAS_PA}),
                     hide_index=True,use_container_width=True)
        st.caption("Reference: T.-W. Lee, Aerospace Propulsion, Chs. 1, 8 and 10; PDAS "
                   "Table of the U.S. Standard Atmosphere, 1976. Pressure/density interpolated "
                   "between 5 km data rows. Nozzle behavior at flow separation is not modeled.")


if __name__ == "__main__":
    st.set_page_config(page_title="NozzleLab — MEE 453 Problem 1",
                       page_icon="🚀", layout="wide")
    render()
