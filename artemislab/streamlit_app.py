"""Streamlit dashboard for MEE 453 Problem 3 orbital perturbations and translunar tradeoffs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from artemislab.physics import (
    MU_E, MU_M, R_E, R_MOON, G0, LUNAR_RATE, circular_start,
    passive_orbit, delta_v_for_days, time_to_moon_radius,
    propellant_from_dv, timing_error_km
)

COLORS = {"orbit":"#33b8c0","circular":"#8e87e7","moon":"#efb456",
          "fuel":"#48a1da","crew":"#e9ac4d","ecc":"#3cbd8a"}

@st.cache_data(show_spinner=False,ttl=3600)
def cached_passive(period,days,phase):
    return passive_orbit(period,days,phase,points=1800)

@st.cache_data(show_spinner=False,ttl=3600)
def cached_transfer_days(day,burn,period):
    return delta_v_for_days(day,burn,period)

@st.cache_data(show_spinner=False,ttl=3600)
def cached_transfer_path(dv,burn,period):
    return time_to_moon_radius(dv,burn,period,include_path=True)

def chart_layout(title,xlabel,ylabel,height=360):
    return dict(title=title,height=height,
                xaxis_title=xlabel,yaxis_title=ylabel,
                margin=dict(t=48,b=36,l=45,r=25),
                legend=dict(orientation="h",y=-0.2),
                hovermode="x unified")

def render():
    st.markdown("""<style>
      .block-container{max-width:1480px;padding-top:1.3rem;}
      [data-testid="stMetric"]{padding:14px;border-radius:12px;
        border:1px solid rgba(126,155,190,.25);
        background:rgba(69,144,180,.075);}
    </style>""",unsafe_allow_html=True)
    st.title("🌙 ArtemisLab — MEE 453 Problem 3")
    st.caption("14-day Earth–Moon orbital wobble · 1.5–4.2 day lunar transfers · consumables · ±11 s timing error")

    st.sidebar.header("Satellite and Moon")
    period=st.sidebar.number_input("Initial LEO period (min)",85.,180.,93.,1.)
    duration=st.sidebar.number_input("Passive orbit study (days)",1.,30.,14.,1.)
    moon_phase=st.sidebar.slider("Initial Moon angle (degrees)",0,360,60,5)
    st.sidebar.subheader("Transfer sizing")
    burn=st.sidebar.number_input("Tangential engine burn duration (s)",60.,3600.,600.,60.)
    isp=st.sidebar.number_input("Engine specific impulse Isp (s)",100.,600.,450.,10.)
    final_mass=st.sidebar.number_input("Dry spacecraft / delivered mass (kg)",1000.,150000.,26000.,1000.)
    crew=st.sidebar.number_input("Crew members",1,12,4,1)
    daily=st.sidebar.number_input("Food and water per crew member (kg/day)",0.5,10.,3.,0.5)
    selected=st.sidebar.slider("Selected transfer time (days)",1.5,4.2,3.0,0.1)
    ignition_error=st.sidebar.number_input("Timing error for extra credit (s)",-60.,60.,11.,1.)
    st.sidebar.caption("Initial Moon phase is a modeling choice, not a supplied assignment value. "
        "Earth radius, mean lunar orbit distance, burn duration, Isp, crew mass, "
        "and consumables are editable or explicitly documented assumptions.")

    try:
        r0, v0=circular_start(period)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    k1,k2,k3,k4=st.columns(4)
    k1.metric("LEO altitude",f"{r0-R_E:,.2f} km")
    k2.metric("Circular orbit speed",f"{v0:.4f} km/s")
    k3.metric("LEO period",f"{period:.1f} min")
    k4.metric("Moon orbital distance",f"{R_MOON:,.0f} km")

    st.info("This is a **teaching model**, not the actual Artemis II flight plan. "
        "The 14-day passive orbit uses Earth–Moon gravitational perturbations. "
        "The translunar design uses an Earth-centered, finite-duration tangential burn "
        "followed by a two-body coast until crossing the Moon's orbital radius: "
        "it does NOT solve for a rendezvous with the moving Moon, lunar gravity on the transfer, "
        "or real-world guidance and correction maneuvers.")

    tabs=st.tabs([
        "🌍 14-day wobble","🚀 Transfer Δv","⚖️ Fuel vs consumables",
        "⏱️ Extra credit","📚 Equations and assumptions"
    ])

    with st.spinner("Simulating 14-day gravitational perturbations and translunar transfers..."):
        try:
            wobble=cached_passive(period,duration,moon_phase)
            target_days=sorted(set([1.5,2.,2.5,3.,3.5,4.,4.2,round(selected,2)]))
            dv_list=[cached_transfer_days(float(d),burn,period) for d in target_days]
        except (ValueError, RuntimeError) as ex:
            st.error(f"Simulation failed with these inputs: {ex}")
            st.stop()

    table=pd.DataFrame({"Flight_time_days":target_days,"Delta_v_km_s":dv_list})
    table["Food_water_kg"]=table.Flight_time_days*crew*daily
    table["Delivered_mass_kg"]=final_mass+table["Food_water_kg"]
    table["Propellant_kg"]=[
        propellant_from_dv(float(dv),isp,float(m))
        for dv,m in zip(table.Delta_v_km_s,table.Delivered_mass_kg)]
    table["Total_departure_mass_kg"]=table.Delivered_mass_kg+table.Propellant_kg
    selected_idx=int(np.argmin(np.abs(table.Flight_time_days-selected)))
    chosen=table.iloc[selected_idx]

    with tabs[0]:
        st.subheader("Passive orbit — Earth–Moon gravitational perturbation")
        c1,c2,c3=st.columns(3)
        c1.metric("Peak osculating eccentricity",f"{wobble['peak_e']:.7f}")
        c2.metric("Final osculating eccentricity",f"{wobble['end_e']:.7f}")
        c3.metric("Max. position offset vs Earth-only",f"{wobble['maximum_position_offset_km']:.3f} km")
        st.caption("Eccentricity is calculated from instantaneous Earth-centered state vectors. "
                   "It changes through the 14 days; the final orbit is not a permanent fixed ellipse.")
        L,R=st.columns(2)
        with L:
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=wobble["time_days"], y=wobble["eccentricity"],
                line=dict(color=COLORS["ecc"],width=2),name="Instantaneous e"))
            fig.update_layout(**chart_layout("Osculating eccentricity over time",
                "Elapsed time (days)","Eccentricity"))
            st.plotly_chart(fig,use_container_width=True)
        with R:
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=wobble["time_days"], y=wobble["position_offset_km"],
                line=dict(color=COLORS["moon"],width=2),name="Offset"))
            fig.update_layout(**chart_layout("Position difference from Earth-only orbit",
                "Elapsed time (days)","Position difference (km)"))
            st.plotly_chart(fig,use_container_width=True)
        pos=wobble["positions"]
        baseline=wobble["baseline"]
        show=max(100,int(len(pos)*2*period/(duration*24*60)))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=baseline[-show:,0],y=baseline[-show:,1],
                name="Unperturbed circular orbit",line=dict(dash="dash",color=COLORS["circular"],width=2)))
        fig.add_trace(go.Scatter(x=pos[-show:,0],y=pos[-show:,1],
                name="Earth–Moon perturbed orbit",line=dict(color=COLORS["orbit"],width=3)))
        fig.update_layout(**chart_layout("Last ~two orbital revolutions",
                    "X position (km)","Y position (km)",440))
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("For a near-circular LEO, lunar gravity is a weak perturbation. "
                   "The model includes the indirect acceleration of the Earth-centered coordinate origin, "
                   "but excludes solar tides, Earth's J₂ oblateness, atmospheric drag, and maneuvers.")
        wobble_csv=pd.DataFrame({
            "Elapsed_days":wobble["time_days"],
            "Eccentricity":wobble["eccentricity"],
            "Radius_km":wobble["r_km"],
            "Earth_only_offset_km":wobble["position_offset_km"],
        })
        st.download_button("Download wobble results (CSV)",
            wobble_csv.to_csv(index=False).encode(),"mee453_p3_wobble.csv","text/csv")

    with tabs[1]:
        st.subheader("Finite-duration tangential burn to lunar-orbit distance")
        a,b,c=st.columns(3)
        a.metric("Selected flight time",f"{chosen.Flight_time_days:.1f} days")
        b.metric("Required ideal Δv",f"{chosen.Delta_v_km_s:.4f} km/s")
        c.metric("Assumed tangential burn",f"{burn:,.0f} seconds")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=table.Flight_time_days,y=table.Delta_v_km_s,
              mode="lines+markers",name="Required Δv",
              line=dict(color=COLORS["orbit"],width=3)))
        fig.update_layout(**chart_layout("Travel-time vs velocity-change tradeoff",
                           "Time to lunar orbital radius (days)","Ideal tangential Δv (km/s)",430))
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("Explore the selected transfer trajectory",expanded=True):
            try:
                _time,trajectory=cached_transfer_path(float(chosen.Delta_v_km_s),burn,period)
                states=trajectory["states"]
                positions=states[:2].T
                theta=np.linspace(0,2*math.pi,180)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=R_MOON*np.cos(theta),y=R_MOON*np.sin(theta),
                     name="Moon's circular orbital radius",line=dict(color=COLORS["moon"],dash="dot")))
                fig.add_trace(go.Scatter(x=positions[:,0],y=positions[:,1],
                     name="Transfer coast",line=dict(color=COLORS["orbit"],width=3)))
                fig.add_trace(go.Scatter(x=[0],y=[0],mode="markers",name="Earth",
                     marker=dict(size=12,color=COLORS["circular"])))
                fig.update_layout(**chart_layout("Earth-centered transfer to Moon orbital distance",
                           "X position (km)","Y position (km)",470))
                fig.update_yaxes(scaleanchor="x",scaleratio=1)
                st.plotly_chart(fig,use_container_width=True)
                st.caption("The ring depicts the Moon's orbital DISTANCE, not the actual Moon's location. "
                           "Reaching this ring is not sufficient to intercept the moving Moon.")
            except (ValueError,RuntimeError) as ex:
                st.warning(f"Could not render selected trajectory: {ex}")
        st.dataframe(table[["Flight_time_days","Delta_v_km_s"]].round(4),
                     hide_index=True,use_container_width=True)
        st.download_button("Download transfer Δv data (CSV)",
                table[["Flight_time_days","Delta_v_km_s"]].to_csv(index=False).encode(),
                "mee453_p3_transfer_dv.csv","text/csv")

    with tabs[2]:
        st.subheader("Propellant mass versus food and water")
        st.caption("Rocket equation sizing: spacecraft mass after the burn is taken as "
                   "the adjustable dry/delivered mass plus all food and water loaded for the chosen flight time. "
                   "No oxygen breathing supply, reserves, mission margins, hardware mass changes, "
                   "or spacecraft dry-mass optimization is included.")
        cp1,cp2,cp3=st.columns(3)
        cp1.metric("Selected propellant",f"{chosen.Propellant_kg/1000:,.2f} t")
        cp2.metric("Selected food + water",f"{chosen.Food_water_kg:,.1f} kg")
        cp3.metric("Modeled departure mass",f"{chosen.Total_departure_mass_kg/1000:,.2f} t")
        left,right=st.columns(2)
        with left:
            fig=go.Figure(go.Scatter(x=table.Flight_time_days,y=table.Propellant_kg/1000,
                mode="lines+markers",line=dict(color=COLORS["fuel"],width=3),
                name="Propellant mass"))
            fig.update_layout(**chart_layout("More Δv needs more propellant",
                "Flight time (days)","Required propellant (metric tons)"))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            fig=go.Figure(go.Scatter(x=table.Flight_time_days,y=table.Food_water_kg,
                mode="lines+markers",line=dict(color=COLORS["crew"],width=3),
                name="Food/water"))
            fig.update_layout(**chart_layout("Longer travel needs more consumables",
                "Flight time (days)","Food and water (kg)"))
            st.plotly_chart(fig,use_container_width=True)
        mass_table=table.copy()
        mass_table["Nonpropellant_mass_kg"]=mass_table.Delivered_mass_kg
        fig=go.Figure()
        fig.add_trace(go.Bar(x=table.Flight_time_days.astype(str),
                             y=table.Propellant_kg/1000,name="Propellant",
                             marker_color=COLORS["fuel"]))
        fig.add_trace(go.Bar(x=table.Flight_time_days.astype(str),
                             y=table.Delivered_mass_kg/1000,name="Dry spacecraft + provisions",
                             marker_color=COLORS["crew"]))
        fig.update_layout(barmode="stack",height=390,
                          xaxis_title="Flight time (days)",yaxis_title="Modeled mass (metric tons)",
                          title="Modeled spacecraft initial mass by transfer time")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(table.round(3),hide_index=True,use_container_width=True)
        st.download_button("Download crew mass and propellant CSV",
            table.to_csv(index=False).encode(),"mee453_p3_mass_tradeoff.csv","text/csv")
        st.markdown("**Interpretation:** Shorter trips require a larger transfer Δv, "
                    "which tends to increase propellant needs. Longer trips require more crew "
                    "supplies. With the default illustrative inputs, the propellant change "
                    "dominates food and water, but real mission sizing includes many other "
                    "mass, navigation, and risk constraints.")

    with tabs[3]:
        st.subheader("Extra credit — ±11-second engine ignition timing error")
        st.write("Assume the **start time** of the burn shifts relative to the Moon, while "
                 "burn magnitude and duration are unchanged. With the spacecraft initially "
                 "in a 93-minute orbit, even a small time shift changes its angular starting position.")
        approximate_error=timing_error_km(ignition_error,period)
        k1,k2,k3=st.columns(3)
        k1.metric("Ignition timing error",f"{ignition_error:+.1f} s")
        k2.metric("Signed downrange displacement",f"{approximate_error:+,.0f} km")
        k3.metric("Fraction of initial orbit",f"{100*abs(ignition_error)/(period*60):.3f}%")
        tdel=np.linspace(-30.,30.,301)
        miss=np.array([timing_error_km(t,period) for t in tdel])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=tdel,y=miss,mode="lines",
                    name="Downrange offset",line=dict(color=COLORS["orbit"],width=3)))
        fig.add_trace(go.Scatter(x=[-11.,11.],y=[timing_error_km(-11,period),
                    timing_error_km(11,period)],mode="markers+text",
                    text=["−11 s","+11 s"],textposition="top center",
                    name="Assignment offsets",marker=dict(color=COLORS["moon"],size=10)))
        fig.update_layout(**chart_layout("Estimated downrange timing sensitivity",
                    "Ignition time offset (s)","Signed lunar-distance arc offset (km)",440))
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"\Delta s \approx R_{\rm Moon}(n_{\rm LEO}-n_{\rm Moon})\,\Delta t")
        st.caption("This angular first-order approximation assumes a nominally aligned target, "
                   "a circular 93-minute parking orbit, a circular lunar orbit, and the same "
                   "velocity impulse shifted in time. It is NOT a numerical lunar-intercept "
                   "miss distance from a full spacecraft+Moon gravity trajectory. "
                   "An engine **burn-duration** error is a different scenario and would "
                   "change the delivered Δv as well.")
        st.markdown(f"At −11 s: **{timing_error_km(-11.,period):+,.0f} km**. "
                    f"At +11 s: **{timing_error_km(11.,period):+,.0f} km**.")

    with tabs[4]:
        st.subheader("How the equations connect")
        st.markdown("**1. Initial near-circular Earth orbit (Lee Eq. 9.49):**")
        st.latex(r"r_0=\left[\mu_E\left(\frac{T}{2\pi}\right)^2\right]^{1/3},"
                 r"\quad v_0=\sqrt{\mu_E/r_0}")
        st.markdown("**2. Earth–Moon gravitational acceleration:**")
        st.latex(r"\ddot{\mathbf r}=-\frac{\mu_E\mathbf r}{|\mathbf r|^3}"
                 r"+\mu_M\left[\frac{\mathbf r_M-\mathbf r}{|\mathbf r_M-\mathbf r|^3}"
                 r"-\frac{\mathbf r_M}{|\mathbf r_M|^3}\right]")
        st.markdown("The final term accounts for acceleration of Earth itself toward the Moon. "
                    "The Moon moves around Earth along an assumed circular path.")
        st.markdown("**3. Measure how noncircular the orbit becomes:**")
        st.latex(r"\mathbf e=\frac{[(v^2-\mu_E/r)\mathbf r-(\mathbf r\cdot\mathbf v)\mathbf v]}{\mu_E},"
                 r"\quad e=|\mathbf e|")
        st.markdown("**4. Finite tangential burn:**")
        st.latex(r"\ddot{\mathbf r}=-\mu_E\frac{\mathbf r}{r^3}"
                 r"+\frac{\Delta v}{t_b}\frac{\mathbf v}{|\mathbf v|}"
                 r"\quad(0<t<t_b)")
        st.markdown("After the burn, the thrust term is set to zero. A root solver "
                    "varies Δv until the first outward radial crossing of 384,400 km "
                    "occurs at the selected flight time. **It is not an actual lunar rendezvous solution.**")
        st.markdown("**5. Rocket equation (Lee Eq. 8.11):**")
        st.latex(r"\Delta v=g_0 I_{sp}\ln\left(\frac{m_0}{m_f}\right),"
                 r"\quad m_p=m_f\left[\exp\left(\frac{\Delta v}{g_0 I_{sp}}\right)-1\right]")
        st.markdown("**6. Provisions:**")
        st.latex(r"m_{\rm consumables}=N_{\rm crew}\,q_{\rm daily}\,t_{\rm flight}")
        st.markdown("**7. Simplified downrange timing offset:**")
        st.latex(r"\Delta s=R_{\rm Moon}\,(n_{\rm LEO}-n_{\rm Moon})\,\Delta t")
        st.divider()
        st.markdown("**Textbook:** T.-W. Lee, *Aerospace Propulsion*, "
                    "Chapter 8 (rocket equation, staging) and Chapter 9 (orbits and gravity). "
                    "Moon-perturbation, finite-burn ODE, and timing-error models are "
                    "additional numerical formulations, not claimed as verbatim book equations.")
        st.warning("Assumptions: circular lunar orbit, mean lunar distance, simplified "
                   "2D Earth–Moon gravity for passive orbit, Earth-only transfer coast, "
                   "constant burn acceleration tangential to instantaneous velocity, "
                   "no launch/ascent losses, no Moon intercept guidance, and no thermal/structural constraints.")


if __name__=="__main__":
    st.set_page_config(page_title="ArtemisLab — MEE 453 Problem 3",page_icon="🌙",layout="wide")
    render()
