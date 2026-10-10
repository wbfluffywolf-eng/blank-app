import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp, trapezoid

MU = 3.986004418e14
RE = 6_371_000.0
G = 9.80665

st.set_page_config(page_title="OrbitLab | MEE 453", page_icon="🚀", layout="wide")
st.markdown("""<style>
.block-container {max-width:1450px;padding-top:1.2rem}
[data-testid="stMetric"] {background-color:rgba(70,135,180,.09);padding:15px;border:1px solid rgba(120,160,190,.22);border-radius:12px}
</style>""", unsafe_allow_html=True)
st.title("🚀 OrbitLab — MEE 453 Problem 2")
st.caption("Editable satellite orbit, multistage rocket design, and 33-second aerodynamic drag study.")

def orbit(T_min,e):
    T=T_min*60
    a=(MU*(T/(2*math.pi))**2)**(1/3)
    rp,ra=a*(1-e),a*(1+e)
    if rp<=RE: raise ValueError("This combination of period and eccentricity makes the orbit intersect Earth.")
    v=lambda r: math.sqrt(MU*(2/r-1/a))
    return a,rp,ra,v(rp),v(ra)

def size_stages(payload,eps,isps,dvs,ofs,names):
    carried=payload
    out=[]
    for i in reversed(range(len(isps))):
        mr=math.exp(dvs[i]/(isps[i]*G))
        denom=1/mr-eps
        if denom<=0:
            raise ValueError(f"Stage {i+1} is not feasible for this Isp, structural coefficient and Δv. Try more stages or higher Isp.")
        wet=carried*(1-eps)/denom
        dry_stage=(wet-carried)*eps
        prop=(wet-carried)*(1-eps)
        fuel=prop/(1+ofs[i])
        out.append(dict(stage=i+1,propellant=names[i],isp=isps[i],dv=dvs[i],ratio=mr,
                        fuel=fuel,oxidizer=prop-fuel,propellant_mass=prop,
                        structure=dry_stage,stack_mass=wet,carried=carried))
        carried=wet
    return list(reversed(out))

def rho_at(h):
    hs=np.arange(0,30000,2000,dtype=float)
    ratios=np.array([1.,.82168,.66885,.53887,.42921,.33756,.25464,.18600,
                     .13589,.099302,.072578,.052660,.038316,.027964,.020470])
    h=max(0.,float(h))
    return float(math.exp(np.interp(h,hs,np.log(1.225*ratios),
               left=np.log(1.225),right=np.log(1.225*ratios[-1])-(h-hs[-1])/6500)))

st.sidebar.header("Mission inputs")
period=st.sidebar.number_input("Orbital period (minutes)",30.,1500.,138.,1.)
payload=st.sidebar.number_input("Satellite payload (kg)",100.,1000000.,14530.,100.)
e=st.sidebar.number_input("Orbit eccentricity e",0.,.8,.05,.01,format="%.3f")
with st.sidebar.expander("Z-ID check (optional)"):
    st.caption("For your team's middle digits, 1 + 4 = 5, so e = 0.05. Keep IDs out of public code.")
    st.write("e = 0.0Y")
st.sidebar.subheader("Launch design")
eps=st.sidebar.slider("Structural coefficient (%)",5,40,18)/100.
loss=st.sidebar.number_input("Assumed launch losses excluding modeled drag (m/s)",0.,4000.,1500.,100.)
insertion=st.sidebar.selectbox("Speed reference",["Perigee","Apogee"])
num=st.sidebar.selectbox("Number of stages",[1,2,3,4],index=2)
names0=["LOX / RP-1","LOX / CH4","LOX / LH2","LOX / LH2"]
isp0=[300.,355.,450.,450.]
of0=[2.6,3.5,5.5,5.5]
shares0={1:[100],2:[45,55],3:[25,35,40],4:[20,25,25,30]}
names=[];isps=[];shares=[];ofs=[]
with st.sidebar.expander("Edit the individual stages",expanded=False):
    for i in range(num):
        st.markdown(f"**Stage {i+1}**")
        names.append(st.text_input(f"Propellants {i+1}",value=names0[i],key=f"n{num}{i}"))
        isps.append(st.number_input(f"Specific impulse (s) {i+1}",100.,600.,isp0[i],5.,key=f"i{num}{i}"))
        shares.append(st.number_input(f"Relative Δv share {i+1}",1,1000,shares0[num][i],1,key=f"s{num}{i}"))
        ofs.append(st.number_input(f"Oxidizer/fuel ratio {i+1}",.1,15.,of0[i],.1,key=f"o{num}{i}"))
try:
    a,rp,ra,vp,va=orbit(period,e)
    dv=(vp if insertion=="Perigee" else va)+loss
    dvs=np.array(shares,dtype=float)/sum(shares)*dv
    stages=size_stages(payload,eps,isps,dvs,ofs,names)
except ValueError as exc:
    st.error(str(exc)); st.stop()

df=pd.DataFrame(stages)
totalprop=df.propellant_mass.sum()
launch_mass=stages[0]["stack_mass"]
m1,m2,m3,m4=st.columns(4)
m1.metric("Perigee altitude",f"{(rp-RE)/1000:,.1f} km")
m2.metric("Apogee altitude",f"{(ra-RE)/1000:,.1f} km")
m3.metric("Launch mass",f"{launch_mass/1000:,.1f} t")
m4.metric("Total propellant",f"{totalprop/1000:,.1f} t")
st.info("The assignment fixes the orbital period, eccentricity, payload, and 18% structure coefficient, but not engine Isp, stage count, stage Δv shares, launch losses or rocket shape. All fuel and drag results below are model estimates based on editable assumptions.")

tab1,tab2,tab3,tab4=st.tabs(["🌎 Orbit explorer","🚀 Rocket stages","🌬️ Extra credit drag","📚 Equations"])

with tab1:
    st.subheader("Orbit shape and velocity")
    th=np.linspace(0,2*np.pi,361)
    r=a*(1-e*e)/(1+e*np.cos(th))
    x,y=r*np.cos(th)/1000,r*np.sin(th)/1000
    c1,c2=st.columns(2)
    with c1:
        theta=np.linspace(0,2*np.pi,180)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=RE/1000*np.cos(theta),y=RE/1000*np.sin(theta),
                         fill="toself",name="Earth",line=dict(color="#5488c0")))
        fig.add_trace(go.Scatter(x=x,y=y,name="Satellite orbit",line=dict(color="#00b5aa",width=3)))
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        fig.update_layout(height=410,xaxis_title="X (km)",yaxis_title="Y (km)",title="Orbital geometry")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        v=np.sqrt(MU*(2/(r)-1/a))/1000
        fig=go.Figure(go.Scatter(x=np.degrees(th),y=v,line=dict(color="#eea946",width=3)))
        fig.update_layout(height=410,title="Speed around orbit",xaxis_title="True anomaly (degrees)",yaxis_title="Speed (km/s)")
        st.plotly_chart(fig,use_container_width=True)
    st.write(f"Semi-major axis **{a/1000:,.2f} km** · perigee speed **{vp/1000:.3f} km/s** · apogee speed **{va/1000:.3f} km/s**.")
    st.latex(r"a=\left[\mu\left(\frac{T}{2\pi}\right)^2\right]^{1/3},\quad r_p=a(1-e),\quad r_a=a(1+e)")
    st.latex(r"v(r)=\sqrt{\mu\left(\frac{2}{r}-\frac{1}{a}\right)}")

with tab2:
    st.subheader("Fuel, oxidizer and structure by stage")
    fig=go.Figure()
    for col,label,color in [("fuel","Fuel","#20bfa9"),("oxidizer","Oxidizer","#458dd9"),("structure","Structure","#ae83e3")]:
        fig.add_trace(go.Bar(x=df.stage.astype(str),y=df[col]/1000,name=label,marker_color=color))
    fig.update_layout(barmode="stack",height=400,xaxis_title="Stage (bottom to top)",yaxis_title="Mass (metric tons)")
    st.plotly_chart(fig,use_container_width=True)
    view=df.rename(columns={"stage":"Stage","propellant":"Propellants","isp":"Isp (s)","dv":"Δv (m/s)",
               "fuel":"Fuel (kg)","oxidizer":"Oxidizer (kg)","propellant_mass":"Total propellant (kg)",
               "structure":"Structure (kg)","stack_mass":"Wet stack (kg)"})
    st.dataframe(view[["Stage","Propellants","Isp (s)","Δv (m/s)","Fuel (kg)","Oxidizer (kg)",
               "Total propellant (kg)","Structure (kg)","Wet stack (kg)"]].round(2),hide_index=True,use_container_width=True)
    st.download_button("Download stage results CSV",view.to_csv(index=False).encode(),"mee453_stage_results.csv","text/csv")
    st.latex(r"\Delta v_i=I_{sp,i}g_0\ln(M_{0,i}/M_{b,i})")
    st.latex(r"\varepsilon=\frac{M_s}{M_s+M_p},\qquad M_0=\frac{L(1-\varepsilon)}{1/MR-\varepsilon}")
    st.caption("Stage masses are computed from the upper stage down. Oxygen is included in propellant. Δv shares are assumptions, not unique optimal choices.")

with tab3:
    st.subheader("Extra credit — first 33 seconds of vertical launch")
    ca,cb,cc,cd=st.columns(4)
    with ca: duration=st.number_input("Duration (s)",1.,180.,33.,1.)
    with cb: burn=st.number_input("Stage 1 burn duration (s)",35.,500.,120.,5.)
    with cc: diameter=st.number_input("Vehicle diameter (m)",.5,15.,4.,.5)
    with cd: Cd=st.number_input("Drag coefficient",.1,2.,.5,.05)
    A=math.pi*diameter**2/4
    mdot=stages[0]["propellant_mass"]/burn
    thrust=mdot*isps[0]*G
    def sim(drag):
        def deriv(t,sv):
            h,v=sv
            mass=max(1.,launch_mass-mdot*t)
            grav=MU/(RE+max(0,h))**2
            D=.5*rho_at(h)*Cd*A*v*abs(v) if drag else 0.
            return [v,(thrust-D)/mass-grav]
        ts=np.linspace(0,duration,400)
        sol=solve_ivp(deriv,(0,duration),(0.,0.),t_eval=ts,rtol=2e-7,atol=1e-8)
        return sol.t,sol.y[0],sol.y[1]
    t,h,v=sim(True)
    _,h0,v0=sim(False)
    D=.5*np.array([rho_at(hi) for hi in h])*Cd*A*v*np.abs(v)
    mass=launch_mass-mdot*t
    drag_dv=float(trapezoid(D/mass,t))
    try:
        newstages=size_stages(payload,eps,isps,dvs+np.array(shares)/sum(shares)*drag_dv,ofs,names)
        extra=sum(s["propellant_mass"] for s in newstages)-totalprop
    except ValueError:
        extra=float("nan")
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Final altitude",f"{h[-1]:,.1f} m")
    k2.metric("Peak drag",f"{D.max()/1000:,.1f} kN")
    k3.metric("Drag Δv",f"{drag_dv:,.3f} m/s")
    k4.metric("Extra propellant",f"{extra:,.1f} kg")
    l,rcol=st.columns(2)
    with l:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=t,y=v,mode="lines",name="With drag"))
        fig.add_trace(go.Scatter(x=t,y=v0,mode="lines",name="Without drag"))
        fig.update_layout(height=340,title="Velocity comparison",xaxis_title="Time (s)",yaxis_title="Velocity (m/s)")
        st.plotly_chart(fig,use_container_width=True)
    with rcol:
        fig=go.Figure(go.Scatter(x=t,y=D/1000,fill="tozeroy",name="Drag (kN)"))
        fig.update_layout(height=340,title="Drag versus time",xaxis_title="Time (s)",yaxis_title="Drag force (kN)")
        st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=h,name="With drag"))
    fig.add_trace(go.Scatter(x=t,y=h0,name="Without drag"))
    fig.update_layout(height=300,title="Ascent altitude",xaxis_title="Time (s)",yaxis_title="Altitude (m)")
    st.plotly_chart(fig,use_container_width=True)
    ascent=pd.DataFrame({"time_s":t,"altitude_m":h,"velocity_mps":v,"velocity_no_drag_mps":v0,
                        "density_kg_m3":[rho_at(hi) for hi in h],"drag_N":D})
    st.download_button("Download ascent simulation CSV",ascent.to_csv(index=False).encode(),"mee453_drag_simulation.csv","text/csv")
    st.caption("A vertical constant-thrust, constant mass-flow model is integrated numerically. Additional propellant is estimated by re-sizing all stages to recover the integrated drag velocity loss. It does not replace a full gravity-turn or orbit-insertion simulation.")

with tab4:
    st.subheader("Textbook connections")
    st.markdown("""
**Orbital period and size:** Lee, *Aerospace Propulsion*, Eq. (9.49).  
**Perigee and apogee:** Eqs. (9.37)–(9.40).  
**Orbital speed:** Vis-viva energy relation, Chapter 9.  
**Rocket equation:** Eq. (8.11). **Staging:** Eq. (8.24).  
**Mass balance and structural fraction:** Eqs. (8.18)–(8.21).  
**Air density:** Appendix A (tabulated pressure and density ratios).  
**Drag:** D = ½ ρ Cᴅ A v² (standard aerodynamic relation).  
**Vertical acceleration:** dv/dt = [thrust − drag]/mass − μ/(Rₑ+h)².

The given structural coefficient is applied to each stage here. Fuel type, Isp, loss allowance, stage count, Δv allocation, diameter, Cᴅ and launch timeline must be assumed or researched to obtain numerical results.
""")
