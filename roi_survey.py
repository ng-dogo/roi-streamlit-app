import streamlit as st
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread, datetime as dt

# ───────── CONFIG ─────────
st.set_page_config(page_title="ROI Weighting Survey", page_icon="⚡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
html, body, [class*="css"]{ font-family:'Montserrat',sans-serif; background:#FFFFFF !important; color:#1D1D1B; }
:root{ --primary:#02593B; }
/* Slider track y handle */
div[data-baseweb="slider"] > div > div{background:var(--primary);}
div[data-baseweb="slider"] > div > div > div{background:var(--primary);}
/* Tooltip + número */
span[data-baseweb="tooltip"]{background:var(--primary)!important;color:#FFFFFF!important;border:none!important;}
div[data-baseweb="slider"] span{color:#FFFFFF!important;}
/* Botón principal */
.stButton>button{background:var(--primary);color:#FFF;border:none;}
.stButton>button:disabled{background:#CCCCCC;color:#666;}
/* Contenedor */
.main .block-container{padding-top:2rem;max-width:850px;margin:auto;}
/* Logo centrado (usa alt del archivo) */
img[alt="ad_hoc_logo.png"] {display:block;margin-left:auto;margin-right:auto;}
/* Ocultar anclas de títulos */
a.anchor-link {display:none;}
</style>
""", unsafe_allow_html=True)

# ───────── STATE ─────────
for k, v in {
    "started": False,
    "submitted": False,
    "submitting": False,
    "user_info": {},
    "FPC": 33,   # valores iniciales
    "QSD": 33,
}.items():
    st.session_state.setdefault(k, v)

# ───────── SHEETS ─────────
def save_to_sheet(nombre, apellido, regulador, pais, fpc, qsd, fea):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    sh.append_row([dt.datetime.now().isoformat(), nombre, apellido, regulador, pais, fpc, qsd, fea])

# ───────── INTRO (datos) ─────────
def intro():
    st.image("ad_hoc_logo.png", width=160)
    st.title("Regulatory Outcome Index (ROI)")
    st.markdown("Before starting the survey, please enter your details:")

    nombre = st.text_input("First name")
    apellido = st.text_input("Last name")
    regulador = st.text_input("Regulatory entity")
    pais = st.text_input("Country")

    btn = st.button("🚀 Start survey", disabled=not all([nombre, apellido, regulador, pais]))
    if btn:
        st.session_state.user_info = {
            "nombre": nombre.strip(),
            "apellido": apellido.strip(),
            "regulador": regulador.strip(),
            "pais": pais.strip(),
        }
        st.session_state.started = True
        st.rerun()

# ───────── CALLBACKS SLIDERS ─────────
def cap_sliders(changed_key: str):
    """
    Mantiene FPC + QSD ≤ 100 ajustando el otro si hace falta.
    FEA se autocalcula como 100 - (FPC + QSD).
    """
    fpc = int(st.session_state.FPC)
    qsd = int(st.session_state.QSD)

    # Si se supera 100, recortamos el que NO cambió.
    if fpc + qsd > 100:
        if changed_key == "FPC":
            # recorto QSD
            qsd = 100 - fpc
            qsd = max(0, qsd)
            st.session_state.QSD = qsd
        else:
            # recorto FPC
            fpc = 100 - qsd
            fpc = max(0, fpc)
            st.session_state.FPC = fpc

# ───────── SURVEY ─────────
def survey():
    # Pantalla "gracias" si ya envió
    if st.session_state.submitted:
        st.image("ad_hoc_logo.png", width=160)
        st.title("✅ Thank you for your response")
        st.markdown("Your weights were successfully recorded.")
        u = st.session_state.user_info
        st.markdown(f"**Participant:** {u.get('nombre','')} {u.get('apellido','')} — {u.get('regulador','')} ({u.get('pais','')})")
        return

    st.image("ad_hoc_logo.png", width=160)
    st.title("ROI Index – Weighting Survey")

    with st.expander("ℹ️ What is the ROI index?"):
        st.markdown("""
**Regulatory Outcome Index (ROI)** measures how effectively a country’s electricity sector delivers outcomes
that matter to the public and investors. It has **three weighted sub-indices**:

- **FPC** – *Financial Performance & Competitiveness*  
  _Cost recovery, credit-worthiness and tariff competitiveness._
- **QSD** – *Quality of Service Delivery*  
  _Reliability, losses and customer service quality._
- **FEA** – *Facilitating Electricity Access*  
  _Policies, investment and progress expanding access._
        """)

    st.subheader("Distribute **exactly 100 %** across the three ROI sub-indices.")
    st.caption("Move the two sliders below. The third value adjusts automatically so that the total is always 100%.")

    # Sliders lado a lado
    c1, c2 = st.columns(2)
    with c1:
        st.slider(
            label="FPC – Financial Performance & Competitiveness",
            min_value=0, max_value=100, key="FPC",
            on_change=cap_sliders, args=("FPC",),
            help="Cost recovery, credit-worthiness and tariff competitiveness."
        )
        st.caption(f"Max allowed now: **{100 - st.session_state.QSD}%**")

    with c2:
        st.slider(
            label="QSD – Quality of Service Delivery",
            min_value=0, max_value=100, key="QSD",
            on_change=cap_sliders, args=("QSD",),
            help="Reliability, losses and customer service quality."
        )
        st.caption(f"Max allowed now: **{100 - st.session_state.FPC}%**")

    # FEA autocalculado
    fpc = int(st.session_state.FPC)
    qsd = int(st.session_state.QSD)
    fea = 100 - (fpc + qsd)
    fea = max(0, fea)  # por seguridad

    st.markdown("---")

    # Barra de total (siempre 100)
    st.markdown(
        f"""
        <h5 style='margin-bottom:0.3rem;'>Total allocated</h5>
        <div style='background:#e0e0e0;border-radius:4px;height:18px;'>
            <div style='width:100%;background:#02593B;height:18px;border-radius:4px;'></div>
        </div>
        <p style='margin-top:4px;'><b>100%</b> of 100%</p>
        """,
        unsafe_allow_html=True
    )

    # Tabla/Gráfico + ranking
    st.markdown("### 📊 Weight distribution")
    df = (pd.DataFrame({"Sub-index": ["FPC", "QSD", "FEA"], "Weight (%)": [fpc, qsd, fea]})
          .sort_values("Weight (%)", ascending=False)
          .set_index("Sub-index"))
    st.bar_chart(df)

    st.markdown("### 🏁 Priority ranking")
    for i, (name, row) in enumerate(df.iterrows(), 1):
        st.write(f"**{i}. {name} – {row['Weight (%)']} %**")

    st.markdown("---")

    # Enviar (con bloqueo anti-múltiple clic y deshabilitado después)
    disabled = st.session_state.submitting
    if st.button("📤 Submit my weights", disabled=disabled):
        if not st.session_state.submitting:
            try:
                st.session_state.submitting = True  # evita dobles envíos rápidos
                u = st.session_state.user_info
                save_to_sheet(u["nombre"], u["apellido"], u["regulador"], u["pais"], fpc, qsd, fea)
                st.session_state.submitted = True
                st.success("✅ Thank you! Your response has been recorded.")
                st.rerun()
            finally:
                # si hubo error, liberamos el lock para volver a intentar
                st.session_state.submitting = False

# ───────── RENDER ─────────
if not st.session_state.started:
    intro()
else:
    survey()
