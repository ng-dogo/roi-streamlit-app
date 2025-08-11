import streamlit as st
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread, datetime as dt
import uuid

# ─────────────── CONFIG ────────────────
st.set_page_config(page_title="ROI Weighting Survey", page_icon="⚡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

html, body, [class*="css"]{
    font-family:'Montserrat',sans-serif;
    background:#FFFFFF !important;
    color:#1D1D1B;
}

:root{ --primary:#02593B; }

/* Slider track y handle */
div[data-baseweb="slider"] > div > div{background:var(--primary);}
div[data-baseweb="slider"] > div > div > div{background:var(--primary);}

/* Tooltip + número */
span[data-baseweb="tooltip"]{
    background:var(--primary)!important;
    color:#FFFFFF!important;
    border:none!important;
}
div[data-baseweb="slider"] span{color:#FFFFFF!important;}

/* Botón principal */
.stButton>button{background:var(--primary);color:#FFF;border:none;}
.stButton>button:disabled{background:#CCCCCC;color:#666;}

/* Contenedor general */
.main .block-container{padding-top:2rem;max-width:850px;margin:auto;}

/* Logo centrado */
img[alt="ad_hoc_logo.png"] {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Ocultar íconos de ancla de títulos */
a.anchor-link { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────── STATE ────────────────
if "started" not in st.session_state:
    st.session_state.started = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ─────────────── GOOGLE SHEETS ─────────
def save_to_sheet(nombre, apellido, regulador, pais, fp, qs, fea, session_id):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    sh.append_row([
        dt.datetime.now().isoformat(),
        nombre, apellido, regulador, pais,
        int(fp), int(qs), int(fea),
        session_id
    ])

# ─────────────── INTRO ────────────────
def intro():
    st.image("ad_hoc_logo.png", width=160)
    st.title("Regulatory Outcome Index (ROI)")
    st.markdown("Before starting the survey, please enter your details:")

    nombre = st.text_input("First name")
    apellido = st.text_input("Last name")
    regulador = st.text_input("Regulatory entity")
    pais = st.text_input("Country")

    if st.button("🚀 Start survey", disabled=not all([nombre, apellido, regulador, pais])):
        st.session_state.user_info = {
            "nombre": nombre.strip(),
            "apellido": apellido.strip(),
            "regulador": regulador.strip(),
            "pais": pais.strip()
        }
        st.session_state.started = True
        st.rerun()

# ─────────────── SURVEY ───────────────
def survey():
    st.image("ad_hoc_logo.png", width=160)
    st.title("ROI Index – Weighting Survey")

    with st.expander("ℹ️ What is the ROI index?"):
        st.markdown("""
        The **Regulatory Outcome Index (ROI)** measures how effectively a country’s electricity sector delivers outcomes
        that matter to the public and investors.  
        It is composed of **three weighted sub-indices**:

        - **FPC** – *Financial Performance & Competitiveness*  
          _Cost recovery, credit-worthiness and tariff competitiveness._

        - **QSD** – *Quality of Service Delivery*  
          _Reliability, losses and customer service quality._

        - **FEA** – *Facilitating Electricity Access*  
          _Policies, investment and progress expanding access._
        """)

    st.subheader("Distribute **exactly 100 %** among the three ROI sub-indices.")
    st.markdown("---")

    # FPC
    fp = st.slider(
        label="Financial Performance & Competitiveness (FPC)",
        min_value=0, max_value=100, value=33,
        help="Cost recovery, credit-worthiness and tariff competitiveness."
    )

    # QSD tope dinámico
    qs_max = 100 - fp
    default_qs = min(33, qs_max)
    qs = st.slider(
        label="Quality of Service Delivery (QSD)",
        min_value=0, max_value=qs_max, value=default_qs,
        help="Reliability, losses and customer service quality.",
        key=f"qsd_{qs_max}"
    )

    # FEA calculado
    fea = 100 - (fp + qs)

    # Barra total
    st.markdown(
        f"""
        <h5 style='margin-bottom:0.3rem;'>Total allocated</h5>
        <div style='background:#e0e0e0;border-radius:4px;height:18px;'>
            <div style='width:100%;background:#02593B;height:18px;border-radius:4px;'></div>
        </div>
        <p style='margin-top:4px;'><b>100%</b> of 100% (FEA auto-calculated)</p>
        """,
        unsafe_allow_html=True
    )

    st.info(f"FEA (Facilitating Electricity Access) is automatically set to **{fea}%** to complete 100%.")

    st.markdown("---")

    # Gráfico + ranking
    df = (
        pd.DataFrame({"Sub-index": ["FPC", "QSD", "FEA"], "Weight (%)": [fp, qs, fea]})
        .sort_values("Weight (%)", ascending=False)
        .set_index("Sub-index")
    )

    st.markdown("### 📊 Weight distribution")
    st.bar_chart(df)

    st.markdown("### 🏁 Priority ranking")
    for i, (name, row) in enumerate(df.iterrows(), 1):
        st.write(f"**{i}. {name} – {row['Weight (%)']} %**")

    st.markdown("---")

    # Submit con anti‑doble‑click
    disabled_submit = st.session_state.saving or st.session_state.submitted
    btn = st.button("📤 Submit my weights", disabled=disabled_submit)

    if btn and not disabled_submit:
        st.session_state.saving = True
        try:
            u = st.session_state.user_info
            save_to_sheet(u["nombre"], u["apellido"], u["regulador"], u["pais"], fp, qs, fea, st.session_state.session_id)
            st.session_state.submitted = True
            # 🚀 MENSAJE INTEGRADO DE AGRADECIMIENTO
            st.success("✅ Thank you for your response! Your response has been recorded.")
            st.balloons()
        except Exception as e:
            st.error("⚠️ There was an error saving your response. Please try again.")
        finally:
            st.session_state.saving = False

    if st.session_state.submitted:
        st.caption("Response already submitted for this session.")

# ─────────────── RENDER ───────────────
if not st.session_state.started:
    intro()
else:
    survey()
