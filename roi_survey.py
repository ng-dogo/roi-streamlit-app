import streamlit as st
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread, datetime as dt

# ─────────────── CONFIGURACIÓN INICIAL ────────────────
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
div[data-baseweb="slider"] > div > div{background:var(--primary);}
div[data-baseweb="slider"] > div > div > div{background:var(--primary);}
span[data-baseweb="tooltip"]{background:var(--primary)!important;color:#FFFFFF!important;border:none!important;}
div[data-baseweb="slider"] span{color:#FFFFFF!important;}
.stButton>button{background:var(--primary);color:#FFF;border:none;}
.stButton>button:disabled{background:#CCCCCC;color:#666;}
.main .block-container{padding-top:2rem;max-width:850px;margin:auto;}
</style>
""", unsafe_allow_html=True)

# ─────────────── VARIABLES DE ESTADO ────────────────
if "started" not in st.session_state:
    st.session_state.started = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# ─────────────── GUARDAR EN GOOGLE SHEETS ─────────────
def save_to_sheet(nombre, apellido, regulador, pais, fp, qs, fea):
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
        dt.datetime.now().isoformat(), nombre, apellido, regulador, pais, fp, qs, fea
    ])

# ─────────────── PANTALLA 1: DATOS PERSONALES ─────────────
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
# ─────────────── PANTALLA 2: ENCUESTA CON SLIDERS ─────────
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

    fp = st.slider(
        label="Financial Performance & Competitiveness (FPC)",
        min_value=0, max_value=100, value=33,
        help="Cost recovery, credit-worthiness and tariff competitiveness."
    )
    qs = st.slider(
        label="Quality of Service Delivery (QSD)",
        min_value=0, max_value=100, value=33,
        help="Reliability, losses and customer service quality."
    )
    fea = st.slider(
        label="Facilitating Electricity Access (FEA)",
        min_value=0, max_value=100, value=34,
        help="Policies, investment and progress expanding access."
    )

    total = fp + qs + fea

    progress_color = "#02593B" if total == 100 else "#d9534f"
    st.markdown(f"""
        <h5 style='margin-bottom:0.3rem;'>Total allocated</h5>
        <div style='background:#e0e0e0;border-radius:4px;height:18px;'>
            <div style='width:{min(total,100)}%;background:{progress_color};
                        height:18px;border-radius:4px;'></div>
        </div>
        <p style='margin-top:4px;'><b>{total}%</b> of 100%</p>
        """, unsafe_allow_html=True)

    if total < 100:
        st.warning("Increase some values to reach 100 %.")    
    elif total > 100:
        st.error("Reduce some values to reach 100 %.")    
    else:
        st.success("✅ Total is exactly 100 %!")

    st.markdown("---")

    df = pd.DataFrame({"Sub-index": ["FPC", "QSD", "FEA"], "Weight (%)": [fp, qs, fea]})
    df = df.sort_values("Weight (%)", ascending=False).set_index("Sub-index")

    st.markdown("### 📊 Weight distribution")
    st.bar_chart(df)

    st.markdown("### 🏁 Priority ranking")
    for i,(name,row) in enumerate(df.iterrows(),1):
        st.write(f"**{i}. {name} – {row['Weight (%)']} %**")

    st.markdown("---")

    if st.session_state.submitted:
        st.success("✅ You have already submitted your response. Thank you!")
    elif st.button("📤 Submit my weights", disabled=(total != 100)):
        u = st.session_state.user_info
        save_to_sheet(u["nombre"], u["apellido"], u["regulador"], u["pais"], fp, qs, fea)
        st.session_state.submitted = True
        st.success("✅ Thank you! Your response has been recorded.")

# ─────────────── RENDER APP ─────────────
if not st.session_state.started:
    intro()
else:
    survey()
