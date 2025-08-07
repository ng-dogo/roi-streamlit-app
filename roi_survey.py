import streamlit as st
import pandas as pd

# ───────────────── CONFIG GLOBAL ──────────────────────────────
st.set_page_config(page_title="ROI Weighting Survey",
                   page_icon="⚡",
                   layout="centered")

# ───────────────── CSS CORPORATIVO ────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    html, body, [class*="css"]{
        font-family:'Montserrat',sans-serif;
        background:#FFFFFF !important;
        color:#1D1D1B;
    }
    :root{ --primary:#02593B; }   /* verde corporativo */

    /* Slider track y handle */
    div[data-baseweb="slider"] > div > div{background:var(--primary);}
    div[data-baseweb="slider"] > div > div > div{background:var(--primary);}

    /* Tooltip + número encima del handle */
    span[data-baseweb="tooltip"]{
        background:var(--primary)!important;
        color:#FFFFFF!important;
        border:none!important;
    }
    div[data-baseweb="slider"] span{
        color:#FFFFFF!important;  /* Texto del número forzado a blanco */
    }

    /* Botón principal */
    .stButton>button{background:var(--primary);color:#FFF;border:none;}
    .stButton>button:disabled{background:#CCCCCC;color:#666;}

    /* Centrar contenedor principal */
    .main .block-container{padding-top:2rem;max-width:850px;margin:auto;}
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────── STATE: flujo intro → encuesta ─────────────
if "started" not in st.session_state:
    st.session_state.started = False

# ───────────────── PANTALLA INTRO ─────────────────────────────
def intro():
    st.image("ad_hoc_logo.png", width=160)        # ← coloca tu logo PNG
    st.title("Regulatory Outcome Index (ROI)")
    st.markdown(
        """
        The **ROI** measures how effectively a country’s electricity sector delivers
        outcomes that matter to the public and investors.  
        It is composed of **three weighted sub-indices**:

        1. **FPC** – Financial Performance & Competitiveness  
        2. **QSD** – Quality of Service Delivery (Commercial & Technical)  
        3. **FEA** – Facilitating Electricity Access  

        Click **Start survey** to assign weights (they must sum to 100 %).
        """
    )
    with st.expander("Sub-index descriptions", expanded=False):
        st.markdown(
            """
            **FPC** – Cost recovery, credit-worthiness and tariff competitiveness.  
            **QSD** – Reliability, losses and customer service quality.  
            **FEA** – Policies, investment and progress expanding access.
            """
        )
    if st.button("🚀 Start survey"):
        st.session_state.started = True            # Streamlit reruns automáticamente

# ───────────────── PANTALLA ENCUESTA ─────────────────────────
def survey():
    st.image("ad_hoc_logo.png", width=160)
    st.title("ROI Index – Weighting Survey")
    st.subheader("Distribute **exactly 100 %** among the three ROI sub-indices.")
    st.markdown("---")

    # ── Sliders
    fp  = st.slider("Financial Performance & Competitiveness (FPC)", 0, 100, 33)
    qs  = st.slider("Quality of Service Delivery (QSD)",             0, 100, 33)
    fea = st.slider("Facilitating Electricity Access (FEA)",          0, 100, 34)
    total = fp + qs + fea

    # ── Barra-termómetro del total
    progress_color = "#02593B" if total == 100 else "#d9534f"
    st.markdown(
        f"""
        <h5 style='margin-bottom:0.3rem;'>Total allocated</h5>
        <div style='background:#e0e0e0;border-radius:4px;height:18px;'>
            <div style='width:{min(total,100)}%;background:{progress_color};
                        height:18px;border-radius:4px;'></div>
        </div>
        <p style='margin-top:4px;'><b>{total}%</b> of 100%</p>
        """,
        unsafe_allow_html=True
    )

    # Mensaje de validación debajo de la barra
    if total < 100:
        st.warning("Increase some values to reach 100 %.")
    elif total > 100:
        st.error("Reduce some values to reach 100 %.")
    else:
        st.success("✅ Total is exactly 100 %!")

    st.markdown("---")

    # ── Gráfico + ranking
    df = (pd.DataFrame({"Sub-index": ["FPC", "QSD", "FEA"],
                        "Weight (%)": [fp, qs, fea]})
          .sort_values("Weight (%)", ascending=False)
          .set_index("Sub-index"))

    st.markdown("### 📊 Weight distribution")
    st.bar_chart(df)

    st.markdown("### 🏁 Priority ranking")
    for i,(name,row) in enumerate(df.iterrows(),1):
        st.write(f"**{i}. {name} – {row['Weight (%)']} %**")

    st.markdown("---")

    # ── Submit (habilitado solo si total = 100)
    if st.button("📤 Submit my weights", disabled=(total!=100)):
        st.success("✅ Thank you! Your response has been recorded.")
        # Aquí podrías guardar las respuestas en CSV, Sheets, etc.

# ───────────────── RENDER SEGÚN ESTADO ─────────────────────
if st.session_state.started:
    survey()
else:
    intro()
