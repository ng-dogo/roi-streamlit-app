# app.py
import streamlit as st
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread, datetime as dt
import uuid

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="ROI Weighting Survey", page_icon="⚡", layout="centered")

st.markdown("""
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
/* sliders */
div[data-baseweb="slider"] > div > div{background:var(--primary);}
div[data-baseweb="slider"] > div > div > div{background:var(--primary);}
span[data-baseweb="tooltip"]{background:var(--primary)!important;color:#fff!important;border:none!important}
div[data-baseweb="slider"] span{color:#fff!important}
/* buttons */
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:10px;padding:.5rem 1rem}
.stButton>button:disabled{background:#ccc;color:#666}
/* container */
.main .block-container{max-width:850px}
</style>
""", unsafe_allow_html=True)

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "user" not in st.session_state:
    st.session_state.user = {"first":"","last":"","entity":"","country":""}
if "fp" not in st.session_state:  # FPC
    st.session_state.fp = 33
if "qs" not in st.session_state:  # QSD
    st.session_state.qs = 33

# ─────────────── SHEETS ──────────────
def save_to_sheet(u, fp, qs, fea, session_id):
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
        u["first"], u["last"], u["entity"], u["country"],
        int(fp), int(qs), int(fea),
        session_id
    ])

# ─────────────── HEADER ──────────────
st.image("ad_hoc_logo.png", width=140, caption=None)
st.title("Regulatory Outcome Index (ROI) – Weighting")

tabs = st.tabs(["🎛️ Weights", "⚡ Quick presets"])

# ─────────────── TAB 1: Weights (original) ──────────────
with tabs[0]:
    st.subheader("Your details")
    c1,c2 = st.columns(2)
    with c1:
        first = st.text_input("First name", value=st.session_state.user["first"])
        entity = st.text_input("Regulatory entity", value=st.session_state.user["entity"])
    with c2:
        last = st.text_input("Last name", value=st.session_state.user["last"])
        country = st.text_input("Country", value=st.session_state.user["country"])

    st.session_state.user = {
        "first": first.strip(), "last": last.strip(),
        "entity": entity.strip(), "country": country.strip()
    }

    st.markdown("---")
    st.subheader("Distribute exactly **100%** among the three ROI sub-indices")
    with st.expander("What are these sub-indices?", expanded=False):
        st.markdown("""
- **FPC** – *Financial Performance & Competitiveness*  
- **QSD** – *Quality of Service Delivery*  
- **FEA** – *Facilitating Electricity Access*
        """)

    # Sliders (keep original logic)
    fp = st.slider("FPC – Finance & Competitiveness", 0, 100, st.session_state.fp)
    qs_max = 100 - fp
    default_qs = min(st.session_state.qs, qs_max)
    qs = st.slider("QSD – Quality of Service", 0, qs_max, default_qs, key=f"qsd_{qs_max}")
    fea = 100 - (fp + qs)

    # persist current slider positions
    st.session_state.fp, st.session_state.qs = fp, qs

    # total bar + info
    st.markdown(
        """
        <h5 style='margin-bottom:0.3rem;'>Total allocated</h5>
        <div style='background:#e0e0e0;border-radius:4px;height:18px;'>
            <div style='width:100%;background:#02593B;height:18px;border-radius:4px;'></div>
        </div>
        <p style='margin-top:4px;'><b>100%</b> of 100% (FEA auto-calculated)</p>
        """,
        unsafe_allow_html=True
    )
    st.info(f"FEA is automatically set to **{fea}%** to complete 100%.")

    st.markdown("---")
    # Bar chart (unchanged)
    df = (
        pd.DataFrame({"Sub-index": ["FPC","QSD","FEA"], "Weight (%)":[fp, qs, fea]})
        .sort_values("Weight (%)", ascending=False)
        .set_index("Sub-index")
    )
    st.markdown("### 📊 Weight distribution")
    st.bar_chart(df)

    st.markdown("### 🏁 Priority ranking")
    for i, (name, row) in enumerate(df.iterrows(), 1):
        st.write(f"**{i}. {name} – {row['Weight (%)']} %**")

    st.markdown("---")
    # Submit
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted or
        not all(st.session_state.user.values())
    )
    if st.button("📤 Submit my weights", disabled=disabled_submit):
        st.session_state.saving = True
        try:
            save_to_sheet(st.session_state.user, fp, qs, fea, st.session_state.session_id)
            st.session_state.submitted = True
            st.success("✅ Thank you for your response! Your submission has been recorded.")
            st.balloons()
        except Exception:
            st.error("⚠️ There was an error saving your response. Please try again.")
        finally:
            st.session_state.saving = False

    if st.session_state.submitted:
        st.caption("Response already submitted for this session.")

# ─────────────── TAB 2: Quick presets (simple extra) ──────────────
with tabs[1]:
    st.subheader("Quick presets (optional)")
    st.caption("Apply a preset or a quick priority ranking. Sliders update automatically.")

    # Preset scenarios (set FPC & QSD; FEA auto-completes)
    presets = {
        "Equal (33/33/34)": (33, 33),
        "Finance-focused (50/30/20)": (50, 30),
        "Service-focused (30/50/20)": (30, 50),
        "Access-focused (30/20/50)": (30, 20),
    }
    preset = st.selectbox("Preset scenarios", list(presets.keys()))
    if st.button("Apply preset"):
        st.session_state.fp, st.session_state.qs = presets[preset]
        st.rerun()

    st.markdown("#### Or: quick priority ranking → weights 50/30/20")
    # Simple ranking with unique choices
    options = ["FPC","QSD","FEA"]
    top = st.selectbox("1st priority (50%)", options, index=0, key="rank1")
    second_opts = [o for o in options if o != top]
    second = st.selectbox("2nd priority (30%)", second_opts, index=0, key="rank2")
    third = [o for o in options if o not in (top, second)][0]
    st.write(f"3rd priority (20%): **{third}**")

    if st.button("Apply ranking (50/30/20)"):
        new_fp, new_qs = st.session_state.fp, st.session_state.qs
        mapping = {top:50, second:30, third:20}
        # Convert mapping to (FPC, QSD); FEA comes from 100 - (FPC+QSD)
        new_fp = mapping["FPC"]
        new_qs = mapping["QSD"]
        st.session_state.fp, st.session_state.qs = new_fp, new_qs
        st.success("Ranking applied. Check the Weights tab.")
        st.rerun()