# app.py
import streamlit as st
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
import gspread, datetime as dt
import uuid

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="ROI Weighting Prototype", page_icon="⚡", layout="centered")

PRIMARY = "#02593B"
FONT_URL = "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap"

st.markdown(f"""
<style>
@import url('{FONT_URL}');
html, body, [class*="css"]{{font-family:'Montserrat',sans-serif;background:#FFFFFF;color:#1D1D1B}}
:root{{ --primary:{PRIMARY};}}
/* Sliders */
div[data-baseweb="slider"] > div > div{{background:var(--primary);}}
div[data-baseweb="slider"] > div > div > div{{background:var(--primary);}}
span[data-baseweb="tooltip"]{{background:var(--primary)!important;color:#fff!important;border:none!important}}
div[data-baseweb="slider"] span{{color:#fff!important}}
/* Buttons */
.stButton>button{{background:var(--primary);color:#FFF;border:none;border-radius:10px;padding:.5rem 1rem}}
.stButton>button:disabled{{background:#CCCCCC;color:#666}}
/* Cards */
.block{{border:1px solid #eee;border-radius:14px;padding:16px;margin:6px 0}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:999px;background:#EEF6F0;color:{PRIMARY};font-weight:600}}
.kpi{{background:#f8faf9;border:1px solid #eef3f0;border-radius:12px;padding:12px;text-align:center}}
/* Container */
.main .block-container{{padding-top:1.2rem;max-width:880px;margin:auto}}
/* Hide anchor icons */
a.anchor-link{{display:none}}
/* Center logo */
img[alt="ad_hoc_logo.png"]{{display:block;margin-left:auto;margin-right:auto}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── SESSION STATE ───────────────────────────
def _init_state():
    defaults = {
        "started": False,
        "submitted": False,
        "saving": False,
        "session_id": str(uuid.uuid4()),
        "user": {"nombre":"", "apellido":"", "regulador":"", "pais":"", "email":""},
        "weights_bap": {"FPC":33, "QSD":33, "FEA":34},
        "weights_ahp": {"FPC":33.3, "QSD":33.3, "FEA":33.3, "CR": None},
        "method_choice": "BAP (Budget)",
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

_init_state()

# ─────────────────────────── DATA PERSISTENCE ───────────────────────────
def save_to_sheet(payload: dict):
    """
    Writes one row to the first sheet of your Google Sheet.
    Requires in st.secrets: gs_email, gs_key, sheet_id
    """
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
        payload.get("nombre",""),
        payload.get("apellido",""),
        payload.get("regulador",""),
        payload.get("pais",""),
        payload.get("email",""),
        int(round(payload.get("FPC",0))),
        int(round(payload.get("QSD",0))),
        int(round(payload.get("FEA",0))),
        payload.get("method",""),
        payload.get("session_id","")
    ])

# ─────────────────────────── HELPERS ───────────────────────────
CRITERIA = ["FPC","QSD","FEA"]

def show_logo():
    try:
        st.image("ad_hoc_logo.png", width=150)
    except:
        st.markdown(f"<span class='badge'>ROI Prototype</span>", unsafe_allow_html=True)

def stepper(current_idx:int):
    labels = ["Details", "Budget", "AHP", "Sandbox", "Review", "Submit"]
    cols = st.columns(len(labels))
    for i, c in enumerate(cols):
        with c:
            dot = "🟢" if i <= current_idx else "⚪"
            st.markdown(f"**{dot} {labels[i]}**")

def as_df(weights: dict) -> pd.DataFrame:
    return pd.DataFrame({"Sub-index": CRITERIA, "Weight (%)":[weights["FPC"],weights["QSD"],weights["FEA"]]}).set_index("Sub-index")

def bar(weights: dict, height=220):
    df = as_df(weights).sort_values("Weight (%)", ascending=False)
    st.bar_chart(df, height=height)

def doughnut(weights: dict):
    c1,c2,c3 = st.columns(3)
    for (label,val),cx in zip(weights.items(), [c1,c2,c3]):
        with cx:
            st.markdown(f"<div class='kpi'><div style='font-size:28px;font-weight:700'>{int(round(val))}%</div><div style='opacity:.8'>{label}</div></div>", unsafe_allow_html=True)

def sum_to_100(a,b):
    a = max(0, min(100, a))
    b = max(0, min(100-a, b))
    c = 100 - (a+b)
    return a,b,c

def ahp_weights(r12, r13, r23):
    """
    AHP 3x3 via geometric means (approximation to principal eigenvector).
    r12 = FPC vs QSD ; r13 = FPC vs FEA ; r23 = QSD vs FEA (Saaty scale, reciprocal matrix)
    """
    A = np.array([
        [1,   r12, r13],
        [1/r12, 1, r23],
        [1/r13, 1/r23, 1]
    ], dtype=float)
    gm = np.prod(A, axis=1)**(1/3)
    w = gm/np.sum(gm)
    Aw = A.dot(w)
    lam = np.sum(Aw / w) / 3.0
    CI = (lam - 3) / (3 - 1)  # n=3
    RI = 0.58                 # Random Index for n=3
    CR = CI/RI if RI > 0 else None
    return {"FPC": float(w[0]*100), "QSD": float(w[1]*100), "FEA": float(w[2]*100), "CR": float(CR)}

def saaty_select():
    """
    Symmetric selector for Saaty scale, left vs right:
    returns value in {1/9,1/7,1/5,1/3,1,3,5,7,9} meaning 'left compared to right'.
    """
    options = [
        ("Much less important (1/9)", 1/9),
        ("Less important (1/7)", 1/7),
        ("Moderately less (1/5)", 1/5),
        ("Slightly less (1/3)", 1/3),
        ("Equal (1)", 1),
        ("Slightly more (3)", 3),
        ("Moderately more (5)", 5),
        ("More important (7)", 7),
        ("Much more important (9)", 9),
    ]
    label_default = "Equal (1)"
    label = st.select_slider("", options=[o[0] for o in options], value=label_default)
    val = dict(options)[label]
    return val, label

def lock_weights(method_name, weights_dict):
    st.session_state.method_choice = method_name
    total = sum([weights_dict[k] for k in CRITERIA])
    if total <= 0: total = 1
    for k in CRITERIA:
        weights_dict[k] = max(0, float(weights_dict[k])) * 100.0/total
    if "BAP" in method_name:
        st.session_state.weights_bap = {k: float(weights_dict[k]) for k in CRITERIA}
    elif "AHP" in method_name:
        st.session_state.weights_ahp = {k: float(weights_dict[k]) for k in CRITERIA} | {"CR": weights_dict.get("CR")}

# ─────────────────────────── PAGES ───────────────────────────
def page_intro():
    show_logo()
    st.title("Regulatory Outcome Index (ROI) – Weighting Prototype")
    st.markdown("Fill in your details to begin. Your preferences are **visualized live** and stored on submit.")
    with st.form("intro"):
        c1,c2 = st.columns(2)
        with c1:
            nombre = st.text_input("First name")
            regulador = st.text_input("Regulatory entity")
        with c2:
            apellido = st.text_input("Last name")
            pais = st.text_input("Country")
        email = st.text_input("Email (optional)")
        consent = st.checkbox("I consent to store these details together with my ROI weights.")
        start = st.form_submit_button("🚀 Start", disabled=not(consent and nombre and apellido and regulador and pais))
    if start:
        st.session_state.user = {"nombre":nombre.strip(), "apellido":apellido.strip(), "regulador":regulador.strip(), "pais":pais.strip(), "email":email.strip()}
        st.session_state.started = True
        st.success("Great. You can move to **Budget** from the sidebar.")

    st.markdown("---")
    with st.expander("ℹ️ What does ROI measure?"):
        st.markdown("""
- **FPC** – *Financial Performance & Competitiveness*  
- **QSD** – *Quality of Service Delivery*  
- **FEA** – *Facilitating Electricity Access*
        """)

def page_bap():
    show_logo()
    stepper(1)
    st.header("⚖️ Budget (BAP): distribute **100%** across sub-indices")
    a = st.slider("FPC – Finance & Competitiveness", 0, 100, int(round(st.session_state.weights_bap["FPC"])))
    b_max = 100 - a
    b = st.slider("QSD – Quality of Service", 0, b_max, min(int(round(st.session_state.weights_bap["QSD"])), b_max), key=f"qsd_{b_max}")
    a,b,c = sum_to_100(a,b)
    st.info(f"FEA – Access is auto-adjusted to **{c}%** to close 100%.")
    weights = {"FPC":a, "QSD":b, "FEA":c}
    st.markdown("#### Distribution")
    bar(weights)
    st.markdown("#### Quick KPIs")
    doughnut(weights)
    st.button("✅ Use these weights (BAP)", on_click=lock_weights, args=("BAP (Budget)", weights))

def page_ahp():
    show_logo()
    stepper(2)
    st.header("🔢 AHP: Pairwise comparisons (Saaty scale)")
    st.caption("Choose the **intensity of preference** for each pair. We compute weights and **Consistency Ratio (CR)** live.")
    st.markdown("**FPC vs QSD** (the choice expresses how much FPC is preferred to QSD)")
    r12, _ = saaty_select()
    st.markdown("**FPC vs FEA** (preference of FPC over FEA)")
    r13, _ = saaty_select()
    st.markdown("**QSD vs FEA** (preference of QSD over FEA)")
    r23, _ = saaty_select()
    w = ahp_weights(r12, r13, r23)
    st.markdown("#### Resulting weights")
    bar({"FPC":w["FPC"], "QSD":w["QSD"], "FEA":w["FEA"]})
    st.markdown(f"**CR (Consistency Ratio):** `{w['CR']:.3f}`  " + ("✅ Acceptable (≤ 0.10)" if w["CR"] <= 0.10 else "⚠️ Please review your pairwise comparisons (ideal ≤ 0.10)"))
    st.button("✅ Use these weights (AHP)", disabled=(w["CR"]>0.10), on_click=lock_weights, args=("AHP (Consistent)", w))

def page_sandbox():
    show_logo()
    stepper(3)
    st.header("🧪 Sandbox: try methods and live visuals")
    method = st.radio("Method to visualize", ["BAP (Budget)","AHP (Consistent)","Equal (33/33/34)"], horizontal=True)
    if method == "BAP (Budget)":
        weights = st.session_state.weights_bap
    elif method == "AHP (Consistent)":
        weights = st.session_state.weights_ahp
    else:
        weights = {"FPC":33, "QSD":33, "FEA":34}
    st.subheader("📊 Sorted bars")
    bar(weights, height=260)
    st.subheader("🔘 Quick KPIs")
    doughnut(weights)

    st.markdown("---")
    st.subheader("✍️ Fine-tuning (optional)")
    st.caption("If you tweak manually, we’ll keep the total at 100 automatically.")
    c1,c2 = st.columns(2)
    with c1:
        fpc = st.number_input("FPC (%)", 0, 100, int(round(weights["FPC"])))
    with c2:
        qsd = st.number_input("QSD (%)", 0, 100-int(round(fpc)), int(round(weights["QSD"])))
    fpc,qsd,fea = sum_to_100(fpc,qsd)
    w2 = {"FPC":fpc,"QSD":qsd,"FEA":fea}
    st.markdown("#### Preview")
    bar(w2)
    if st.button("✅ Apply these weights to my session"):
        lock_weights("BAP (Budget)" if method!="AHP (Consistent)" else "AHP (Consistent)", w2)
        st.success("Weights updated in your session.")

def page_review():
    show_logo()
    stepper(4)
    st.header("📝 Review")
    st.markdown("Confirm your details and final distribution before submitting.")
    u = st.session_state.user
    w = st.session_state.weights_ahp if "AHP" in st.session_state.method_choice else st.session_state.weights_bap
    st.markdown("#### Your details")
    c1,c2 = st.columns(2)
    with c1:
        st.write(f"**Name:** {u['nombre']} {u['apellido']}")
        st.write(f"**Regulatory entity:** {u['regulador']}")
        st.write(f"**Email:** {u['email'] or '—'}")
    with c2:
        st.write(f"**Country:** {u['pais']}")
        st.write(f"**Chosen method:** {st.session_state.method_choice}")
    st.markdown("#### Your weights")
    bar(w)
    doughnut(w)

def page_submit():
    show_logo()
    stepper(5)
    st.header("📤 Submit")
    if st.session_state.submitted:
        st.success("✅ Thank you! Your response for this session has already been recorded.")
        st.caption("If you need to submit again, reload the app to start a new session.")
        return
    st.markdown("On submit, your details and weights will be stored.")
    w = st.session_state.weights_ahp if "AHP" in st.session_state.method_choice else st.session_state.weights_bap
    disabled = st.session_state.saving or not st.session_state.started
    if st.button("Submit now", disabled=disabled):
        st.session_state.saving = True
        try:
            payload = st.session_state.user | w | {
                "method": st.session_state.method_choice,
                "session_id": st.session_state.session_id
            }
            save_to_sheet(payload)
            st.session_state.submitted = True
            st.success("✅ Thank you for your response! Your submission has been recorded.")
            st.balloons()
        except Exception as e:
            st.error("⚠️ Error while saving. Please check your connection/credentials and try again.")
        finally:
            st.session_state.saving = False

# ─────────────────────────── ROUTER ───────────────────────────
def sidebar_nav():
    st.sidebar.markdown("### 🧭 Navigation")
    pages = {
        "🏁 Home": page_intro,
        "⚖️ Budget (BAP)": page_bap,
        "🔢 AHP": page_ahp,
        "🧪 Sandbox": page_sandbox,
        "📝 Review": page_review,
        "📤 Submit": page_submit,
    }
    default = "🏁 Home" if not st.session_state.started else "⚖️ Budget (BAP)"
    choice = st.sidebar.radio("", list(pages.keys()), index=list(pages.keys()).index(default))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Session:** `{st.session_state.session_id[:8]}`")
    if st.session_state.submitted:
        st.sidebar.success("Submitted ✅")
    return pages[choice]

# ─────────────────────────── RENDER ───────────────────────────
Page = sidebar_nav()
Page()
