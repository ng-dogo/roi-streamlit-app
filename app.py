# app.py
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt, re, os
from typing import Tuple
from google.oauth2.service_account import Credentials
import gspread

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

# Dark-mode friendly CSS + hide number_input spinners
CSS = """
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container { max-width: 980px; }
.row { padding: .45rem .6rem; border-radius: 10px; border: 1px solid rgba(128,128,128,.20); margin-bottom: .45rem; }
.name { font-weight: 600; margin-bottom: .35rem; }
hr { border: none; border-top: 1px solid rgba(128,128,128,.2); margin: 1rem 0; }
.stButton>button { border-radius: 8px; padding: .25rem .55rem; }
.center-cell { display:flex; align-items:center; justify-content:center; }
.value-input .stNumberInput input { text-align: center; font-weight: 600; }
.value-input .stNumberInput { width: 8rem; }
.value-input input[type=number]::-webkit-outer-spin-button,
.value-input input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.value-input input[type=number] { -moz-appearance: textfield; } /* Firefox */
.badge { display:inline-block; padding:.18rem .55rem; border-radius:999px; background: rgba(127,127,127,.15); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # expects columns: component,weight
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "components" not in st.session_state:
    st.session_state.components = []
if "points" not in st.session_state:
    st.session_state.points = {}  # comp -> int
if "email" not in st.session_state:
    st.session_state.email = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False

# ─────────────── HELPERS ───────────────
def load_defaults(csv_path: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return pd.DataFrame(), f"Could not read {csv_path}. Error: {e}"

    cols = {c.strip().lower(): c for c in df.columns}
    if "component" not in cols or "weight" not in cols:
        return pd.DataFrame(), f"CSV must have columns 'component' and 'weight'. Found: {list(df.columns)}"

    comp_col, w_col = cols["component"], cols["weight"]
    df = df[[comp_col, w_col]].rename(columns={comp_col: "Component", w_col: "weight_raw"})
    if df["Component"].isna().any() or len(df) == 0:
        return pd.DataFrame(), "CSV has empty component names or is empty."

    def parse_w(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str): x = x.replace("%", "").strip()
        try: return float(x)
        except: return 0.0

    w = df["weight_raw"].apply(parse_w).clip(lower=0).to_numpy(dtype=float)
    total = float(w.sum())

    if total <= 0:
        n = len(w)
        ints = np.array([100 // n] * n, dtype=int)
        for i in range(100 - ints.sum()): ints[i] += 1
    else:
        scaled = w / total * 100.0
        base = np.floor(scaled).astype(int)
        delta = 100 - int(base.sum())
        if delta != 0:
            restos = (scaled - base).astype(float)
            order = np.argsort(restos)[::-1]
            for i in range(abs(delta)):
                base[order[i % len(order)]] += 1 if delta > 0 else -1
        ints = np.clip(base, 0, None).astype(int)

    df["Points"] = ints
    return df[["Component", "Points"]], ""

def init_state_from_df(df: pd.DataFrame):
    st.session_state.components = df["Component"].tolist()
    st.session_state.points = {r.Component: int(r.Points) for r in df.itertuples()}

def df_from_state() -> pd.DataFrame:
    return pd.DataFrame({
        "Component": st.session_state.components,
        "Points": [int(st.session_state.points[c]) for c in st.session_state.components],
    })

def _round_to_target(values_float: np.ndarray, target: int) -> np.ndarray:
    base = np.floor(values_float).astype(int)
    delta = target - int(base.sum())
    if delta != 0:
        restos = (values_float - base).astype(float)
        order = np.argsort(restos)[::-1]
        for i in range(abs(delta)):
            base[order[i % len(order)]] += 1 if delta > 0 else -1
    return base

def _rebalance_after(comp: str):
    """Keep total exactly 100 by adjusting all other components proportionally."""
    comps = st.session_state.components
    pts = st.session_state.points

    pts[comp] = max(0, min(100, int(pts[comp])))
    remaining = 100 - pts[comp]
    others = [c for c in comps if c != comp]

    if not others:
        return

    current_sum_others = sum(pts[c] for c in others)

    if remaining <= 0:
        for c in others: pts[c] = 0
        return

    if current_sum_others <= 0:
        n = len(others)
        base = remaining // n
        arr = np.array([base] * n, dtype=int)
        for i in range(remaining - base * n): arr[i] += 1
        for c, v in zip(others, arr.tolist()):
            pts[c] = int(v)
        return

    arr = np.array([pts[c] for c in others], dtype=float)
    scaled = arr / arr.sum() * remaining
    ints = _round_to_target(scaled, remaining)
    ints = np.clip(ints, 0, None)
    for c, v in zip(others, ints.tolist()):
        pts[c] = int(v)

def bump(comp: str, delta: int):
    st.session_state.points[comp] = int(st.session_state.points.get(comp, 0)) + int(delta)
    _rebalance_after(comp)

def set_exact(comp: str, new_val: int):
    st.session_state.points[comp] = int(new_val)
    _rebalance_after(comp)

def ensure_sheet_headers(sh, components: list):
    headers = ["timestamp", "email", "session_id"] + components
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)

def save_to_sheet(email: str, df: pd.DataFrame, session_id: str):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    ensure_sheet_headers(sh, df["Component"].tolist())
    row = [dt.datetime.now().isoformat(), email, session_id] + df["Points"].astype(int).tolist()
    sh.append_row(row)

# ─────────────── INIT ───────────────
if not st.session_state.components:
    df0, err = load_defaults(CSV_PATH)
    if err:
        st.error(err); st.stop()
    init_state_from_df(df0)

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation")
st.caption("Distribute **100 points** across the indicators. The total always stays at 100 automatically. No sliders, no locks.")

email = st.text_input("Your email (required to submit)", value=st.session_state.email, placeholder="name@organization.org")
if email != st.session_state.email:
    st.session_state.email = email.strip()

st.write("---")
st.subheader("Allocation")

# Rows: −10 · −1 · [single number_input] · +1 · +10
for comp in st.session_state.components:
    pts = int(st.session_state.points[comp])

    st.markdown(f"<div class='row'><div class='name'>{comp}</div></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([0.9, 0.9, 1.6, 0.9, 0.9], vertical_alignment="center")

    with c1:
        st.button("−10", key=f"m10_{comp}", on_click=bump, args=(comp, -10))
    with c2:
        st.button("−1",  key=f"m1_{comp}",  on_click=bump, args=(comp,  -1))

    with c3:
        with st.container():
            st.markdown("<div class='center-cell value-input'>", unsafe_allow_html=True)
            new_val = st.number_input(" ", key=f"num_{comp}", value=pts, min_value=0, max_value=100, step=1,
                                      label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            if new_val != pts:
                set_exact(comp, int(new_val))

    with c4:
        st.button("+1",  key=f"p1_{comp}",  on_click=bump, args=(comp,  +1))
    with c5:
        st.button("+10", key=f"p10_{comp}", on_click=bump, args=(comp, +10))

# Totals + chart
df_view = df_from_state()
total_now = int(df_view["Points"].sum())

st.write("---")
colA, colB = st.columns([2,1])
with colA:
    st.markdown(f"**Total allocated:** <span class='badge'>{total_now} / 100</span>", unsafe_allow_html=True)
with colB:
    if st.button("Split equally"):
        n = len(st.session_state.components)
        base = 100 // n
        arr = np.array([base]*n, dtype=int)
        for i in range(100 - base*n): arr[i] += 1
        for c, v in zip(st.session_state.components, arr.tolist()):
            st.session_state.points[c] = int(v)
        df_view = df_from_state()

st.bar_chart(df_view.set_index("Component")["Points"])

st.write("---")
left, right = st.columns([2,1])
with left:
    st.caption("Submit is enabled with a valid email. The total is always 100.")
with right:
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted
        or not EMAIL_RE.match(st.session_state.email or "")
    )
    if st.button("Submit", use_container_width=True, disabled=disabled_submit):
        st.session_state.saving = True
        try:
            save_to_sheet(st.session_state.email, df_from_state(), st.session_state.session_id)
            st.session_state.submitted = True
            st.success("✅ Response saved. Thank you!")
        except Exception as e:
            st.error(f"Error saving to Google Sheets. {e}")
        finally:
            st.session_state.saving = False

if st.session_state.submitted:
    st.caption("You already submitted a response in this session.")
