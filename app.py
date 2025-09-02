# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os
from typing import Dict
from google.oauth2.service_account import Credentials
import gspread

# ───────── CONFIG ─────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width:860px}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:1rem 0}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid rgba(127,127,127,.25);font-size:.8rem;color:var(--muted)}
.name{font-weight:600;margin:.35rem 0 .25rem}
.rowbox{padding:.45rem .5rem;border-radius:12px;border:1px solid rgba(127,127,127,.18);}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.45rem .9rem}
.stButton>button:disabled{background:#bbb;color:#fff}
.center input[type=number]{text-align:center;font-weight:600}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 100
STEP_BIG = 10
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ───────── STATE ─────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "weights" not in st.session_state:
    st.session_state.weights: Dict[str, float] = {}
if "defaults" not in st.session_state:
    st.session_state.defaults: Dict[str, float] = {}
if "email" not in st.session_state:
    st.session_state.email = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ───────── HELPERS ─────────
@st.cache_data(ttl=300)
def load_defaults_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    name_col = "indicator" if "indicator" in cols else cols[0]
    weight_col = "avg_weight" if "avg_weight" in cols else cols[1]
    out = df[[name_col, weight_col]].copy()
    out.columns = ["indicator", "avg_weight"]
    out["indicator"] = out["indicator"].astype(str).str.strip()
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").fillna(0.0)
    return out

def normalize_to_100(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        n = len(weights); eq = 100.0 / max(1, n)
        return {k: eq for k in weights}
    return {k: 100.0 * v / s for k, v in weights.items()}

def remaining_points(weights: Dict[str, float]) -> float:
    return TOTAL_POINTS - float(sum(weights.values()))

def save_to_sheet(email: str, weights: Dict[str, float], session_id: str):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    headers = ["timestamp","email","session_id"] + list(weights.keys()) + ["total"]
    if not sh.get_all_values():
        sh.append_row(headers)
    row = [dt.datetime.now().isoformat(), email, session_id] + [round(weights[k],2) for k in weights] + [round(sum(weights.values()),2)]
    sh.append_row(row)

# ───────── LOAD DEFAULTS (once) ─────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    defaults = normalize_to_100(defaults_raw)
    st.session_state.defaults = {k: round(v, 2) for k, v in defaults.items()}
    st.session_state.weights = dict(st.session_state.defaults)
else:
    indicators = list(st.session_state.weights.keys())

# ───────── UI ─────────
st.title("RGI – Budget Allocation")

# Top bar: email | remaining | reset
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.session_state.email = st.text_input("Email (identifier)", value=st.session_state.email, placeholder="name@example.org")
with c2:
    rem = remaining_points(st.session_state.weights)
    st.markdown(f"<span class='badge'>Remaining</span> <strong>{rem:.0f}</strong>", unsafe_allow_html=True)
with c3:
    if st.button("Reset to averages"):
        st.session_state.weights = dict(st.session_state.defaults)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Rows: –10 | number_input (±1 nativo) | +10
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])

    # –10: siempre permitido (libera puntos)
    with colL:
        if st.button("−10", key=f"m10_{comp}"):
            st.session_state.weights[comp] = max(0.0, st.session_state.weights[comp] - STEP_BIG)

    # número central: cap al máximo permitido (valor_actual + remaining)
    with colC:
        with st.container():
            st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
            current = float(st.session_state.weights[comp])
            # Permitimos teclear libremente, luego clamp
            new_val = st.number_input(
                label="", key=f"num_{comp}",
                value=int(round(current)), min_value=0, max_value=100, step=1,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Si sube, cap al máximo posible según remaining
            if float(new_val) != current:
                rem_now = remaining_points(st.session_state.weights)
                if new_val > current:
                    allowed = min(100.0, current + rem_now)
                    st.session_state.weights[comp] = float(min(new_val, allowed))
                else:
                    # Si baja, liberamos puntos (siempre ok)
                    st.session_state.weights[comp] = float(new_val)

    # +10: solo permitir si hay al menos 10 libres
    with colR:
        can_add = remaining_points(st.session_state.weights) >= STEP_BIG
        if st.button("+10", key=f"p10_{comp}", disabled=not can_add):
            st.session_state.weights[comp] = min(100.0, st.session_state.weights[comp] + STEP_BIG)

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(st.session_state.email or ""))
disabled_submit = (not ok_email) or st.session_state.submitted or (abs(remaining_points(st.session_state.weights)) > 1e-9)

left, right = st.columns([1,1])
with left:
    if st.button("Submit", disabled=disabled_submit):
        try:
            save_to_sheet(st.session_state.email.strip(), st.session_state.weights, st.session_state.session_id)
            st.session_state.submitted = True
            st.success("Saved. Thank you.")
        except Exception as e:
            st.error(f"Error saving your response. {e}")
with right:
    st.caption("Start from averages. Adjust values; increases are limited by Remaining. No hidden rebalancing.")
