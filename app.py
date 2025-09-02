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
    st.session_state.weights: Dict[str, int] = {}
if "defaults" not in st.session_state:
    st.session_state.defaults: Dict[str, int] = {}
if "email" not in st.session_state:
    st.session_state.email = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "_init_inputs" not in st.session_state:
    st.session_state._init_inputs = False  # se volverá True al cargar defaults

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

def normalize_to_100_ints(weights: Dict[str, float]) -> Dict[str, int]:
    """
    Escala a 100 y redondea a enteros con 'mayores restos' (Hare-Niemeyer) para que sumen EXACTAMENTE 100.
    """
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        base = 100 // n
        res = 100 - base * n
        out = {k: base for k in weights}
        for k in list(out.keys())[:res]:
            out[k] += 1
        return out

    raw = {k: 100.0 * v / total for k, v in weights.items()}
    floors = {k: int(np.floor(x)) for k, x in raw.items()}
    leftover = 100 - sum(floors.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    out = floors.copy()
    for k in order[:leftover]:
        out[k] += 1
    return out

def remaining_points(weights: Dict[str, int]) -> int:
    return int(TOTAL_POINTS - int(sum(weights.values())))

# Ajustes de UI/estado
def adjust(comp: str, delta: int):
    """Sube/baja respetando 0–100 y el pool disponible (Remaining). También sincroniza el number_input."""
    cur = int(st.session_state.weights[comp])
    if delta > 0:
        allowed = min(delta, remaining_points(st.session_state.weights))
        new_val = min(100, cur + allowed)
    else:
        new_val = max(0, cur + delta)
    st.session_state.weights[comp] = int(new_val)
    st.session_state[f"num_{comp}"] = int(new_val)  # mantener sincronizado el input

def make_on_change(comp: str):
    def _cb():
        cur = int(st.session_state.weights[comp])
        new_val = int(st.session_state[f"num_{comp}"])
        delta = new_val - cur
        if delta > 0:
            allowed = min(delta, remaining_points(st.session_state.weights))
            st.session_state.weights[comp] = cur + allowed
        else:
            st.session_state.weights[comp] = max(0, new_val)
        # reescribe el input con el valor realmente aplicado (fuerza rerun coherente)
        st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])
    return _cb

def save_to_sheet(email: str, weights: Dict[str, int], session_id: str, indicator_order: list[str]):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    headers = ["timestamp","email","session_id"] + indicator_order + ["total"]
    if not sh.get_all_values():
        sh.append_row(headers)
    row = [dt.datetime.now().isoformat(), email, session_id] + [int(weights[k]) for k in indicator_order] + [int(sum(weights.values()))]
    sh.append_row(row)

# ───────── LOAD DEFAULTS (una sola vez) ─────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    defaults_int = normalize_to_100_ints(defaults_raw)  # enteros que suman EXACTAMENTE 100
    st.session_state.defaults = defaults_int
    st.session_state.weights = dict(defaults_int)
    st.session_state._init_inputs = True
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
    st.markdown(f"<span class='badge'>Remaining</span> <strong>{rem}</strong>", unsafe_allow_html=True)
with c3:
    if st.button("Reset to averages"):
        st.session_state.weights = dict(st.session_state.defaults)
        # sincronizar inputs
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Inicializa los inputs la primera vez que se carga
if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])
    st.session_state._init_inputs = False

# Rows: –10 | number_input (±1 nativo) | +10
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])

    with colL:
        cur = int(st.session_state.weights[comp])
        st.button("−10", key=f"m10_{comp}", on_click=lambda c=comp: adjust(c, -STEP_BIG), disabled=(cur <= 0))

    with colC:
        st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
        cur = int(st.session_state.weights[comp])
        rem_now = remaining_points(st.session_state.weights)
        max_allowed = cur + rem_now  # no se puede superar el pool disponible
        st.number_input(
            label="",
            key=f"num_{comp}",
            value=int(cur),
            min_value=0,
            max_value=int(max_allowed),
            step=1,
            format="%d",
            label_visibility="collapsed",
            on_change=make_on_change(comp),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        can_add = (remaining_points(st.session_state.weights) > 0) and (int(st.session_state.weights[comp]) < 100)
        st.button("+10", key=f"p10_{comp}", on_click=lambda c=comp: adjust(c, STEP_BIG), disabled=not can_add)

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(st.session_state.email or ""))
disabled_submit = (not ok_email) or st.session_state.submitted or (remaining_points(st.session_state.weights) != 0)

left, right = st.columns([1,1])
with left:
    if st.button("Submit", disabled=disabled_submit):
        try:
            save_to_sheet(
                st.session_state.email.strip(),
                st.session_state.weights,
                st.session_state.session_id,
                indicator_order=indicators
            )
            st.session_state.submitted = True
            st.success("Saved. Thank you.")
        except Exception as e:
            st.error(f"Error saving your response. {e}")
with right:
    st.caption("Start from averages. Adjust values; increases are limited by Remaining. No hidden rebalancing.")
