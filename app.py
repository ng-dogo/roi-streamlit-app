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
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.9); }
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

def clamp01(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))

def rebalance_keep_total(weights: Dict[str, float], target: str, new_val: float, total: int = TOTAL_POINTS) -> Dict[str, float]:
    """Rebalance others proportionally to keep exact total; bounds [0,100]. No st.rerun here."""
    w = {k: float(v) for k, v in weights.items()}
    old = w[target]
    new = clamp01(new_val)
    delta = new - old
    if abs(delta) < 1e-9:
        return w

    others = [k for k in w.keys() if k != target]
    if not others:
        w[target] = new
        return w

    if delta > 0:
        # take from others proportional to their current positive share
        need = delta
        for _ in range(3):
            if need <= 1e-9: break
            share = sum(max(w[k], 0.0) for k in others)
            if share <= 1e-12: break
            for k in others:
                if w[k] <= 0: continue
                take = need * (w[k] / share)
                if w[k] - take < 0:
                    need -= w[k]; w[k] = 0.0
                else:
                    w[k] -= take; need -= take
        w[target] = clamp01(old + (delta - max(need, 0.0)))
    else:
        # give to others proportional to headroom
        give = -delta
        for _ in range(3):
            if give <= 1e-9: break
            cap = sum(max(100.0 - w[k], 0.0) for k in others)
            if cap <= 1e-12: break
            for k in others:
                room = max(100.0 - w[k], 0.0)
                if room <= 0: continue
                g = give * (room / cap)
                if w[k] + g > 100.0:
                    give -= (100.0 - w[k]); w[k] = 100.0
                else:
                    w[k] += g; give -= g
        w[target] = clamp01(old - ( -delta - max(give, 0.0) ))

    # small correction on target to hit exact total
    s = sum(w.values())
    if abs(s - total) > 1e-9:
        w[target] = clamp01(w[target] + (total - s))
    return w

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
    defaults = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    s = sum(defaults.values())
    if s > 0:
        defaults = {k: 100.0 * v / s for k, v in defaults.items()}
    else:
        eq = 100.0 / max(1, len(indicators))
        defaults = {k: eq for k in indicators}
    st.session_state.defaults = {k: round(v, 2) for k, v in defaults.items()}
    st.session_state.weights = dict(st.session_state.defaults)
else:
    indicators = list(st.session_state.weights.keys())

# ───────── UI ─────────
st.title("RGI – Budget Allocation")

# Top bar: email | total | reset
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.session_state.email = st.text_input("Email (identifier)", value=st.session_state.email, placeholder="name@example.org")
with c2:
    total = float(sum(st.session_state.weights.values()))
    st.markdown(f"<span class='badge'>Total</span> <strong>{total:.0f} / {TOTAL_POINTS}</strong>", unsafe_allow_html=True)
with c3:
    if st.button("Reset to averages"):
        st.session_state.weights = dict(st.session_state.defaults)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Rows: –10 | number_input | +10  (sin st.rerun)
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])

    with colL:
        if st.button("−10", key=f"m10_{comp}"):
            st.session_state.weights = rebalance_keep_total(
                st.session_state.weights, comp, st.session_state.weights[comp] - STEP_BIG
            )

    with colC:
        with st.container():
            st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
            new_val = st.number_input(
                label="", key=f"num_{comp}",
                value=int(round(st.session_state.weights[comp])),
                min_value=0, max_value=100, step=1, label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            # apply manual edit (no rerun explicit)
            if float(new_val) != st.session_state.weights[comp]:
                st.session_state.weights = rebalance_keep_total(st.session_state.weights, comp, float(new_val))

    with colR:
        if st.button("+10", key=f"p10_{comp}"):
            st.session_state.weights = rebalance_keep_total(
                st.session_state.weights, comp, st.session_state.weights[comp] + STEP_BIG
            )

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(st.session_state.email or ""))
disabled_submit = (not ok_email) or st.session_state.submitted

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
    st.caption("Adjust values using −10 / +10 or the central field. The total always stays at 100.")
