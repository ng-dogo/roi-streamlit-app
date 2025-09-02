# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt
from typing import Dict
from google.oauth2.service_account import Credentials
import gspread

# ───────────── CONFIG ─────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.9); }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width:860px}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:1rem 0}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid rgba(127,127,127,.25);font-size:.8rem;color:var(--muted)}
.row{padding:.6rem .5rem;border-radius:14px;border:1px solid rgba(127,127,127,.18);}
.name{font-weight:600;margin-bottom:.35rem}
.ctrl{display:flex;gap:.6rem;align-items:center;justify-content:center}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.5rem .9rem}
.stButton>button:disabled{background:#bbb;color:#fff}
.kpi{display:flex;gap:.6rem;align-items:center}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────── CONSTANTS ─────────────
CSV_PATH = "rgi_bap_defaults.csv"     # columns: indicator, avg_weight
TOTAL_POINTS = 100
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
STEP_BIG = 10

# ───────────── STATE ─────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "weights" not in st.session_state:
    st.session_state.weights = {}
if "defaults" not in st.session_state:
    st.session_state.defaults = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ───────────── HELPERS ─────────────
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

def clamp01(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def redistribute_to_keep_total(weights: Dict[str, float], target: str, new_val: float, total: int = TOTAL_POINTS):
    """Proportionally rebalance other indicators to keep the sum exactly 'total'."""
    w = {k: float(v) for k, v in weights.items()}
    old_val = w[target]
    new_val = clamp01(new_val)
    delta = new_val - old_val
    if abs(delta) < 1e-9:
        return w

    keys = [k for k in w.keys() if k != target]
    if not keys:
        w[target] = new_val
        return w

    if delta > 0:
        # take 'delta' from others proportional to their shares, respecting 0 floor
        remainder = delta
        for _ in range(4):
            if remainder <= 1e-9: break
            share_total = sum(max(w[k], 0.0) for k in keys)
            if share_total <= 1e-12: break
            for k in keys:
                if w[k] <= 0: continue
                take = remainder * (w[k] / share_total)
                if w[k] - take < 0:
                    remainder -= w[k]
                    w[k] = 0.0
                else:
                    w[k] -= take
                    remainder -= take
        w[target] = clamp01(old_val + (delta - max(remainder, 0.0)))
    else:
        # give '-delta' to others proportional to their headroom up to 100
        remainder = -delta
        for _ in range(4):
            if remainder <= 1e-9: break
            cap = sum(max(100.0 - w[k], 0.0) for k in keys)
            if cap <= 1e-12: break
            for k in keys:
                avail = max(100.0 - w[k], 0.0)
                if avail <= 0: continue
                give = remainder * (avail / cap)
                if w[k] + give > 100.0:
                    remainder -= (100.0 - w[k])
                    w[k] = 100.0
                else:
                    w[k] += give
                    remainder -= give
        w[target] = clamp01(old_val - ( -delta - max(remainder, 0.0) ))

    # small correction to hit exact total
    s = sum(w.values())
    if abs(s - total) > 1e-9:
        w[target] = clamp01(w[target] + (total - s))
    return w

def save_to_sheet(email: str, weights: dict, session_id: str):
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
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)
    row = [dt.datetime.now().isoformat(), email, session_id] + [round(weights[k],2) for k in weights] + [round(sum(weights.values()),2)]
    sh.append_row(row)

# ───────────── LOAD DEFAULTS ─────────────
try:
    df_defaults = load_defaults_csv(CSV_PATH)
    indicators = df_defaults["indicator"].tolist()
    defaults = {r.indicator: float(r.avg_weight) for r in df_defaults.itertuples()}
    s = sum(defaults.values())
    if s > 0:
        defaults = {k: 100.0 * v / s for k, v in defaults.items()}
    else:
        eq = 100.0 / len(indicators)
        defaults = {k: eq for k in indicators}
    st.session_state.defaults = {k: round(v, 2) for k, v in defaults.items()}
    if not st.session_state.weights:
        st.session_state.weights = dict(st.session_state.defaults)
except Exception as e:
    st.error(f"Could not read '{CSV_PATH}'. {e}")
    st.stop()

# ───────────── UI ─────────────
st.title("RGI – Budget Allocation")

# Top bar: email | total | reset
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    email = st.text_input("Email (identifier)", placeholder="name@example.org")
with c2:
    total = float(np.sum(list(st.session_state.weights.values())))
    st.markdown(f"<div class='kpi'><span class='badge'>Total</span> <strong>{total:.0f} / {TOTAL_POINTS}</strong></div>", unsafe_allow_html=True)
with c3:
    if st.button("Reset to averages"):
        st.session_state.weights = dict(st.session_state.defaults)
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Rows: –10 | value | +10 (symmetric)
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])
    with colL:
        if st.button("−10", key=f"m10_{comp}"):
            st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, st.session_state.weights[comp] - STEP_BIG)
            st.rerun()
    with colC:
        new_val = st.number_input(
            label="", min_value=0, max_value=100, step=1,
            value=int(round(st.session_state.weights.get(comp, 0.0))),
            key=f"num_{comp}", label_visibility="collapsed"
        )
    with colR:
        if st.button("+10", key=f"p10_{comp}"):
            st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, st.session_state.weights[comp] + STEP_BIG)
            st.rerun()

    # manual edit via number_input
    current_val = float(new_val)
    if abs(current_val - st.session_state.weights[comp]) > 1e-9:
        st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, current_val)
        st.rerun()

    st.markdown("&nbsp;", unsafe_allow_html=True)

# Footer / submit
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(email or ""))
disabled_submit = (not ok_email) or st.session_state.submitted

colA, colB = st.columns([1,1])
with colA:
    if st.button("Submit", disabled=disabled_submit):
        try:
            save_to_sheet(email.strip(), st.session_state.weights, st.session_state.session_id)
            st.session_state.submitted = True
            st.success("Saved. Thank you.")
        except Exception as e:
            st.error(f"Error saving your response. {e}")
with colB:
    st.caption("Adjust any value; the app auto-balances the rest to keep the total at 100.")
