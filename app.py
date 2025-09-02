# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt
from typing import Dict
from google.oauth2.service_account import Credentials
import gspread

# -------------- PAGE CONFIG & THEME-SAFE CSS --------------
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{
  --brand:#0E7C66;
  --bg: transparent;
  --muted: rgba(128,128,128,.9);
}
html, body, [class*="css"]{font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width: 860px;}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:1rem 0}
h1,h2,h3,strong{letter-spacing:.2px}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;
  border:1px solid rgba(127,127,127,.25);font-size:.8rem;color:var(--muted)}
.row{padding:.6rem .5rem;border-radius:14px;background:var(--bg);border:1px solid rgba(127,127,127,.18);}
.ctrl{display:flex;gap:.6rem;align-items:center;justify-content:center}
.ctrl .btn{min-width:66px; text-align:center; border-radius:12px; padding:.45rem .6rem;
  border:1px solid rgba(127,127,127,.28); background:var(--bg); cursor:pointer; user-select:none}
.ctrl .btn:hover{border-color:rgba(127,127,127,.6)}
input[type=number]{text-align:center;}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.5rem .9rem}
.stButton>button:disabled{background:#bbb;color:#fff}
.header-actions{display:flex;gap:.5rem;align-items:center;justify-content:flex-end}
.small{color:var(--muted);font-size:.9rem}
.kpi{display:flex;gap:.8rem;align-items:center}
.kpi strong{font-size:1rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------- CONSTANTS --------------
CSV_PATH = "rgi_bap_defaults.csv"   # two columns: indicator, avg_weight
REQUIRE_TOTAL_100 = True
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
POINT_STEP_BIG = 10
TOTAL_POINTS = 100

# -------------- STATE --------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "weights" not in st.session_state:
    st.session_state.weights = {}
if "defaults" not in st.session_state:
    st.session_state.defaults = {}
if "last_changed" not in st.session_state:
    st.session_state.last_changed = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# -------------- HELPERS --------------
@st.cache_data(ttl=300)
def load_defaults_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # standardize columns
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    # accept either indicator / avg_weight or similar variants
    name_col = "indicator" if "indicator" in cols else cols[0]
    weight_col = "avg_weight" if "avg_weight" in cols else cols[1]
    out = df[[name_col, weight_col]].copy()
    out.columns = ["indicator", "avg_weight"]
    out["indicator"] = out["indicator"].astype(str).str.strip()
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").fillna(0.0)
    return out

def clamp01(x, lo=0.0, hi=100.0): 
    return float(max(lo, min(hi, x)))

def redistribute_to_keep_total(weights: Dict[str, float], target: str, new_val: float, total=TOTALE_POINTS:=TOTAL_POINTS):
    """Redistribute deltas across other indicators proportionally while honoring [0,100] bounds."""
    w = {k: float(v) for k, v in weights.items()}
    old_val = w[target]
    new_val = clamp01(new_val)
    delta = new_val - old_val
    if abs(delta) < 1e-9:
        return w  # nothing to do

    keys = [k for k in w.keys() if k != target]
    if not keys:
        w[target] = clamp01(new_val); 
        return w

    if delta > 0:
        # need to subtract delta from others proportionally to their current share
        others_sum = sum(w[k] for k in keys)
        if others_sum <= 1e-12:
            # nowhere to take from — limit the increase
            w[target] = old_val  # reject change
            return w
        remainder = delta
        # iterative proportional take while clamping at 0
        for _ in range(3):  # 8 items max; a couple passes suffice
            if remainder <= 1e-9: break
            share_total = sum(w[k] for k in keys if w[k] > 0)
            if share_total <= 1e-12: break
            for k in keys:
                if w[k] <= 0: continue
                take = remainder * (w[k] / share_total)
                newk = w[k] - take
                if newk < 0:
                    remainder -= w[k]  # we can only take what's there
                    w[k] = 0.0
                else:
                    w[k] = newk
                    remainder -= take
        w[target] = clamp01(old_val + (delta - max(remainder,0)))
    else:
        # delta < 0: we freed points; add to others by available headroom
        give_total_cap = sum(100.0 - w[k] for k in keys)
        remainder = -delta
        if give_total_cap <= 1e-12:
            w[target] = old_val  # nowhere to give; reject
            return w
        for _ in range(3):
            if remainder <= 1e-9: break
            cap = sum(100.0 - w[k] for k in keys if w[k] < 100.0)
            if cap <= 1e-12: break
            for k in keys:
                avail = 100.0 - w[k]
                if avail <= 0: continue
                give = remainder * (avail / cap)
                newk = w[k] + give
                if newk > 100.0:
                    remainder -= (100.0 - w[k])
                    w[k] = 100.0
                else:
                    w[k] = newk
                    remainder -= give
        w[target] = clamp01(old_val - ( -delta - max(remainder,0) ))

    # final small rounding correction to hit exact TOTAL
    s = sum(w.values())
    if s != total:
        corr = total - s
        w[target] = clamp01(w[target] + corr)
    return w

def save_to_sheet(email: str, weights: dict, session_id: str):
    """Optional Google Sheet write (requires Streamlit secrets)."""
    if not hasattr(st.secrets, "sheet_id"):
        raise RuntimeError("Google Sheet is not configured in Streamlit secrets.")
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    # Ensure headers
    headers = ["timestamp","email","session_id"] + list(weights.keys()) + ["total"]
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)
    row = [dt.datetime.now().isoformat(), email, session_id] + [round(weights[c],2) for c in weights.keys()] + [round(sum(weights.values()),2)]
    sh.append_row(row)

# -------------- LOAD DEFAULTS --------------
try:
    df_defaults = load_defaults_csv(CSV_PATH)
    indicators = df_defaults["indicator"].tolist()
    defaults = {r.indicator: float(r.avg_weight) for r in df_defaults.itertuples()}
    # normalize defaults to total 100 just in case
    s = sum(defaults.values())
    if s > 0:
        defaults = {k: 100.0 * v / s for k, v in defaults.items()}
    else:
        eq = 100.0 / len(indicators)
        defaults = {k: eq for k in indicators}
    st.session_state.defaults = {k: round(v,2) for k,v in defaults.items()}
    if not st.session_state.weights:
        st.session_state.weights = dict(st.session_state.defaults)
except Exception as e:
    st.error(f"Could not read '{CSV_PATH}'. {e}")
    st.stop()

# -------------- HEADER --------------
st.title("RGI – Budget Allocation")

# top row: email (left), totals (center), reset (right)
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    email = st.text_input("Email (identifier)", placeholder="name@example.org")
with c2:
    total = float(np.sum(list(st.session_state.weights.values())))
    remaining = TOTAL_POINTS - total
    st.markdown("<div class='kpi'><span class='badge'>Total</span> "
                f"<strong>{total:.0f} / {TOTAL_POINTS}</strong></div>", unsafe_allow_html=True)
    st.caption(f"Remaining: {remaining:.0f} points")
with c3:
    if st.button("Reset to averages"):
        st.session_state.weights = dict(st.session_state.defaults)
        st.session_state.last_changed = None
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# -------------- ROWS (–10 | VALUE | +10) --------------
for comp in indicators:
    st.text_input("", value=comp, key=f"label_{comp}", disabled=True, label_visibility="collapsed")
    # symmetric columns: [-10 | value | +10]
    colL, colC, colR = st.columns([1, 3, 1])
    with colL:
        if st.button("−10", key=f"m10_{comp}"):
            st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, st.session_state.weights[comp] - POINT_STEP_BIG)
            st.session_state.last_changed = comp
            st.rerun()
    with colC:
        with st.container():
            st.markdown("<div class='row'>", unsafe_allow_html=True)
            # number input in the center (step=1 lets them adjust by ±1 with native arrows)
            new_val = st.number_input(
                label="", min_value=0, max_value=100, step=1,
                value=int(round(st.session_state.weights.get(comp, 0.0))),
                key=f"num_{comp}", label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        if st.button("+10", key=f"p10_{comp}"):
            st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, st.session_state.weights[comp] + POINT_STEP_BIG)
            st.session_state.last_changed = comp
            st.rerun()

    # detect manual edit (number_input)
    current_val = float(new_val)
    if abs(current_val - st.session_state.weights[comp]) > 1e-9:
        st.session_state.weights = redistribute_to_keep_total(st.session_state.weights, comp, current_val)
        st.session_state.last_changed = comp
        st.rerun()

    st.markdown("&nbsp;", unsafe_allow_html=True)

# -------------- FOOTER / SUBMIT --------------
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(email or ""))
disabled_submit = (
    not ok_email or
    (REQUIRE_TOTAL_100 and abs(sum(st.session_state.weights.values()) - TOTAL_POINTS) > 1e-6) or
    st.session_state.submitted
)

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
    st.caption("You can adjust any value; the app auto-balances the rest to keep the total at 100.")

if st.session_state.submitted:
    st.caption("Response already submitted for this session.")
