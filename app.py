# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os
from typing import Dict, List, Tuple
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
.donor-tray{padding:.5rem;border-radius:10px;border:1px solid rgba(127,127,127,.25); background:transparent}
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

# donor picker transient state
if "pending_target" not in st.session_state:
    st.session_state.pending_target: str | None = None
if "pending_needed" not in st.session_state:
    st.session_state.pending_needed: float = 0.0
if "pending_selection" not in st.session_state:
    st.session_state.pending_selection: List[str] = []

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

def apply_delta(weights: Dict[str, float], comp: str, delta: float) -> Dict[str, float]:
    """Apply +/- to a component if allowed by remaining. If not enough remaining, open donor tray."""
    w = dict(weights)
    rem = remaining_points(w)
    if delta <= rem:  # we can increase directly (or always decrease)
        w[comp] = max(0.0, min(100.0, w[comp] + delta))
        return w
    else:
        # Not enough remaining: set pending donor picker for exact shortfall
        need = delta - rem
        st.session_state.pending_target = comp
        st.session_state.pending_needed = float(np.ceil(need))  # work in integer points for UI clarity
        # preselect top donors (largest weights except target)
        donors_sorted = sorted((k for k in w if k != comp), key=lambda k: w[k], reverse=True)
        st.session_state.pending_selection = donors_sorted[:3]
        return w

def apply_manual_set(weights: Dict[str, float], comp: str, new_val: float) -> Dict[str, float]:
    """Set value via number_input; if needs points and remaining=0, open donor tray for the shortfall."""
    w = dict(weights)
    new_val = float(max(0.0, min(100.0, new_val)))
    old = w[comp]
    delta = new_val - old
    if delta <= 0:
        # freeing points is always allowed
        w[comp] = new_val
        return w
    # need more points
    rem = remaining_points(w)
    if delta <= rem:
        w[comp] = new_val
        return w
    else:
        need = delta - rem
        st.session_state.pending_target = comp
        st.session_state.pending_needed = float(np.ceil(need))
        donors_sorted = sorted((k for k in w if k != comp), key=lambda k: w[k], reverse=True)
        st.session_state.pending_selection = donors_sorted[:3]
        return w

def try_apply_donor_transfer(weights: Dict[str, float], target: str, needed: float, donors: List[str]) -> Tuple[Dict[str, float], bool, str]:
    """Take 'needed' points from selected donors proportionally to their current weights."""
    if not donors:
        return weights, False, "Select at least one donor."
    w = dict(weights)
    # compute current remaining (if any) to reduce the needed amount)
    rem = remaining_points(w)
    shortfall = max(0.0, needed - rem)  # only take what's truly missing
    if shortfall <= 0.0:
        # no need to take from donors; we can just increase target by 'needed'
        w[target] = min(100.0, w[target] + needed)
        return w, True, ""

    donors = [d for d in donors if d in w and d != target]
    if not donors:
        return weights, False, "Invalid donors."

    avail_total = sum(max(w[d], 0.0) for d in donors)
    if avail_total < shortfall - 1e-9:
        return weights, False, "Selected donors don't have enough points."

    # take proportionally
    to_take = shortfall
    # first pass proportional
    for d in donors:
        share = w[d] / avail_total if avail_total > 0 else 0
        take_d = to_take * share
        w[d] = max(0.0, w[d] - take_d)

    # correction: due to floors/rounds, recompute remaining
    rem2 = remaining_points(w)
    missing = max(0.0, shortfall - rem2)
    if missing > 1e-6:
        # small second pass: take evenly from donors with leftover >0
        donors_left = [d for d in donors if w[d] > 0]
        if donors_left:
            step = missing / len(donors_left)
            for d in donors_left:
                w[d] = max(0.0, w[d] - step)

    # increase target by needed
    w[target] = min(100.0, w[target] + needed)
    return w, True, ""

def reset_pending():
    st.session_state.pending_target = None
    st.session_state.pending_needed = 0.0
    st.session_state.pending_selection = []

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
    defaults = normalize_to_100(defaults)
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
        reset_pending()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Rows: –10 | number_input | +10  (no auto-rebalance; donor tray when needed)
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])

    with colL:
        # −10: always allowed (it frees points)
        if st.button("−10", key=f"m10_{comp}"):
            st.session_state.weights[comp] = max(0.0, st.session_state.weights[comp] - STEP_BIG)
            reset_pending()

    with colC:
        with st.container():
            st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
            new_val = st.number_input(
                label="", key=f"num_{comp}",
                value=int(round(st.session_state.weights[comp])),
                min_value=0, max_value=100, step=1, label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if float(new_val) != st.session_state.weights[comp]:
                st.session_state.weights = apply_manual_set(st.session_state.weights, comp, float(new_val))

    with colR:
        # +10: allowed only if remaining >= 10, otherwise open donor tray
        can_add = remaining_points(st.session_state.weights) >= STEP_BIG
        if st.button("+10", key=f"p10_{comp}", disabled=False):
            st.session_state.weights = apply_delta(st.session_state.weights, comp, STEP_BIG)

    # Donor tray (inline) if this is the pending target
    if st.session_state.pending_target == comp and st.session_state.pending_needed > 0:
        st.markdown(f"<div class='donor-tray'>", unsafe_allow_html=True)
        st.write(f"**Take {int(st.session_state.pending_needed)} points from:**")
        # donor candidates sorted by size desc
        candidates = [k for k in indicators if k != comp]
        donors_sorted = sorted(candidates, key=lambda k: st.session_state.weights[k], reverse=True)
        default_sel = [d for d in st.session_state.pending_selection if d in donors_sorted]
        picked = st.multiselect("Donors", donors_sorted, default=default_sel, key=f"donors_{comp}")
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Apply transfer", key=f"apply_{comp}"):
                new_w, ok, msg = try_apply_donor_transfer(
                    st.session_state.weights,
                    comp,
                    float(st.session_state.pending_needed),
                    picked
                )
                if ok:
                    st.session_state.weights = new_w
                    reset_pending()
                else:
                    st.warning(msg)
        with colB:
            if st.button("Cancel", key=f"cancel_{comp}"):
                reset_pending()
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.caption("Start from averages. Increase using +10 or typing; if no points remain, pick donors explicitly.")
