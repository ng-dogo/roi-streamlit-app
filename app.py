# app.py
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt
import re
from google.oauth2.service_account import Credentials
import gspread

# ─────────────── CONFIG ───────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

# Dark mode compatible styling
CUSTOM_CSS = """
<style>
:root {
    --primary: #4CAF50;
}
html, body, [class*="css"] {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.stButton>button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-size: 0.9rem;
}
.stButton>button:disabled {
    background: #555;
    color: #999;
}
.value-box {
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    background: rgba(255,255,255,0.1);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
CSV_PATH = "defaults.csv"  # prototype csv (Indicator,DefaultWeight)
REQUIRE_TOTAL_100 = True

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "email" not in st.session_state:
    st.session_state.email = ""
if "weights" not in st.session_state:
    st.session_state.weights = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ─────────────── HELPERS ───────────────
def load_defaults(path: str) -> dict:
    try:
        df = pd.read_csv(path)
        return dict(zip(df["Indicator"], df["DefaultWeight"]))
    except Exception:
        return {}

def save_to_sheet(email: str, weights: dict, session_id: str):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    headers = ["timestamp","email","session_id"] + list(weights.keys())
    if not sh.get_all_values():
        sh.append_row(headers)
    row = [dt.datetime.now().isoformat(), email, session_id] + [weights[k] for k in weights]
    sh.append_row(row)

# ─────────────── MAIN APP ───────────────
st.title("RGI – Budget Allocation")

# Email input
email = st.text_input("Your Email", placeholder="name@example.org", value=st.session_state.email)
if EMAIL_RE.match(email or ""):
    st.session_state.email = email.strip()

# Load defaults if not loaded yet
if not st.session_state.weights:
    st.session_state.weights = load_defaults(CSV_PATH)

st.markdown("---")
st.subheader("Distribute 100 points across the indicators")

# Display each indicator with -10/-1 value +1/+10
for comp, val in st.session_state.weights.items():
    cols = st.columns([1,1,1,1,1])
    with cols[0]:
        if st.button("-10", key=f"{comp}_m10"):
            st.session_state.weights[comp] = max(0, st.session_state.weights[comp] - 10)
    with cols[1]:
        if st.button("-1", key=f"{comp}_m1"):
            st.session_state.weights[comp] = max(0, st.session_state.weights[comp] - 1)
    with cols[2]:
        st.markdown(f"<div class='value-box'>{st.session_state.weights[comp]}</div>", unsafe_allow_html=True)
    with cols[3]:
        if st.button("+1", key=f"{comp}_p1"):
            st.session_state.weights[comp] = min(100, st.session_state.weights[comp] + 1)
    with cols[4]:
        if st.button("+10", key=f"{comp}_p10"):
            st.session_state.weights[comp] = min(100, st.session_state.weights[comp] + 10)

# Total
total = sum(st.session_state.weights.values())
st.markdown("---")
st.write(f"**Total allocated:** {total} / 100")

# Submit
disabled = (
    not EMAIL_RE.match(st.session_state.email or "") or
    (REQUIRE_TOTAL_100 and abs(total - 100) > 1e-9) or
    st.session_state.submitted
)
if st.button("Submit", disabled=disabled):
    try:
        save_to_sheet(st.session_state.email, st.session_state.weights, st.session_state.session_id)
        st.session_state.submitted = True
        st.success("Saved successfully. Thank you!")
    except Exception as e:
        st.error(f"Error saving response: {e}")

if st.session_state.submitted:
    st.caption("Response already submitted for this session.")
