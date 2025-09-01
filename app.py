# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt
from google.oauth2.service_account import Credentials
import gspread

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

MINI_CSS = """
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:8px;padding:.5rem 1rem}
.stButton>button:disabled{background:#ccc;color:#666}
.main .block-container{max-width:820px}
hr{border:none;border-top:1px solid #e6e6e6;margin:1rem 0}
.label-small{color:#666;font-size:0.9rem}
</style>
"""
st.markdown(MINI_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
RGI_COMPONENTS = [
    "Legal Framework",
    "Independence & Accountability",
    "Tariff Methodology",
    "Participation & Transparency",
    "Legal Mandate",
    "Clarity of Roles & Objectives",
    "Open Access to Information",
    "Transparency",
]
DEFAULTS_CSV_PATH = "rgi_defaults.csv"
REQUIRE_TOTAL_100 = True

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = 1
if "email" not in st.session_state:
    st.session_state.email = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "weights" not in st.session_state:
    st.session_state.weights = {c: 0.0 for c in RGI_COMPONENTS}

# ─────────────── HELPERS ───────────────
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def ensure_headers(sh):
    headers = ["timestamp","email","session_id"] + RGI_COMPONENTS
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)

@st.cache_data(ttl=300)
def load_defaults_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.astype(str).str.strip()
        key_col = "email" if "email" in df.columns else ("name" if "name" in df.columns else None)
        if key_col is None:
            return pd.DataFrame()
        df[key_col] = df[key_col].astype(str).str.strip()
        df["_key_"] = df[key_col].str.casefold()
        return df
    except Exception:
        return pd.DataFrame()

def map_defaults_row_to_dict(row: pd.Series) -> dict:
    out = {}
    for comp in RGI_COMPONENTS:
        val = None
        c1, c2 = f"w::{comp}", comp
        if c1 in row and pd.notna(row[c1]):
            val = row[c1]
        elif c2 in row and pd.notna(row[c2]):
            val = row[c2]
        if isinstance(val, str):
            val = val.replace("%", "").strip()
        try:
            out[comp] = float(val) if val not in (None, "") else 0.0
        except Exception:
            out[comp] = 0.0
    return out

def load_defaults_for_key(df: pd.DataFrame, key: str) -> dict | None:
    if df.empty:
        return None
    k = (key or "").strip().casefold()
    row = df.loc[df["_key_"] == k]
    if row.empty:
        return None
    return map_defaults_row_to_dict(row.iloc[0])

def autoload_defaults_by_email(email: str):
    df = load_defaults_csv(DEFAULTS_CSV_PATH)
    defaults = load_defaults_for_key(df, email)
    if defaults:
        st.session_state.weights = {k: round(v, 2) for k, v in defaults.items()}
        st.success("Defaults loaded from CSV.")
    else:
        st.info("No defaults found for this email. You can allocate manually.")

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
    ensure_headers(sh)
    row = [dt.datetime.now().isoformat(), email, session_id] + [weights[c] for c in RGI_COMPONENTS]
    sh.append_row(row)

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation")

# ====== STAGE 1: EMAIL ======
if st.session_state.stage == 1:
    st.subheader("Step 1 · Your email")
    email = st.text_input("Email", placeholder="name@example.org", value=st.session_state.email)

    can_continue = bool(EMAIL_RE.match(email))
    if st.button("Continue", disabled=not can_continue):
        st.session_state.email = email.strip()
        st.session_state.stage = 2
        autoload_defaults_by_email(st.session_state.email)
        st.rerun()

    st.caption("We use your email to match initial weights (if available) and to record your submission.")

# ====== STAGE 2: ALLOCATION ======
if st.session_state.stage == 2:
    st.markdown(f"*Email:* {st.session_state.email}")
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Step 2 · Allocate a total of 100 points")
    st.caption("Distribute points across the 8 RGI components. Submit is enabled only if the total equals exactly 100.")

    new_vals = {}
    for comp in RGI_COMPONENTS:
        new_vals[comp] = st.number_input(
            comp, min_value=0.0, max_value=100.0, step=1.0,
            value=float(st.session_state.weights.get(comp, 0.0)),
            key=f"num_{comp}"
        )

    st.session_state.weights = {k: float(v) for k, v in new_vals.items()}
    total = float(np.sum(list(st.session_state.weights.values())))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write(f"*Total allocated:* {total:.0f} / 100")

    disabled_submit = (
        st.session_state.saving or st.session_state.submitted or
        not st.session_state.email or
        (REQUIRE_TOTAL_100 and abs(total - 100.0) > 1e-9)
    )
    if st.button("Submit", disabled=disabled_submit):
        st.session_state.saving = True
        try:
            save_to_sheet(st.session_state.email, st.session_state.weights, st.session_state.session_id)
            st.session_state.submitted = True
            st.success("Saved. Thank you.")
        except Exception as e:
            st.error(f"Error saving your response. {e}")
        finally:
            st.session_state.saving = False

    if st.session_state.submitted:
        st.caption("Response already submitted for this session.")