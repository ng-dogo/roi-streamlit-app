# app.py
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt
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

# ─────────────── PARAMS (editables) ───────────────
# Usá los nombres EXACTOS que quieras mostrar/registar
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

DEFAULTS_CSV_PATH = "rgi_defaults.csv"  # subilo a tu repo
REQUIRE_TOTAL_100 = True                # sólo habilita submit si suman 100

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "user" not in st.session_state:
    st.session_state.user = {"first":"","last":"","entity":"","country":""}
if "weights" not in st.session_state:
    st.session_state.weights = {c: 0.0 for c in RGI_COMPONENTS}  # arranca en 0

# ─────────────── HELPERS ───────────────
@st.cache_data
def load_defaults_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Normalizo nombre para el match
        if "name" not in df.columns:
            return pd.DataFrame()
        df["__name_norm__"] = df["name"].astype(str).str.strip().str.casefold()
        return df
    except Exception:
        return pd.DataFrame()

def map_defaults_row_to_dict(row: pd.Series) -> dict:
    """
    Acepta dos formatos de columnas en el CSV:
      1) 'w::<Componente>'
      2) '<Componente>' (exacto)
    Devuelve dict {componente: valor_float}
    """
    out = {}
    for comp in RGI_COMPONENTS:
        val = None
        col1 = f"w::{comp}"
        col2 = comp
        if col1 in row and pd.notna(row[col1]):
            val = row[col1]
        elif col2 in row and pd.notna(row[col2]):
            val = row[col2]
        if val is not None:
            try:
                out[comp] = float(val)
            except Exception:
                out[comp] = 0.0
        else:
            out[comp] = 0.0
    return out

def load_defaults_for_name(df: pd.DataFrame, first: str, last: str) -> dict | None:
    if df.empty:
        return None
    target = f"{first} {last}".strip().casefold()
    if not target.strip():
        return None
    row = df.loc[df["__name_norm__"] == target]
    if row.empty:
        return None
    return map_defaults_row_to_dict(row.iloc[0])

def save_to_sheet(user, weights: dict, session_id: str):
    # Requiere en st.secrets: gs_email, gs_key, sheet_id
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1

    row = [
        dt.datetime.now().isoformat(),
        user["first"], user["last"], user["entity"], user["country"],
        session_id
    ] + [weights[c] for c in RGI_COMPONENTS]

    sh.append_row(row)

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation")

# Datos del usuario
st.subheader("Your details")
c1, c2 = st.columns(2)
with c1:
    first = st.text_input("First name", value=st.session_state.user["first"])
    entity = st.text_input("Organization / Regulatory entity", value=st.session_state.user["entity"])
with c2:
    last = st.text_input("Last name", value=st.session_state.user["last"])
    country = st.text_input("Country", value=st.session_state.user["country"])

st.session_state.user = {
    "first": first.strip(),
    "last": last.strip(),
    "entity": entity.strip(),
    "country": country.strip()
}

# Botón para precargar defaults desde CSV por nombre completo
defaults_df = load_defaults_csv(DEFAULTS_CSV_PATH)
prefill = st.button("Load my Mentimeter-based defaults")

if prefill:
    defaults = load_defaults_for_name(defaults_df, st.session_state.user["first"], st.session_state.user["last"])
    if defaults:
        st.session_state.weights = {k: round(v, 2) for k, v in defaults.items()}
        st.success("Defaults loaded.")
    else:
        st.warning("No defaults found for that name. You can fill manually.")

st.markdown("<hr/>", unsafe_allow_html=True)

# Inputs minimalistas (sin gráficos)
st.subheader("Allocate a total of 100 points")
st.caption("Distribute points across the 8 RGI components. No auto-normalization. The Submit button is enabled only if the total equals exactly 100.")

new_vals = {}
for comp in RGI_COMPONENTS:
    new_vals[comp] = st.number_input(
        comp, min_value=0.0, max_value=100.0, step=1.0,
        value=float(st.session_state.weights.get(comp, 0.0)),
        key=f"num_{comp}"
    )

# Guardamos en estado y mostramos total
st.session_state.weights = {k: float(v) for k, v in new_vals.items()}
total = float(np.sum(list(st.session_state.weights.values())))

st.markdown("<hr/>", unsafe_allow_html=True)
st.write(f"**Total allocated:** {total:.0f} / 100")

# Submit
disabled_submit = (
    st.session_state.saving or st.session_state.submitted or
    not all(st.session_state.user.values()) or
    (REQUIRE_TOTAL_100 and abs(total - 100.0) > 1e-9)
)
if st.button("Submit", disabled=disabled_submit):
    st.session_state.saving = True
    try:
        save_to_sheet(st.session_state.user, st.session_state.weights, st.session_state.session_id)
        st.session_state.submitted = True
        st.success("Saved. Thank you.")
    except Exception as e:
        st.error(f"Error saving your response. {e}")
    finally:
        st.session_state.saving = False

if st.session_state.submitted:
    st.caption("Response already submitted for this session.")
