# app.py
import streamlit as st
import pandas as pd
import numpy as np
import gspread, datetime as dt, uuid
from google.oauth2.service_account import Credentials

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

st.markdown("""
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
/* sliders & number inputs */
div[data-baseweb="slider"] > div > div{background:var(--primary);}
div[data-baseweb="slider"] > div > div > div{background:var(--primary);}
.stNumberInput input {text-align:right;}
/* buttons */
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:10px;padding:.5rem 1rem}
.stButton>button:disabled{background:#ccc;color:#666}
/* container */
.main .block-container{max-width:900px}
</style>
""", unsafe_allow_html=True)

# ─────────────── RGI COMPONENTS (8) ───────────────
RGI_COMPONENTS = [
    # poné aquí los nombres EXACTOS que usarás en la presentación/reportes
    "Legal Framework",
    "Independence & Accountability",
    "Tariff Methodology",
    "Participation & Transparency",
    "Legal Mandate",
    "Clarity of Roles & Objectives",
    "Open Access to Information",
    "Transparency",
]

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
    # default: iguales
    st.session_state.weights = {c: round(100/len(RGI_COMPONENTS), 2) for c in RGI_COMPONENTS}

# ─────────────── HELPERS ───────────────
@st.cache_data
def load_defaults_csv(path: str = "rgi_defaults.csv") -> pd.DataFrame:
    # Espera columnas: name, w::Legal Framework, w::Independence & Accountability, ...
    try:
        df = pd.read_csv(path)
        # estandarizo nombre
        df["__name_norm__"] = df["name"].astype(str).str.strip().str.casefold()
        return df
    except Exception:
        return pd.DataFrame()

def get_defaults_for_name(df: pd.DataFrame, name: str) -> dict | None:
    if df.empty or not name.strip():
        return None
    row = df.loc[df["__name_norm__"] == name.strip().casefold()]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    # columnas esperadas: w::<COMPONENT>
    out = {}
    for comp in RGI_COMPONENTS:
        col = f"w::{comp}"
        if col in r and pd.notna(r[col]):
            out[comp] = float(r[col])
    # normalizo a 100
    if out:
        s = sum(v for v in out.values() if v is not None)
        if s > 0:
            out = {k: round(v*100/s, 2) for k,v in out.items()}
        # completa faltantes con 0
        for comp in RGI_COMPONENTS:
            out.setdefault(comp, 0.0)
        return out
    return None

def normalize_to_100(d: dict[str,float]) -> dict[str,float]:
    vals = np.array([max(0.0, float(d[k])) for k in RGI_COMPONENTS], dtype=float)
    s = vals.sum()
    if s <= 0:
        vals[:] = 100.0/len(vals)
    else:
        vals = vals * (100.0/s)
    return {k: round(v, 2) for k, v in zip(RGI_COMPONENTS, vals)}

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
    ]
    # agrego los 8 pesos en el mismo orden fijo
    row += [weights[c] for c in RGI_COMPONENTS]
    sh.append_row(row)

# ─────────────── HEADER ───────────────
st.image("ad_hoc_logo.png", width=140)
st.title("Regulatory Governance Index (RGI) – Budget Allocation")

defaults_df = load_defaults_csv()  # rgi_defaults.csv junto al app.py

with st.expander("What is this?", expanded=False):
    st.markdown("""
Distribute **100%** across the **8 RGI components** according to their relative importance.
If we have your **Mentimeter**-based weights, we'll pre-fill them when you enter your name.
""")

# ─────────────── USER INFO ───────────────
st.subheader("Your details")
c1, c2 = st.columns(2)
with c1:
    first = st.text_input("First name", value=st.session_state.user["first"])
    entity = st.text_input("Organization / Regulatory entity", value=st.session_state.user["entity"])
with c2:
    last = st.text_input("Last name", value=st.session_state.user["last"])
    country = st.text_input("Country", value=st.session_state.user["country"])

st.session_state.user = {
    "first": first.strip(), "last": last.strip(),
    "entity": entity.strip(), "country": country.strip()
}

# Botón para precargar defaults desde CSV por nombre completo
prefill_col = st.columns([1,1,3])[0]
if prefill_col.button("🔎 Load my defaults (by name)"):
    name_join = f"{st.session_state.user['first']} {st.session_state.user['last']}".strip()
    found = get_defaults_for_name(defaults_df, name_join)
    if found:
        st.session_state.weights = found
        st.success("Defaults loaded from CSV.")
    else:
        st.warning("No defaults found for that name. Using equal shares.")

st.markdown("---")

# ─────────────── BUDGET ALLOCATION ───────────────
st.subheader("Allocate **100%** across components")

# número de columnas (4 filas x 2 columnas → 8 inputs prolijos)
rows = [RGI_COMPONENTS[i:i+2] for i in range(0, len(RGI_COMPONENTS), 2)]
new_vals = {}

for row in rows:
    cA, cB = st.columns(2)
    with cA:
        comp = row[0]
        new_vals[comp] = st.number_input(
            f"{comp}", min_value=0.0, max_value=100.0, step=1.0,
            value=float(st.session_state.weights.get(comp, 0.0)), key=f"num_{comp}"
        )
    if len(row) > 1:
        with cB:
            comp = row[1]
            new_vals[comp] = st.number_input(
                f"{comp}", min_value=0.0, max_value=100.0, step=1.0,
                value=float(st.session_state.weights.get(comp, 0.0)), key=f"num_{comp}"
            )

# normalizo y muestro resumen
normalized = normalize_to_100(new_vals)
st.session_state.weights = normalized

st.markdown("#### Total allocated")
total = sum(normalized.values())
st.progress(min(int(total), 100))
st.caption(f"Sum after normalization: **{total:.2f}%** (always kept at 100%)")

# tabla + gráfico
df = (pd.DataFrame({"Component": list(normalized.keys()), "Weight (%)": list(normalized.values())})
        .sort_values("Weight (%)", ascending=False))
st.markdown("### 📊 Distribution")
st.bar_chart(df.set_index("Component"))

st.markdown("### 🏁 Priority ranking")
for i, r in enumerate(df.itertuples(index=False), 1):
    st.write(f"**{i}. {r.Component} – {r._2:.2f}%**")

st.markdown("---")

# ─────────────── PRESETS ───────────────
with st.expander("⚡ Quick presets", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Equal (12.5% each)"):
        st.session_state.weights = {c: round(100/len(RGI_COMPONENTS),2) for c in RGI_COMPONENTS}
        st.rerun()
    if col_b.button("Emphasize Governance (LF + I&A)"):
        tmp = {c: 5.0 for c in RGI_COMPONENTS}
        tmp["Legal Framework"] = 30.0
        tmp["Independence & Accountability"] = 30.0
        st.session_state.weights = normalize_to_100(tmp)
        st.rerun()
    if col_c.button("Emphasize Transparency"):
        tmp = {c: 7.0 for c in RGI_COMPONENTS}
        for c in ["Open Access to Information","Transparency","Participation & Transparency"]:
            if c in tmp: tmp[c] = tmp[c] + 15.0
        st.session_state.weights = normalize_to_100(tmp)
        st.rerun()

# ─────────────── SUBMIT ───────────────
st.markdown("---")
disabled_submit = (
    st.session_state.saving or st.session_state.submitted or
    not all(st.session_state.user.values())
)
if st.button("📤 Submit my allocation", disabled=disabled_submit):
    st.session_state.saving = True
    try:
        save_to_sheet(st.session_state.user, st.session_state.weights, st.session_state.session_id)
        st.session_state.submitted = True
        st.success("✅ Thank you! Your submission has been recorded.")
        st.balloons()
    except Exception as e:
        st.error(f"⚠️ There was an error saving your response. {e}")
    finally:
        st.session_state.saving = False

if st.session_state.submitted:
    st.caption("Response already submitted for this session.")
