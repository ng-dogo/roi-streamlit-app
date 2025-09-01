# app.py
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt, re, os
from typing import Tuple
from google.oauth2.service_account import Credentials
import gspread

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation (BAP)", page_icon="⚡", layout="centered")

MINI_CSS = """
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:10px;padding:.25rem .6rem}
.stButton>button:disabled{background:#ccc;color:#666}
.main .block-container{max-width:900px}
hr{border:none;border-top:1px solid #e6e6e6;margin:1rem 0}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:#f5f5f5}
.total-ok{color:#0a7a3b}
.total-hi{color:#b00020}
.total-low{color:#6b6b6b}
.row-wrap{padding:.35rem .5rem;border-radius:10px;border:1px solid #eee;margin-bottom:.4rem}
.comp{font-weight:600}
.locklab{color:#555;font-size:.9rem}
.points-pill{display:inline-block; min-width:3rem; text-align:center; padding:.25rem .45rem; border-radius:999px; background:#f1f5f9; font-variant-numeric: tabular-nums;}
</style>
"""
st.markdown(MINI_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
DEFAULTS_CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # 2 columnas: component,weight
REQUIRE_TOTAL_100 = True
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "components" not in st.session_state:
    st.session_state.components = []  # lista de nombres
if "points" not in st.session_state:
    st.session_state.points = {}      # dict comp -> int
if "locks" not in st.session_state:
    st.session_state.locks = {}       # dict comp -> bool
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "email" not in st.session_state:
    st.session_state.email = ""

# ─────────────── HELPERS ───────────────
def load_components_and_weights(csv_path:str) -> Tuple[pd.DataFrame, str]:
    """Lee CSV con columnas: component,weight. Normaliza a enteros que suman 100 (para iniciar)."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return pd.DataFrame(), f"No pude leer {csv_path}. Error: {e}"

    cols = {c.strip().lower(): c for c in df.columns}
    if "component" not in cols or "weight" not in cols:
        return pd.DataFrame(), f"El CSV debe tener columnas 'component' y 'weight'. Columnas detectadas: {list(df.columns)}"

    comp_col, w_col = cols["component"], cols["weight"]
    df = df[[comp_col, w_col]].rename(columns={comp_col:"Component", w_col:"weight_raw"})
    if df["Component"].isna().any():
        return pd.DataFrame(), "Hay filas sin nombre de componente en el CSV."

    def parse_w(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str): x = x.replace("%","").strip()
        try: return float(x)
        except: return 0.0

    df["w"] = df["weight_raw"].apply(parse_w).clip(lower=0)
    total = df["w"].sum()

    if total <= 0:
        n = len(df)
        ints = np.array([100//n]*n, dtype=int)
        for i in range(100 - ints.sum()): ints[i] += 1
    else:
        scaled = df["w"] / total * 100.0
        ints = np.floor(scaled).astype(int)
        delta = 100 - ints.sum()
        if delta != 0:
            residuos = (scaled - ints).values
            order = np.argsort(residuos)[::-1]
            for i in range(abs(delta)):
                idx = order[i % len(order)]
                ints.iloc[idx] += 1 if delta > 0 else -1
        ints = ints.clip(lower=0).astype(int).values

    df["Points"] = ints
    df["Lock"] = False
    return df[["Component","Points","Lock"]], ""

def set_initial_state_from_df(df: pd.DataFrame):
    st.session_state.components = df["Component"].tolist()
    st.session_state.points = {r.Component: int(r.Points) for r in df.itertuples()}
    st.session_state.locks = {r.Component: bool(r.Lock) for r in df.itertuples()}

def df_from_state() -> pd.DataFrame:
    return pd.DataFrame({
        "Component": st.session_state.components,
        "Points": [st.session_state.points[c] for c in st.session_state.components],
        "Lock":   [st.session_state.locks[c]  for c in st.session_state.components],
    })

def normalize_to_100():
    df = df_from_state()
    locked_sum = int(df.loc[df["Lock"], "Points"].sum())
    free_idx = df.index[~df["Lock"]]
    remaining = max(0, 100 - locked_sum)

    if len(free_idx) == 0:
        # No hay libres: sólo recorte si excede 100
        if locked_sum > 100:
            # recortar proporcionalmente locks
            locks = df.loc[df["Lock"], "Points"].astype(float)
            if locks.sum() > 0:
                scaled = locks / locks.sum() * 100
                ints = np.floor(scaled).astype(int)
                delta = 100 - ints.sum()
                residuos = (scaled - ints).values
                order = np.argsort(residuos)[::-1]
                for i in range(abs(delta)):
                    ints.iloc[order[i % len(order)]] += 1 if delta > 0 else -1
                for comp, v in zip(df.loc[df["Lock"], "Component"], ints.tolist()):
                    st.session_state.points[comp] = int(v)
        return

    free_vals = df.loc[free_idx, "Points"].astype(float).clip(lower=0)
    if free_vals.sum() <= 0:
        n = len(free_idx)
        base = remaining // n
        vals = [base]*n
        for i in range(remaining - base*n):
            vals[i] += 1
    else:
        scaled = free_vals / free_vals.sum() * remaining
        ints = np.floor(scaled).astype(int)
        delta = remaining - ints.sum()
        residuos = (scaled - ints).values
        order = np.argsort(residuos)[::-1]
        for i in range(abs(delta)):
            ints.iloc[order[i % len(order)]] += 1 if delta > 0 else -1
        vals = ints.clip(lower=0).astype(int).tolist()

    for idx, v in zip(free_idx, vals):
        comp = df.loc[idx, "Component"]
        st.session_state.points[comp] = int(v)

def equal_split():
    df = df_from_state()
    locked_sum = int(df.loc[df["Lock"], "Points"].sum())
    free_idx = df.index[~df["Lock"]]
    remaining = max(0, 100 - locked_sum)
    n = len(free_idx)
    if n == 0: return
    base = remaining // n
    vals = [base]*n
    for i in range(remaining - base*n):
        vals[i] += 1
    for idx, v in zip(free_idx, vals):
        comp = df.loc[idx, "Component"]
        st.session_state.points[comp] = int(v)

def adjust(comp: str, delta: int):
    """Suma/resta delta a un componente si no está bloqueado. Recorta en [0, 1000] para edición libre; luego normalizás."""
    if st.session_state.locks.get(comp, False):
        return
    newv = int(st.session_state.points.get(comp, 0)) + int(delta)
    if newv < 0: newv = 0
    if newv > 1000: newv = 1000  # límite alto por si quieren cargar y normalizar después
    st.session_state.points[comp] = newv

def ensure_sheet_headers(sh, components: list):
    headers = ["timestamp","email","session_id"] + components
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)

def save_to_gsheet(email: str, df: pd.DataFrame, session_id: str):
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    ensure_sheet_headers(sh, df["Component"].tolist())
    row = [dt.datetime.now().isoformat(), email, session_id] + df["Points"].astype(int).tolist()
    sh.append_row(row)

# ─────────────── DATA INIT ───────────────
if not st.session_state.components:
    df_init, err = load_components_and_weights(DEFAULTS_CSV_PATH)
    if err:
        st.error(err); st.stop()
    set_initial_state_from_df(df_init)

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation (BAP)")
st.caption("Asigná **100 puntos** a los subindicadores. Usá los botones rápidos (−10/−1/+1/+10) o normalizá automáticamente. Podés bloquear con 🔒.")

# Email en la misma pantalla
email = st.text_input("Email (obligatorio para enviar)", value=st.session_state.email, placeholder="nombre@organizacion.org")
if email != st.session_state.email:
    st.session_state.email = email.strip()

st.write("---")

# Controles por fila
st.subheader("Asignación")
for comp in st.session_state.components:
    pts = int(st.session_state.points[comp])
    lock = bool(st.session_state.locks[comp])

    with st.container():
        st.markdown(f"<div class='row-wrap'><span class='comp'>{comp}</span></div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.0, 0.7, 0.7, 0.7, 0.7, 1.2, 1.2])

        with c1:
            st.markdown(f"<span class='points-pill'> {pts} </span>", unsafe_allow_html=True)

        with c2:
            st.button("−10", key=f"m10_{comp}", on_click=adjust, args=(comp, -10), disabled=lock)
        with c3:
            st.button("−1", key=f"m1_{comp}", on_click=adjust, args=(comp, -1), disabled=lock)
        with c4:
            st.button("+1", key=f"p1_{comp}", on_click=adjust, args=(comp, +1), disabled=lock)
        with c5:
            st.button("+10", key=f"p10_{comp}", on_click=adjust, args=(comp, +10), disabled=lock)

        with c6:
            # Entrada numérica opcional (rápida) por si quieren tipear
            new_val = st.number_input(" ", key=f"num_{comp}", value=pts, min_value=0, max_value=1000, step=1, label_visibility="collapsed", disabled=lock)
            if new_val != pts:
                st.session_state.points[comp] = int(new_val)

        with c7:
            st.checkbox("🔒 Lock", key=f"lock_{comp}", value=lock, on_change=lambda c=comp: st.session_state.locks.__setitem__(c, not st.session_state.locks[c]) if False else None)
            # Nota: el checkbox de Streamlit ya sincroniza el valor en session_state; la fila usa st.session_state.locks

# Total y acciones globales
df_view = df_from_state()
total_now = int(df_view["Points"].sum())
restantes = 100 - total_now

st.write("---")
colA, colB = st.columns([2,1])
with colA:
    if restantes == 0:
        st.markdown(f"**Total asignado:** <span class='badge total-ok'>{total_now} / 100 (OK)</span>", unsafe_allow_html=True)
    elif restantes > 0:
        st.markdown(f"**Total asignado:** <span class='badge total-low'>{total_now} / 100</span> · te faltan **{restantes}**", unsafe_allow_html=True)
    else:
        st.markdown(f"**Total asignado:** <span class='badge total-hi'>{total_now} / 100</span> · te pasaste por **{abs(restantes)}**", unsafe_allow_html=True)

with colB:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Repartir igual"):
            equal_split()
    with c2:
        if st.button("Normalizar a 100"):
            normalize_to_100()
            df_view = df_from_state()
            total_now = int(df_view["Points"].sum())
            restantes = 100 - total_now

# Vista rápida
st.bar_chart(df_view.set_index("Component")["Points"])

st.write("---")
left, right = st.columns([2,1])
with left:
    st.caption("El envío se habilita cuando el total es 100 y el email es válido.")
with right:
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted
        or not EMAIL_RE.match(st.session_state.email or "")
        or (REQUIRE_TOTAL_100 and total_now != 100)
    )
    if st.button("Enviar", use_container_width=True, disabled=disabled_submit):
        st.session_state.saving = True
        try:
            save_to_gsheet(st.session_state.email, df_from_state(), st.session_state.session_id)
            st.session_state.submitted = True
            st.success("✅ Respuesta guardada. ¡Gracias!")
        except Exception as e:
            st.error(f"Error guardando en Google Sheets. {e}")
        finally:
            st.session_state.saving = False

if st.session_state.submitted:
    st.caption("Ya enviaste tu respuesta en esta sesión.")
