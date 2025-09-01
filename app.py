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

# CSS minimalista: NO forzamos colores de fondo, respetamos el tema (incluye modo oscuro)
BASE_CSS = """
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container { max-width: 980px; }
.row-wrap { padding: .4rem .6rem; border-radius: 10px; border: 1px solid rgba(128,128,128,.2); margin-bottom: .35rem; }
.comp { font-weight: 600; }
.points-pill { display:inline-block; min-width:3.5rem; text-align:center; padding:.25rem .5rem; border-radius: 999px;
               background: rgba(127,127,127,.15); font-variant-numeric: tabular-nums; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; background: rgba(127,127,127,.15); }
hr { border: none; border-top: 1px solid rgba(128,128,128,.2); margin: 1rem 0; }
.small { opacity:.8; font-size:.92rem; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
DEFAULTS_CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # CSV con columnas: component,weight
REQUIRE_TOTAL_100 = True
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "components" not in st.session_state:
    st.session_state.components = []   # lista ordenada de componentes
if "points" not in st.session_state:
    st.session_state.points = {}       # dict comp -> int
if "locks" not in st.session_state:
    st.session_state.locks = {}        # dict comp -> bool
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "email" not in st.session_state:
    st.session_state.email = ""

# ─────────────── HELPERS ───────────────
def load_components_and_weights(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """Lee CSV (component, weight). Devuelve DF con Points enteros que suman ~100 al iniciar y Lock=False."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return pd.DataFrame(), f"No pude leer {csv_path}. Error: {e}"

    cols = {c.strip().lower(): c for c in df.columns}
    if "component" not in cols or "weight" not in cols:
        return pd.DataFrame(), f"El CSV debe tener columnas 'component' y 'weight'. Columnas: {list(df.columns)}"

    comp_col, w_col = cols["component"], cols["weight"]
    df = df[[comp_col, w_col]].rename(columns={comp_col: "Component", w_col: "weight_raw"})
    if df["Component"].isna().any():
        return pd.DataFrame(), "Hay filas sin nombre de componente en el CSV."

    def parse_w(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str): x = x.replace("%", "").strip()
        try: return float(x)
        except: return 0.0

    df["w"] = df["weight_raw"].apply(parse_w).clip(lower=0)
    total = df["w"].sum()

    if len(df) == 0:
        return pd.DataFrame(), "CSV vacío."

    if total <= 0:
        # todos 0 → reparto igual
        n = len(df)
        ints = np.array([100 // n] * n, dtype=int)
        for i in range(100 - ints.sum()): ints[i] += 1
    else:
        scaled = df["w"] / total * 100.0
        ints = np.floor(scaled).astype(int)
        delta = 100 - ints.sum()
        if delta != 0:
            residuos = (scaled - ints).values
            order = np.argsort(residuos)[::-1]
            for i in range(abs(delta)):
                ints.iloc[order[i % len(order)]] += 1 if delta > 0 else -1
        ints = ints.clip(lower=0).astype(int).values

    df["Points"] = ints
    df["Lock"] = False
    return df[["Component", "Points", "Lock"]], ""

def set_initial_state_from_df(df: pd.DataFrame):
    st.session_state.components = df["Component"].tolist()
    st.session_state.points = {r.Component: int(r.Points) for r in df.itertuples()}
    st.session_state.locks  = {r.Component: bool(r.Lock)  for r in df.itertuples()}

def df_from_state() -> pd.DataFrame:
    return pd.DataFrame({
        "Component": st.session_state.components,
        "Points": [int(st.session_state.points[c]) for c in st.session_state.components],
        "Lock":   [bool(st.session_state.locks[c]) for c in st.session_state.components],
    })

def normalize_to_100():
    """Ajusta SOLO no bloqueados para que Locks + Libres = 100, corrigiendo ±1 por redondeo."""
    df = df_from_state()
    locked_sum = int(df.loc[df["Lock"], "Points"].sum())
    free_idx = df.index[~df["Lock"]]
    remaining = max(0, 100 - locked_sum)

    if len(free_idx) == 0:
        # si solo hay locks y excede 100, recorta proporcionalmente locks
        if locked_sum > 100:
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
        # repartir igual entre los libres
        n = len(free_idx)
        base = remaining // n
        vals = [base] * n
        for i in range(remaining - base * n):
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
    """Reparte igual solo entre no bloqueados, respetando lo que esté lockeado."""
    df = df_from_state()
    locked_sum = int(df.loc[df["Lock"], "Points"].sum())
    free_idx = df.index[~df["Lock"]]
    remaining = max(0, 100 - locked_sum)
    n = len(free_idx)
    if n == 0: return
    base = remaining // n
    vals = [base] * n
    for i in range(remaining - base * n):
        vals[i] += 1
    for idx, v in zip(free_idx, vals):
        comp = df.loc[idx, "Component"]
        st.session_state.points[comp] = int(v)

def adjust(comp: str, delta: int):
    """Suma/resta delta al componente si no está lockeado. Mantiene 0..1000 (después normalizás)."""
    if st.session_state.locks.get(comp, False):
        return
    newv = int(st.session_state.points.get(comp, 0)) + int(delta)
    if newv < 0: newv = 0
    if newv > 1000: newv = 1000
    st.session_state.points[comp] = newv

def ensure_sheet_headers(sh, components: list):
    headers = ["timestamp", "email", "session_id"] + components
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
st.caption("Asigná **100 puntos** a los subindicadores. Controles rápidos: −10 / −1 / +1 / +10. Podés bloquear con 🔒. Sin sliders.")

# Email en la misma pantalla (identificador)
email = st.text_input("Email (obligatorio para enviar)", value=st.session_state.email, placeholder="nombre@organizacion.org")
if email != st.session_state.email:
    st.session_state.email = email.strip()

st.write("---")
st.subheader("Asignación")

# Fila por componente
for comp in st.session_state.components:
    pts = int(st.session_state.points[comp])
    lock_key = f"lock_{comp}"
    if lock_key not in st.session_state:
        st.session_state[lock_key] = bool(st.session_state.locks[comp])

    with st.container():
        st.markdown(f"<div class='row-wrap'><span class='comp'>{comp}</span></div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.1, 0.9, 0.9, 0.9, 0.9, 1.4, 1.4])

        with c1:
            st.markdown(f"<span class='points-pill'>{pts}</span>", unsafe_allow_html=True)

        with c2:
            st.button("−10", key=f"m10_{comp}", on_click=adjust, args=(comp, -10), disabled=st.session_state[lock_key])
        with c3:
            st.button("−1",  key=f"m1_{comp}",  on_click=adjust, args=(comp,  -1), disabled=st.session_state[lock_key])
        with c4:
            st.button("+1",  key=f"p1_{comp}",  on_click=adjust, args=(comp,  +1), disabled=st.session_state[lock_key])
        with c5:
            st.button("+10", key=f"p10_{comp}", on_click=adjust, args=(comp, +10), disabled=st.session_state[lock_key])

        with c6:
            # Entrada opcional por si quieren tipear un número exacto
            new_val = st.number_input(" ", key=f"num_{comp}", value=pts, min_value=0, max_value=1000, step=1,
                                      label_visibility="collapsed", disabled=st.session_state[lock_key])
            if new_val != pts:
                st.session_state.points[comp] = int(new_val)

        with c7:
            st.checkbox("🔒 Lock", key=lock_key, value=st.session_state[lock_key])
            st.session_state.locks[comp] = bool(st.session_state[lock_key])

# Totales + acciones globales
df_view = df_from_state()
total_now = int(df_view["Points"].sum())
restantes = 100 - total_now

st.write("---")
colA, colB = st.columns([2, 1])
with colA:
    if restantes == 0:
        st.markdown(f"**Total asignado:** <span class='badge'>{total_now} / 100 (OK)</span>", unsafe_allow_html=True)
    elif restantes > 0:
        st.markdown(f"**Total asignado:** <span class='badge'>{total_now} / 100</span> · te faltan **{restantes}**", unsafe_allow_html=True)
    else:
        st.markdown(f"**Total asignado:** <span class='badge'>{total_now} / 100</span> · te pasaste por **{abs(restantes)}**", unsafe_allow_html=True)

with colB:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Repartir igual"):
            equal_split()
            df_view = df_from_state(); total_now = int(df_view["Points"].sum()); restantes = 100 - total_now
    with c2:
        if st.button("Normalizar a 100"):
            normalize_to_100()
            df_view = df_from_state(); total_now = int(df_view["Points"].sum()); restantes = 100 - total_now

# Vista rápida
st.bar_chart(df_view.set_index("Component")["Points"])

st.write("---")
left, right = st.columns([2, 1])
with left:
    st.caption("El envío se habilita cuando el total es 100 y el email es válido. No usamos sliders para mantener la app fluida con muchos usuarios simultáneos.")
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
