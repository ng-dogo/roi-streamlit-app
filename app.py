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

# CSS minimalista compatible con modo oscuro (no forzamos fondos claros)
BASE_CSS = """
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container { max-width: 980px; }
.row-wrap { padding: .45rem .6rem; border-radius: 10px; border: 1px solid rgba(128,128,128,.20); margin-bottom: .4rem; }
.comp { font-weight: 600; }
.points-pill { display:inline-block; min-width:3.9rem; text-align:center; padding:.28rem .55rem; border-radius: 999px;
               background: rgba(127,127,127,.15); font-variant-numeric: tabular-nums; }
.badge { display:inline-block; padding:.18rem .55rem; border-radius:999px; background: rgba(127,127,127,.15); }
hr { border: none; border-top: 1px solid rgba(128,128,128,.2); margin: 1rem 0; }
.small { opacity:.85; font-size:.92rem; }
.button-mini > div > button { padding:.25rem .5rem; border-radius:10px; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
DEFAULTS_CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # CSV con 2 columnas: component,weight
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "components" not in st.session_state:
    st.session_state.components = []   # lista ordenada de componentes
if "points" not in st.session_state:
    st.session_state.points = {}       # dict comp -> int
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "email" not in st.session_state:
    st.session_state.email = ""

# ─────────────── HELPERS ───────────────
def load_components_and_weights(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """Lee CSV (component, weight). Devuelve DF con Points enteros sumando 100."""
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
    if len(df) == 0:
        return pd.DataFrame(), "CSV vacío."

    def parse_w(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str): x = x.replace("%", "").strip()
        try: return float(x)
        except: return 0.0

    df["w"] = df["weight_raw"].apply(parse_w).clip(lower=0)
    total = df["w"].sum()

    if total <= 0:
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
    return df[["Component", "Points"]], ""

def set_initial_state(df: pd.DataFrame):
    st.session_state.components = df["Component"].tolist()
    st.session_state.points = {r.Component: int(r.Points) for r in df.itertuples()}

def df_from_state() -> pd.DataFrame:
    return pd.DataFrame({
        "Component": st.session_state.components,
        "Points": [int(st.session_state.points[c]) for c in st.session_state.components],
    })

def _round_to_target(values: np.ndarray, target: int) -> np.ndarray:
    """
    Redondeo por mayores restos para que 'values' (float) cierre EXACTO en 'target' al pasar a enteros.
    """
    base = np.floor(values).astype(int)
    delta = target - base.sum()
    if delta != 0:
        restos = (values - base).astype(float)
        order = np.argsort(restos)[::-1]  # mayores restos primero
        for i in range(abs(delta)):
            idx = order[i % len(order)]
            base[idx] += 1 if delta > 0 else -1
    return base

def _rebalance_except(target_comp: str):
    """
    Mantiene SIEMPRE total=100 ajustando todos los demás proporcionalmente (no negativos).
    • Si el comp editado > 100 → se recorta a 100 y el resto a 0.
    • Si el resto suma 0 y hay que repartir, reparte parejo entre los demás.
    """
    comps = st.session_state.components
    points = st.session_state.points
    # Clamp del editado a [0,100]
    points[target_comp] = max(0, min(100, int(points[target_comp])))

    remaining = 100 - points[target_comp]
    if remaining < 0:
        points[target_comp] = 100
        remaining = 0

    others = [c for c in comps if c != target_comp]
    if not others:
        return

    current_sum_others = sum(points[c] for c in others)
    if remaining == 0:
        for c in others: points[c] = 0
        return

    if current_sum_others <= 0:
        # repartir igual
        n = len(others)
        base = remaining // n
        arr = np.array([base] * n, dtype=int)
        for i in range(remaining - base * n):
            arr[i] += 1
        for c, v in zip(others, arr.tolist()): points[c] = int(v)
        return

    # Escalar proporcional y cerrar a entero con mayores restos
    arr_float = np.array([points[c] for c in others], dtype=float)
    arr_scaled = arr_float / arr_float.sum() * remaining
    arr_int = _round_to_target(arr_scaled, remaining)
    # No negativos (por estructura no deberían serlo)
    arr_int = np.clip(arr_int, 0, None)

    for c, v in zip(others, arr_int.tolist()):
        points[c] = int(v)

def adjust(comp: str, delta: int):
    """Click en −10, −1, +1, +10."""
    st.session_state.points[comp] = int(st.session_state.points.get(comp, 0)) + int(delta)
    _rebalance_except(comp)

def set_exact(comp: str, new_val: int):
    """Entrada numérica exacta opcional → rebalance instantáneo del resto."""
    st.session_state.points[comp] = int(new_val)
    _rebalance_except(comp)

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
    set_initial_state(df_init)

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation (BAP)")
st.caption("Asigná **100 puntos**. Los cambios mantienen el total **siempre en 100** automáticamente. Sin sliders, sin locks, sin normalizar.")

# Email (identificador)
email = st.text_input("Email (obligatorio para enviar)", value=st.session_state.email, placeholder="nombre@organizacion.org")
if email != st.session_state.email:
    st.session_state.email = email.strip()

st.write("---")
st.subheader("Asignación")

# Filas con: −10 · −1 · [VALOR] · +1 · +10
for comp in st.session_state.components:
    pts = int(st.session_state.points[comp])

    with st.container():
        st.markdown(f"<div class='row-wrap'><span class='comp'>{comp}</span></div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns([0.9, 0.9, 1.2, 0.9, 0.9, 1.6])

        with c1:
            st.button("−10", key=f"m10_{comp}", on_click=adjust, args=(comp, -10), help="Restar 10", type="secondary")
        with c2:
            st.button("−1",  key=f"m1_{comp}",  on_click=adjust, args=(comp,  -1), help="Restar 1",  type="secondary")

        with c3:
            # Valor en el centro (muestra y permite tipear exacto si hace falta)
            new_val = st.number_input(" ", key=f"num_{comp}", value=pts, min_value=0, max_value=100, step=1,
                                      label_visibility="collapsed")
            if new_val != pts:
                set_exact(comp, new_val)
                pts = int(st.session_state.points[comp])

            # Además mostramos la pill siempre actualizada
            st.markdown(f"<div style='margin-top:.35rem'><span class='points-pill'>{pts}</span></div>", unsafe_allow_html=True)

        with c4:
            st.button("+1",  key=f"p1_{comp}",  on_click=adjust, args=(comp,  +1), help="Sumar 1")
        with c5:
            st.button("+10", key=f"p10_{comp}", on_click=adjust, args=(comp, +10), help="Sumar 10")

        with c6:
            st.caption("Consejo: usá ±10 para cambios grandes y ±1 para fino.", help="UI tip")

# Resumen y gráfico
df_view = df_from_state()
total_now = int(df_view["Points"].sum())

st.write("---")
colA, colB = st.columns([2,1])
with colA:
    st.markdown(f"**Total asignado:** <span class='badge'>{total_now} / 100</span>", unsafe_allow_html=True)
with colB:
    if st.button("Repartir igual", help="Distribuye 100 en partes iguales"):
        n = len(st.session_state.components)
        base = 100 // n
        arr = np.array([base]*n, dtype=int)
        for i in range(100 - base*n):
            arr[i] += 1
        for c, v in zip(st.session_state.components, arr.tolist()):
            st.session_state.points[c] = int(v)
        df_view = df_from_state()
        total_now = int(df_view["Points"].sum())

st.bar_chart(df_view.set_index("Component")["Points"])

st.write("---")
left, right = st.columns([2,1])
with left:
    st.caption("El envío se habilita con email válido (total siempre es 100).")
with right:
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted
        or not EMAIL_RE.match(st.session_state.email or "")
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
