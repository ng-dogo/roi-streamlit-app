# app.py
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt, re
from typing import Tuple
from google.oauth2.service_account import Credentials
import gspread
import os

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation (BAP)", page_icon="⚡", layout="centered")

MINI_CSS = """
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:8px;padding:.5rem 1rem}
.stButton>button:disabled{background:#ccc;color:#666}
.main .block-container{max-width:880px}
hr{border:none;border-top:1px solid #e6e6e6;margin:1rem 0}
.label-small{color:#666;font-size:0.9rem}
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
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Component","Points","Lock"])
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "email" not in st.session_state:
    st.session_state.email = ""

# ─────────────── HELPERS ───────────────
def load_components_and_weights(csv_path:str) -> Tuple[pd.DataFrame, str]:
    """
    Lee un CSV con 2 columnas: component, weight (puede ser % o numérico).
    Devuelve DF normalizado a enteros (solo como inicio; el usuario puede editar).
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return pd.DataFrame(), f"No pude leer {csv_path}. Error: {e}"

    # Normalizar nombres de columnas
    cols = {c.strip().lower(): c for c in df.columns}
    if "component" not in cols or "weight" not in cols:
        return pd.DataFrame(), f"El CSV debe tener columnas 'component' y 'weight'. Columnas detectadas: {list(df.columns)}"

    comp_col = cols["component"]
    w_col = cols["weight"]

    df = df[[comp_col, w_col]].rename(columns={comp_col:"Component", w_col:"weight_raw"})
    if df["Component"].isna().any():
        return pd.DataFrame(), "Hay filas sin nombre de componente en el CSV."

    # Parsear pesos: aceptar números o strings con %
    def parse_weight(x):
        if pd.isna(x): return 0.0
        if isinstance(x, str): x = x.replace("%","").strip()
        try: return float(x)
        except: return 0.0

    df["weight_float"] = df["weight_raw"].apply(parse_weight).clip(lower=0)
    total = df["weight_float"].sum()

    # Si parece que vienen en [0..100] pero no suman ~100, normalizo a 100.
    if total <= 0:
        # Si todos son 0, repartir igual
        n = len(df)
        df["Points"] = [int(100//n)]*n
        resto = 100 - df["Points"].sum()
        for i in range(resto):
            df.loc[i, "Points"] += 1
    else:
        # Escalar para que sumen 100 con enteros
        scaled = df["weight_float"] / total * 100.0
        ints = np.floor(scaled).astype(int)
        delta = 100 - ints.sum()
        # Ajuste por los mayores residuos
        residuos = (scaled - ints).values
        order = np.argsort(residuos)[::-1]
        for i in range(abs(delta)):
            idx = order[i % len(order)]
            ints.iloc[idx] += 1 if delta > 0 else -1
        df["Points"] = ints.clip(lower=0)

    df["Lock"] = False
    return df[["Component","Points","Lock"]], ""

def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza sólo los no bloqueados a que (Locks + Libres) = 100.
    Respeta enteros. Corrige ±1 por redondeo.
    """
    out = df.copy()
    locked_sum = out.loc[out["Lock"], "Points"].sum()
    free_idx = out.index[~out["Lock"]]
    free = out.loc[free_idx, "Points"].astype(float).clip(lower=0)

    remaining = max(0, 100 - int(locked_sum))
    if free.sum() <= 0:
        # Reparto igual entre libres
        n = len(free_idx)
        if n == 0:
            return out
        base = remaining // n
        pts = [base]*n
        add = remaining - base*n
        for i in range(add):
            pts[i] += 1
        out.loc[free_idx, "Points"] = pts
        return out

    scaled = free / free.sum() * remaining
    ints = np.floor(scaled).astype(int)
    delta = remaining - ints.sum()
    residuos = (scaled - ints).values
    order = np.argsort(residuos)[::-1]
    for i in range(abs(delta)):
        idx = order[i % len(order)]
        ints.iloc[idx] += 1 if delta > 0 else -1

    out.loc[free_idx, "Points"] = ints.clip(lower=0).astype(int)
    return out

def equal_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reparte igual solo entre no bloqueados.
    """
    out = df.copy()
    locked_sum = out.loc[out["Lock"], "Points"].sum()
    free_idx = out.index[~out["Lock"]]
    remaining = max(0, 100 - int(locked_sum))

    n = len(free_idx)
    if n == 0:
        return out

    base = remaining // n
    pts = [base]*n
    add = remaining - base*n
    for i in range(add):
        pts[i] += 1
    out.loc[free_idx, "Points"] = pts
    return out

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

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation (BAP)")
st.caption("Distribuí **100 puntos** entre los subindicadores. Podés bloquear algunos, normalizar o repartir en partes iguales. Guardamos tu email como identificador.")

# 1) Email (en la misma pantalla)
email = st.text_input("Email (obligatorio para enviar)", value=st.session_state.email, placeholder="nombre@organizacion.org")
if email != st.session_state.email:
    st.session_state.email = email.strip()

# 2) Cargar DF desde CSV (una sola vez por sesión)
if st.session_state.df.empty:
    df_loaded, err = load_components_and_weights(DEFAULTS_CSV_PATH)
    if err:
        st.error(err)
        st.stop()
    st.session_state.df = df_loaded.copy()

# 3) Editor dentro de un form (aplica cambios en lote → evita reruns por tecla)
with st.form("bap_form"):
    st.subheader("Asignación de puntos")
    st.markdown("<span class='label-small'>Tips: Podés marcar 🔒 para fijar un valor. Luego usá “Normalizar a 100” para ajustar el resto.</span>", unsafe_allow_html=True)

    edited = st.data_editor(
        st.session_state.df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_order=["Component","Points","Lock"],
        column_config={
            "Component": st.column_config.TextColumn(disabled=True),
            "Points": st.column_config.NumberColumn(min_value=0, step=1),
            "Lock": st.column_config.CheckboxColumn()
        }
    )

    total = int(pd.to_numeric(edited["Points"], errors="coerce").fillna(0).sum())
    restantes = 100 - total
    if restantes == 0:
        st.success("🎯 La suma es 100.")
    elif restantes > 0:
        st.info(f"Te faltan **{restantes}** puntos para llegar a 100.")
    else:
        st.warning(f"Te pasaste por **{abs(restantes)}** puntos.")

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        btn_apply = st.form_submit_button("Aplicar cambios")
    with c2:
        btn_equal = st.form_submit_button("Repartir igual")
    with c3:
        btn_norm = st.form_submit_button("Normalizar a 100")
    with c4:
        btn_clear = st.form_submit_button("Limpiar (poner 0 a no bloqueados)")

    # Procesar acciones
    if btn_apply:
        st.session_state.df = edited.copy()

    if btn_equal:
        base = edited.copy()
        st.session_state.df = equal_split(base)

    if btn_norm:
        base = edited.copy()
        st.session_state.df = normalize_to_100(base)

    if btn_clear:
        base = edited.copy()
        free_idx = base.index[~base["Lock"]]
        base.loc[free_idx, "Points"] = 0
        st.session_state.df = base

# 4) Resumen + gráfico
df_view = st.session_state.df.copy()
st.subheader("Vista previa")
st.bar_chart(df_view.set_index("Component")["Points"])

# 5) Submit
total_now = int(df_view["Points"].sum())
disabled_submit = (
    st.session_state.saving or st.session_state.submitted
    or not EMAIL_RE.match(st.session_state.email or "")
    or (REQUIRE_TOTAL_100 and total_now != 100)
)

st.write("---")
left, right = st.columns([2,1])
with left:
    st.markdown(f"**Total asignado:** {total_now} / 100")
    st.caption("El botón de enviar se habilita sólo cuando total = 100 y el email es válido.")
with right:
    if st.button("Enviar", disabled=disabled_submit, use_container_width=True):
        st.session_state.saving = True
        try:
            save_to_gsheet(st.session_state.email, st.session_state.df, st.session_state.session_id)
            st.session_state.submitted = True
            st.success("✅ Respuesta guardada. ¡Gracias!")
        except Exception as e:
            st.error(f"Error guardando en Google Sheets. {e}")
        finally:
            st.session_state.saving = False

if st.session_state.submitted:
    st.caption("Ya enviaste tu respuesta en esta sesión.")
