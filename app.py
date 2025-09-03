# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib
from typing import Dict, List
from threading import Lock
import random
import os, psutil

from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError

# ───────── CONFIG ─────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18); }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width:860px;padding-top:14px;padding-bottom:36px}
hr{border:none;border-top:1px solid var(--border);margin:.65rem 0}

.name{font-weight:600;margin:.15rem 0 .15rem}
.small-note{font-size:.9rem;color:var(--muted);margin:.25rem 0 0}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;border:1px solid var(--border);font-size:.85rem;color:var(--muted)}
.kpis{display:flex;gap:.6rem;align-items:center}
.kpis .strong{font-weight:700}

/* Buttons */
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.4rem .85rem}
.stButton>button:hover{filter:brightness(0.95)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default}

/* —— Minimal tables —— */
.table-title{font-weight:700;text-align:center;margin:.25rem 0 .35rem}
.table-box{padding:.35rem .5rem;border-radius:12px;border:1px solid var(--border);}
.minitable{width:100%;border-collapse:collapse;font-size:.95rem}
.minitable th, .minitable td{padding:.35rem .5rem;border-bottom:1px solid var(--border)}
.minitable th{font-weight:600;color:var(--muted);text-align:center}

/* Ranking table */
.rank .r{text-align:center}
.rank td:first-child, .rank td:last-child{text-align:center}
.rank td:nth-child(2){text-align:left}

/* —— Allocation 'tablita' look (rows with widgets) —— */
.alloc-head{display:flex;align-items:center}
.alloc-row{display:flex;align-items:center;border-top:1px solid var(--border);padding:.3rem .15rem}
.alloc-col-name{flex:1 1 auto;padding:0 .35rem}
.alloc-col-input{flex:0 0 160px;padding:0 .35rem}

/* Inputs tighter + centered numbers */
.alloc-col-input input[type=number]{
  padding:.25rem .45rem !important;
  height:34px !important;
  font-variant-numeric:tabular-nums;
  text-align:center;
  font-weight:600;
}
/* shrink vertical gaps inside number_input container */
div[data-baseweb="input"]{min-height:34px}
label[for^="num_"]{display:none}

/* Centered header cells alignment harmony */
.head-cell{font-weight:600;color:var(--muted);text-align:center;width:160px}

/* —— Floating HUD —— */
.hud{
  position: fixed; left: 12px; bottom: 12px; width: 62vw; max-width: 680px;
  background: rgba(255,255,255,.9); backdrop-filter: blur(6px);
  border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.08);
  padding: .45rem .65rem; z-index: 9999;
}
.dark .hud { background: rgba(28,28,28,.85) }
.hud-row{ display:flex; align-items:center; gap:.6rem }
.hud-mono{ font-variant-numeric: tabular-nums; font-weight:600 }
.hud-spacer{ flex:1 }
.hud-bar{ position:relative; height:8px; background: rgba(127,127,127,.18); border-radius:999px; overflow:hidden; width:52% }
.hud-fill{ position:absolute; left:0; top:0; bottom:0; background: var(--brand); width:0% }
@media (hover:hover){ .hud:hover{ box-shadow:0 8px 26px rgba(0,0,0,.12) } }
@media (max-width: 520px){
  .hud { width: calc(100vw - 24px); left:12px; right:12px; }
  .alloc-col-input{flex-basis:130px}
  .head-cell{width:130px}
}
@media (prefers-color-scheme: dark){
  .hud{ background: rgba(18,18,18,.85); border-color: rgba(255,255,255,.12); }
  .hud-mono{ color: rgba(255,255,255,.92); }
  .hud-bar{ background: rgba(255,255,255,.15); }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 1.0  # pesos suman 1.00
EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
EPS = 1e-6  # tolerancia numérica

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
if "_init_inputs" not in st.session_state:
    st.session_state._init_inputs = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "last_submit_ts" not in st.session_state:
    st.session_state.last_submit_ts = 0.0
if "last_payload_hash" not in st.session_state:
    st.session_state.last_payload_hash = ""
if "inflight_payload_hash" not in st.session_state:
    st.session_state.inflight_payload_hash = ""
if "status" not in st.session_state:
    st.session_state.status = "idle"

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
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    return out

def round_to_cents_preserve_total(weights: Dict[str, float]) -> Dict[str, float]:
    """Redondea a 0.01 manteniendo suma exacta = 1.00 (en centésimas)."""
    if not weights:
        return {}
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        cents_each = int(round(100 / n))
        cents = [cents_each]*n
        diff = 100 - sum(cents)
        for i in range(abs(diff)):
            idx = i % n
            cents[idx] += 1 if diff > 0 else -1
        return {k: v/100.0 for k, v in zip(weights.keys(), cents)}
    scaled = {k: 100.0 * (v / total) for k, v in weights.items()}
    rounded = {k: int(np.floor(s + 0.5)) for k, s in scaled.items()}
    resid = {k: (scaled[k] - rounded[k]) for k in weights}
    diff = 100 - sum(rounded.values())
    if diff > 0:
        order = sorted(weights.keys(), key=lambda k: resid[k], reverse=True)
        for k in order[:diff]:
            rounded[k] += 1
    elif diff < 0:
        order = sorted(weights.keys(), key=lambda k: resid[k])
        for k in order[:abs(diff)]:
            rounded[k] -= 1
    return {k: rounded[k] / 100.0 for k in rounded}

def remaining_points(weights: Dict[str, float]) -> float:
    return float(TOTAL_POINTS - float(sum(weights.values())))

def make_on_change_free(comp: str):
    """Actualiza el valor sin forzar que la suma <= 1 (solo acota cada indicador a [0,1])."""
    def _cb():
        try:
            v = float(st.session_state.get(f"num_{comp}", 0.0))
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))  # cada indicador 0..1
        st.session_state.weights[comp] = float(np.round(v + 1e-9, 2))
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    return _cb

def payload_hash(email: str, indicators: List[str], weights: Dict[str, float]) -> str:
    tpl = (email.strip().lower(), tuple(indicators), tuple(float(weights[k]) for k in indicators))
    return hashlib.sha256(repr(tpl).encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def get_worksheet():
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\\n").replace("\\\\n", "\\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    return sh

@st.cache_resource(show_spinner=False)
def get_submit_lock() -> Lock:
    return Lock()

def _ensure_header_once(sh, headers: List[str]) -> None:
    """Escribe encabezado solo si hace falta (tolerante a concurrencia)."""
    try:
        first_row = sh.row_values(1)
        if first_row and first_row[:len(headers)] == headers:
            return
    except Exception:
        pass
    try:
        sh.update("A1", [headers])
    except Exception:
        pass

def save_to_sheet(email: str, weights: Dict[str, float], session_id: str, indicator_order: List[str]):
    sh = get_worksheet()
    headers = ["timestamp","email","session_id"] + indicator_order + ["total"]
    _ensure_header_once(sh, headers)
    row = (
        [dt.datetime.now().isoformat(), email, session_id]
        + [float(np.round(weights[k], 2)) for k in indicator_order]
        + [float(np.round(sum(weights.values()), 2))]
    )
    base_delay = 0.4
    attempts = 5
    for attempt in range(1, attempts + 1):
        time.sleep(random.uniform(0.0, 0.35))  # jitter anti-rafa
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError as e:
            if attempt == attempts:
                raise RuntimeError(
                    "No pudimos guardar en Google Sheets tras varios intentos. "
                    "Por favor, esperá unos segundos y hacé un único reintento."
                ) from e
            time.sleep(min(4.0, base_delay * (2 ** (attempt - 1))))

# ───────── LOAD DEFAULTS ─────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    defaults_cents = round_to_cents_preserve_total(defaults_raw)
    st.session_state.defaults = defaults_cents
    st.session_state.weights = dict(defaults_cents)
    st.session_state._init_inputs = True
else:
    indicators = list(st.session_state.weights.keys())

# ───────── UI ─────────
st.title("RGI – Budget Allocation Points")

# Email + Reset
row_top = st.columns([3,1])
with row_top[0]:
    st.session_state.email = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")
with row_top[1]:
    if st.button("Reset to averages", disabled=st.session_state.saving):
        st.session_state.weights = dict(st.session_state.defaults)
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
        st.rerun()

# Inicializa inputs visibles exactamente a los valores en estado
if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

# ───────── HUD FLOTANTE ─────────
def render_floating_hud(used: float):
    pct = max(0.0, min(1.0, used)) * 100.0 if TOTAL_POINTS else 0.0
    st.markdown(f"""
    <div class="hud">
      <div class="hud-row">
        <div class="hud-mono">{used:.2f}/1.00</div>
        <div class="hud-spacer"></div>
        <div class="hud-bar"><div class="hud-fill" style="width:{pct:.2f}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

used_now = float(sum(st.session_state.weights.values()))
render_floating_hud(used_now)

# ───────── WARNING (arriba del Allocation) ─────────
def render_sum_warning():
    used = float(sum(st.session_state.weights.values()))
    rem = remaining_points(st.session_state.weights)
    if abs(rem) <= EPS:
        st.markdown(
            f"""
            <div style="
                margin:.5rem 0;
                padding:.55rem .85rem;
                border:1px solid rgba(0,140,78,.35);
                background:rgba(0,140,78,.08);
                border-radius:8px;
                font-size:.95rem;
                color:#0E7C66;">
                ✅ The weights sum to 1.00. You can submit.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
        st.markdown(
            f"""
            <div style="
                margin:.5rem 0;
                padding:.55rem .85rem;
                border:1px solid rgba(217,48,37,.35);
                background:rgba(217,48,37,.08);
                border-radius:8px;
                font-size:.95rem;
                color:#b3261e;">
                ⚠️ The weights must sum to 1.00. {tip} to continue.
            </div>
            """,
            unsafe_allow_html=True
        )

render_sum_warning()

# ───────── ALLOCATION (tablita compacta: Indicador | Input) ─────────
st.subheader("Allocation", divider="gray")

# Caja que envuelve a la "tabla"
st.markdown('<div class="table-box">', unsafe_allow_html=True)

# Header con dos "columnas"
head_l, head_r = st.columns([1, 0.34])
with head_l:
    st.markdown('<div class="table-title" style="margin:.1rem 0 .25rem"> </div>', unsafe_allow_html=True)
    st.markdown('<div class="name" style="margin:.1rem 0 .35rem">Indicator</div>', unsafe_allow_html=True)
with head_r:
    st.markdown('<div class="table-title" style="margin:.1rem 0 .25rem"> </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="head-cell" style="margin:.1rem 0 .35rem">Weight</div>', unsafe_allow_html=True)

# Filas con aspecto de tabla
for comp in indicators:
    row_l, row_r = st.columns([1, 0.34])
    with row_l:
        st.markdown(f'<div class="alloc-row"><div class="alloc-col-name">{comp}</div></div>', unsafe_allow_html=True)
    with row_r:
        # Para que el input quede alineado con la fila, pintamos solo el input (el borde de fila lo simula la columna izquierda)
        st.number_input(
            label="",
            key=f"num_{comp}",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            label_visibility="collapsed",
            on_change=make_on_change_free(comp),
            disabled=st.session_state.saving
        )

st.markdown('</div>', unsafe_allow_html=True)  # /table-box

# ───────── RANKING (debajo de Allocation, arriba del Submit) ─────────
def render_ranking_html(weights: Dict[str, float]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
    rows = []
    for idx, (name, pts) in enumerate(ordered, start=1):
        rows.append(f"<tr><td>{idx}</td><td>{name}</td><td class='r'>{float(pts):.2f}</td></tr>")
    table_html = f"""
    <div class='table-box'>
      <div class='table-title'>Ranking</div>
      <table class="minitable rank">
        <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
render_ranking_html(st.session_state.weights)

# Memoria (útil para diagnóstico)
mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
st.caption(f"RAM usada por el proceso: {mem_mb:.1f} MB")

# ───────── FOOTER / SUBMIT ─────────
st.markdown("<hr/>", unsafe_allow_html=True)
email_raw = st.session_state.email or ""
email_norm = email_raw.strip()
ok_email = bool(EMAIL_RE.match(email_norm))
now = time.time()
cooling = (now - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC

disabled_submit = (
    (not ok_email)
    or st.session_state.submitted
    or (abs(remaining_points(st.session_state.weights)) > EPS)  # debe sumar 1.00 exacto
    or st.session_state.saving
    or cooling
)

status_box = st.empty()

submit_label = "Submit" if not st.session_state.submitted else "✅ Submitted — Thank you!"
if st.button(submit_label, disabled=disabled_submit, use_container_width=True):
    submit_lock = get_submit_lock()
    if not submit_lock.acquire(blocking=False):
        st.toast("Submission already in progress…", icon="⏳")
        st.stop()
    try:
        now2 = time.time()
        if (now2 - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC:
            st.session_state.status = "cooldown"
        else:
            ph = payload_hash(st.session_state.email, indicators, st.session_state.weights)
            if st.session_state.inflight_payload_hash == ph or st.session_state.last_payload_hash == ph:
                st.session_state.status = "duplicate"
            else:
                st.session_state.inflight_payload_hash = ph
                st.session_state.saving = True
                st.session_state.status = "saving"
                try:
                    save_to_sheet(
                        st.session_state.email.strip(),
                        st.session_state.weights,
                        st.session_state.session_id,
                        indicator_order=indicators
                    )
                    st.session_state.last_payload_hash = ph
                    st.session_state.submitted = True
                    st.session_state.status = "saved"
                    st.toast("Submitted. Thank you!", icon="✅")
                except Exception as e:
                    st.session_state.status = "error"
                    st.session_state.error_msg = str(e)
                finally:
                    st.session_state.saving = False
                    st.session_state.inflight_payload_hash = ""
                    st.session_state.last_submit_ts = time.time()
    finally:
        try:
            submit_lock.release()
        except Exception:
            pass

# ───────── STATUS ─────────
if st.session_state.get("saving", False):
    status_box.info("⏳ Saving your response… please wait. Do not refresh.")
else:
    if st.session_state.submitted:
        pass
    elif st.session_state.status == "duplicate":
        status_box.info("You’ve already saved this exact configuration.")
        st.session_state.status = "idle"
    elif st.session_state.status == "cooldown":
        status_box.info("Please wait a moment before submitting again.")
        st.session_state.status = "idle"
    elif st.session_state.status == "error":
        status_box.error(f"Error saving your response. {st.session_state.get('error_msg','')}")
        st.session_state.status = "idle"
    else:
        status_box.empty()
