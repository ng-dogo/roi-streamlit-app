# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib, random, psutil
from typing import Dict, List
from threading import Lock

from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18); }

html, body, [class*="css"]{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
.main .block-container{max-width:860px;padding-top:.35rem}
h1,h2,h3,h4{margin:.3rem 0 .35rem}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:.5rem 0}

/* Botones y number inputs compactos */
.stButton>button{
  background:var(--brand);color:#fff;border:none;border-radius:9px;
  padding:.25rem .5rem;min-height:32px;line-height:1;font-weight:600
}
.stButton>button:hover{filter:brightness(.95)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default}
.smallbtn .stButton>button{padding:.2rem .45rem;min-width:36px;border-radius:8px}

/* Number input estilizado y compacto */
.stNumberInput>div{padding:.05rem 0}
.stNumberInput input{
  text-align:center;font-weight:700;padding:.25rem .4rem;height:32px
}

/* Encabezados ligeros */
.label-head{font-size:.95rem;font-weight:600;color:var(--muted);margin:.15rem 0 .25rem}

/* — Tabla “visual” para Allocation (sin HTML table) — */
.row-sep{border-top:1px solid var(--border)}
.alloc-row{padding:.2rem 0 .3rem}

/* Forzar columnas en la misma fila también en móvil y reducir gaps */
[data-testid="stHorizontalBlock"]{gap:.35rem}
@media (max-width: 520px){
  [data-testid="stHorizontalBlock"]{display:flex;flex-wrap:nowrap!important}
  [data-testid="column"]{width:auto!important;min-width:0;flex:1 1 0!important}
}
/* En la celda de controles, fijamos un ancho mínimo que evita scroll horizontal */
.controls-wrap{max-width:220px;margin-left:auto}
@media (max-width: 520px){
  .controls-wrap{max-width:200px}
}
.controls-grid [data-testid="stHorizontalBlock"]{gap:.25rem}

/* Ranking minimalista */
.rank {width:100%;border-collapse:collapse;font-size:.95rem}
.rank th, .rank td {padding:.35rem .5rem;border-bottom:1px solid var(--border)}
.rank th {font-weight:600;color:var(--muted);text-align:center}
.rank td {text-align:left}
.rank td:first-child, .rank th:first-child {text-align:center;width:52px}
.rank td:last-child, .rank th:last-child {text-align:center;width:92px}
.table-title{font-weight:700;text-align:center;margin:.25rem 0 .35rem}

/* Alertas */
.alert{
  margin:.5rem 0;padding:.6rem .9rem;border-radius:10px;font-size:.95rem
}
.alert.red{border:1px solid rgba(217,48,37,.35);background:rgba(217,48,37,.08);color:#b3261e}
.alert.green{border:1px solid rgba(12,120,90,.35);background:rgba(12,120,90,.08);color:#0b6b59}

/* Reset botón a la derecha del email */
.topline{display:flex;gap:.6rem;align-items:flex-end}
.topline .flexsp{flex:1}

/* Pequeño pie de página */
.small-note{font-size:.85rem;color:var(--muted)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────── CONSTANTES ─────────────────────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 1.0
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
EPS = 1e-6

# ───────────────────────── STATE ─────────────────────────
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
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""

# ───────────────────────── HELPERS ─────────────────────────
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

def make_on_change(comp: str):
    """Permite pasar de 1.00 en la suma total; sólo acota por indicador [0,1] y redondea a 2 decimales."""
    def _cb():
        val = float(st.session_state.get(f"num_{comp}", 0.0) or 0.0)
        val = max(0.0, min(1.0, val))
        val = float(np.round(val + 1e-9, 2))
        st.session_state.weights[comp] = val
        st.session_state[f"num_{comp}"] = val
    return _cb

def payload_hash(email: str, indicators: List[str], weights: Dict[str, float]) -> str:
    tpl = (email.strip().lower(), tuple(indicators), tuple(float(weights[k]) for k in indicators))
    return hashlib.sha256(repr(tpl).encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def get_worksheet():
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
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
        time.sleep(random.uniform(0.0, 0.35))  # jitter
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError as e:
            if attempt == attempts:
                raise RuntimeError(
                    "No pudimos guardar en Google Sheets tras varios intentos. "
                    "Esperá unos segundos y hacé un único reintento."
                ) from e
            time.sleep(min(4.0, base_delay * (2 ** (attempt - 1))))

# ───────────────────────── CARGA DEFAULTS ─────────────────────────
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

# ───────────────────────── UI ─────────────────────────
st.title("RGI – Budget Allocation Points")

# Email + Reset (alineados en una sola línea)
st.markdown('<div class="topline">', unsafe_allow_html=True)
email_val = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")
st.session_state.email = email_val
st.markdown('<div class="flexsp"></div>', unsafe_allow_html=True)
if st.button("Reset to averages", disabled=st.session_state.saving):
    st.session_state.weights = dict(st.session_state.defaults)
    for comp in st.session_state.weights:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── Warning arriba del Allocation
used = float(sum(st.session_state.weights.values()))
rem = remaining_points(st.session_state.weights)
if abs(rem) <= EPS:
    st.markdown(
        f"<div class='alert green'>✅ Sum of weights: {used:.2f}/1.00 — Ready to submit.</div>",
        unsafe_allow_html=True
    )
else:
    tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
    st.markdown(
        f"<div class='alert red'>⚠️ The weights must sum to 1.00. <strong>{tip}</strong> to continue.</div>",
        unsafe_allow_html=True
    )

# ── Allocation (compacto, indicador a la izquierda y controles a la derecha)
st.subheader("Allocation")
st.markdown('<div class="label-head row-sep">Indicator</div>', unsafe_allow_html=True)
st.markdown('<div class="label-head" style="text-align:right">Weight</div>', unsafe_allow_html=True)

if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

for i, comp in enumerate(indicators):
    # Fila con dos columnas: nombre | controles
    left, right = st.columns([1.35, 1.0])
    with left:
        st.markdown(f"<div class='alloc-row'><strong>{comp}</strong></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='controls-wrap'>", unsafe_allow_html=True)
        minus, numc, plus = st.columns([1, 3.2, 1])
        with minus:
            if st.button("−", key=f"dec_{comp}", help="−0.01", disabled=st.session_state.saving):
                v = float(st.session_state.weights.get(comp, 0.0))
                v = max(0.0, v - 0.01)
                v = float(np.round(v + 1e-9, 2))
                st.session_state.weights[comp] = v
                st.session_state[f"num_{comp}"] = v
                st.rerun()
        with numc:
            st.number_input(
                label="",
                key=f"num_{comp}",
                min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                label_visibility="collapsed",
                on_change=make_on_change(comp),
                disabled=st.session_state.saving
            )
        with plus:
            if st.button("+", key=f"inc_{comp}", help="+0.01", disabled=st.session_state.saving):
                v = float(st.session_state.weights.get(comp, 0.0))
                v = min(1.0, v + 0.01)
                v = float(np.round(v + 1e-9, 2))
                st.session_state.weights[comp] = v
                st.session_state[f"num_{comp}"] = v
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if i < len(indicators) - 1:
        st.markdown('<div class="row-sep"></div>', unsafe_allow_html=True)

# ── Ranking (entre Allocation y Submit)
def render_ranking_html(weights: Dict[str, float]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
    rows = []
    for rank, (name, pts) in enumerate(ordered, start=1):
        rows.append(f"<tr><td>{rank}</td><td>{name}</td><td>{float(pts):.2f}</td></tr>")
    table_html = f"""
    <div class='row-sep' style="margin-top:.6rem"></div>
    <div class='table-title'>Ranking</div>
    <div class='rowbox'>
      <table class="rank">
        <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

render_ranking_html(st.session_state.weights)

# ── Submit
st.markdown("<hr/>", unsafe_allow_html=True)
email_norm = (st.session_state.email or "").strip()
ok_email = bool(EMAIL_RE.match(email_norm))
now = time.time()
cooling = (now - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC

disabled_submit = (
    (not ok_email)
    or st.session_state.submitted
    or (abs(remaining_points(st.session_state.weights)) > EPS)  # debe sumar 1.00
    or st.session_state.saving
    or cooling
)

status_box = st.empty()

submit_label = "Submit" if not st.session_state.submitted else "✅ Submitted — Thank you!"
if st.button(submit_label, disabled=disabled_submit):
    submit_lock = get_submit_lock()
    if not submit_lock.acquire(blocking=False):
        st.toast("Submission already in progress…", icon="⏳")
        st.stop()
    try:
        now2 = time.time()
        if (now2 - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC:
            st.session_state.status = "cooldown"
        else:
            ph = hashlib.sha256(
                repr((email_norm.lower(), tuple(indicators),
                      tuple(float(st.session_state.weights[k]) for k in indicators))).encode()
            ).hexdigest()
            if st.session_state.inflight_payload_hash == ph or st.session_state.last_payload_hash == ph:
                st.session_state.status = "duplicate"
            else:
                st.session_state.inflight_payload_hash = ph
                st.session_state.saving = True
                st.session_state.status = "saving"
                try:
                    save_to_sheet(email_norm, st.session_state.weights, st.session_state.session_id, indicators)
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
        try: submit_lock.release()
        except Exception: pass

# ── Estados / Mensajes
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

# ── Pequeño dato de RAM (ayuda para ensayos de concurrencia)
mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
st.caption(f"RAM used by process: {mem_mb:.1f} MB")
