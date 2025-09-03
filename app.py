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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18); }

html, body, [class*="css"]{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container{ max-width: 860px; padding-top: 0.5rem; }

/* Compacción vertical suave */
h1, h2, h3, h4, h5, h6{ margin: .25rem 0 .35rem; }
hr{ border:none; border-top:1px solid var(--border); margin:.5rem 0; }
.block-container > div{ margin-top:.35rem; margin-bottom:.35rem; }

/* Controles */
.stTextInput > div > div input{ padding:.35rem .5rem; }
.stNumberInput div[data-baseweb="input"]{ min-height: 34px; }
.stNumberInput input{ text-align:center; font-weight:600; padding:.3rem .4rem; }

/* Botones */
.stButton>button{
  background:var(--brand); color:#fff; border:none; border-radius:10px;
  padding:.4rem .9rem; line-height:1; min-height:34px;
}
.stButton>button:hover{ filter:brightness(0.95) }
.stButton>button:disabled{ background:#0b6b59; color:#fff; opacity:1; cursor:default; }

/* Badges / notas */
.small-note{ font-size:.9rem; color:var(--muted); margin:.25rem 0 0 }

/* Filas de asignación compactas */
.alloc-row{ padding:.25rem .4rem; border-radius:10px; border:1px solid var(--border); }
.alloc-label{ font-weight:600; padding:.2rem 0; }
.centered-input input{ text-align:center; }

/* Ranking minimalista */
.rank-wrap{ padding:.55rem .6rem; border:1px solid var(--border); border-radius:10px; }
.rank-title{ font-weight:600; text-align:center; margin-bottom:.25rem; }
.rank{ width:100%; border-collapse:collapse; font-size:.95rem; }
.rank th, .rank td{ padding:.35rem .4rem; border-bottom:1px solid var(--border);}
.rank th{ font-weight:600; color:var(--muted); text-align:center; }
.rank td:first-child, .rank td:last-child{ text-align:center; }
.rank td:nth-child(2){ text-align:left; }

/* Alertas */
.alert{
  margin:.4rem 0; padding:.6rem .8rem; border-radius:8px; font-size:.95rem;
  border:1px solid; display:flex; align-items:center; gap:.5rem;
}
.alert.red{  border-color:rgba(217,48,37,.35); background:rgba(217,48,37,.08); color:#b3261e; }
.alert.green{border-color:rgba(12,131,61,.35); background:rgba(12,131,61,.08); color:#0b6b59; }

/* HUD flotante inferior */
.hud {
  position: fixed; left: 12px; bottom: 12px; width: 65vw; max-width: 720px;
  background: rgba(255,255,255,.9); backdrop-filter: blur(6px);
  border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.08);
  padding: .45rem .6rem; z-index: 9999;
}
.dark .hud { background: rgba(28,28,28,.85) }
.hud-row{ display:flex; align-items:center; gap:.6rem }
.hud-mono{ font-variant-numeric: tabular-nums; font-weight:700 }
.hud-spacer{ flex:1 }
.hud-bar{ position:relative; height:8px; background: rgba(127,127,127,.18); border-radius:999px; overflow:hidden; width:52%;}
.hud-fill{ position:absolute; left:0; top:0; bottom:0; background: var(--brand); width:0%; }

/* Forzar 2 columnas también en pantallas angostas cuando se usa el layout 2× */
#alloc-2col [data-testid="column"]{ flex: 0 0 50% !important; width:50% !important; }
@media (max-width: 420px){
  /* Hacemos el input un poco más ancho en móviles */
  .stNumberInput input{ font-size:.95rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 1.0  # pesos deben sumar 1.00
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
EPS = 1e-6  # tolerancia numérica

# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────
ss = st.session_state
if "session_id" not in ss: ss.session_id = str(uuid.uuid4())
if "weights"    not in ss: ss.weights = {}            # Dict[str, float]
if "defaults"   not in ss: ss.defaults = {}           # Dict[str, float]
if "email"      not in ss: ss.email = ""
if "submitted"  not in ss: ss.submitted = False
if "_init"      not in ss: ss._init = False
if "saving"     not in ss: ss.saving = False
if "last_submit_ts" not in ss: ss.last_submit_ts = 0.0
if "last_payload_hash" not in ss: ss.last_payload_hash = ""
if "inflight_payload_hash" not in ss: ss.inflight_payload_hash = ""
if "status"     not in ss: ss.status = "idle"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
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
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").clip(0.0, 1.0).fillna(0.0)
    return out

def round_to_cents_preserve_total(weights: Dict[str, float]) -> Dict[str, float]:
    """Redondea a 0.01 manteniendo suma exacta = 1.00 (en centésimas)."""
    if not weights: return {}
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
        for k in order[:diff]: rounded[k] += 1
    elif diff < 0:
        order = sorted(weights.keys(), key=lambda k: resid[k])
        for k in order[:abs(diff)]: rounded[k] -= 1
    return {k: rounded[k] / 100.0 for k in rounded}

def remaining_points(weights: Dict[str, float]) -> float:
    return float(TOTAL_POINTS - float(sum(weights.values())))

def make_on_change(comp: str):
    """Permite pasar(se) de 1.00: no limitamos en el input; sólo bloqueamos el submit si != 1."""
    def _cb():
        new_val = float(ss.get(f"num_{comp}", 0.0))
        new_val = float(np.round(np.clip(new_val, 0.0, 1.0) + 1e-9, 2))
        ss.weights[comp] = new_val
        ss[f"num_{comp}"] = new_val
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
        if first_row and first_row[:len(headers)] == headers: return
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
    base_delay, attempts = 0.4, 5
    for attempt in range(1, attempts + 1):
        time.sleep(random.uniform(0.0, 0.35))  # jitter anti-ráfagas
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

def render_ranking_html(weights: Dict[str, float]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
    rows = []
    for i, (name, pts) in enumerate(ordered, start=1):
        rows.append(f"<tr><td>{i}</td><td>{name}</td><td>{float(pts):.2f}</td></tr>")
    html = f"""
    <div class='rank-wrap'>
      <div class='rank-title'>Ranking</div>
      <table class="rank">
        <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_floating_hud(used: float):
    pct = max(0.0, min(1.0, used / TOTAL_POINTS if TOTAL_POINTS else 0.0)) * 100.0
    st.markdown(f"""
    <div class="hud">
      <div class="hud-row">
        <div class="hud-mono">{used:.2f}/1.00</div>
        <div class="hud-spacer"></div>
        <div class="hud-bar"><div class="hud-fill" style="width:{pct:.2f}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
if not ss.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    defaults_cents = round_to_cents_preserve_total(defaults_raw)
    ss.defaults = defaults_cents
    ss.weights = dict(defaults_cents)
    ss._init = True
else:
    indicators = list(ss.weights.keys())

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("RGI – Budget Allocation Points")

# Email
ss.email = st.text_input("Email", value=ss.email, placeholder="name@example.org")

# Reset
right_align = st.columns([3,1])[1]
with right_align:
    if st.button("Reset to averages", disabled=ss.saving):
        ss.weights = dict(ss.defaults)
        for comp in ss.weights:
            ss[f"num_{comp}"] = float(ss.weights[comp])
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# Inicializar inputs en primer render
if ss.get("_init"):
    for comp in indicators:
        ss[f"num_{comp}"] = float(ss.weights[comp])
    ss._init = False

# HUD
used = float(sum(ss.weights.values()))
render_floating_hud(used)

# ── ALERTA arriba del Allocation ──
rem = remaining_points(ss.weights)
if abs(rem) <= EPS:
    st.markdown("<div class='alert green'>✅ The weights sum to 1.00. You can submit.</div>", unsafe_allow_html=True)
else:
    tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
    st.markdown(f"<div class='alert red'>⚠️ The weights must sum to 1.00. {tip} to continue.</div>", unsafe_allow_html=True)

# ── Allocation (compacto) ──
st.subheader("Allocation")

layout = st.radio("Layout", ["Two columns (2×)", "Single column (label + input en la misma fila)"],
                  horizontal=True, index=0)

def render_input_row(container, comp: str):
    lc, rc = container.columns([0.62, 0.38], gap="small")
    lc.markdown(f"<div class='alloc-label'>{comp}</div>", unsafe_allow_html=True)
    with rc:
        st.number_input(
            label="",
            key=f"num_{comp}",
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            label_visibility="collapsed", on_change=make_on_change(comp),
            disabled=ss.saving
        )

if layout.startswith("Two"):
    st.markdown("<div id='alloc-2col'>", unsafe_allow_html=True)
    colL, colR = st.columns(2, gap="small")
    # Repartimos alternando para equilibrar alturas
    for idx, comp in enumerate(indicators):
        container = colL if (idx % 2 == 0) else colR
        with container:
            st.markdown("<div class='alloc-row'>", unsafe_allow_html=True)
            render_input_row(container, comp)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    for comp in indicators:
        st.markdown("<div class='alloc-row'>", unsafe_allow_html=True)
        render_input_row(st, comp)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Ranking entre Allocation y Submit ──
st.markdown("<hr/>", unsafe_allow_html=True)
render_ranking_html(ss.weights)

# Memoria (útil para monitoreo rápido)
mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
st.caption(f"RAM usada por el proceso: {mem_mb:.1f} MB")

# ── SUBMIT ──
st.markdown("<hr/>", unsafe_allow_html=True)
email_raw = ss.email or ""
email_norm = email_raw.strip()
ok_email = bool(EMAIL_RE.match(email_norm))
now = time.time()
cooling = (now - ss.last_submit_ts) < SUBMISSION_COOLDOWN_SEC

disabled_submit = (
    (not ok_email)
    or ss.submitted
    or (abs(remaining_points(ss.weights)) > EPS)  # sólo permitir submit si suma 1.00 exacto
    or ss.saving
    or cooling
)

status_box = st.empty()
submit_label = "Submit" if not ss.submitted else "✅ Submitted — Thank you!"
if st.button(submit_label, disabled=disabled_submit):
    submit_lock = get_submit_lock()
    if not submit_lock.acquire(blocking=False):
        st.toast("Submission already in progress…", icon="⏳")
        st.stop()
    try:
        now2 = time.time()
        if (now2 - ss.last_submit_ts) < SUBMISSION_COOLDOWN_SEC:
            ss.status = "cooldown"
        else:
            ph = payload_hash(ss.email, indicators, ss.weights)
            if ss.inflight_payload_hash == ph or ss.last_payload_hash == ph:
                ss.status = "duplicate"
            else:
                ss.inflight_payload_hash = ph
                ss.saving = True
                ss.status = "saving"
                try:
                    save_to_sheet(
                        ss.email.strip(),
                        ss.weights,
                        ss.session_id,
                        indicator_order=indicators
                    )
                    ss.last_payload_hash = ph
                    ss.submitted = True
                    ss.status = "saved"
                    st.toast("Submitted. Thank you!", icon="✅")
                except Exception as e:
                    ss.status = "error"
                    ss.error_msg = str(e)
                finally:
                    ss.saving = False
                    ss.inflight_payload_hash = ""
                    ss.last_submit_ts = time.time()
    finally:
        try: submit_lock.release()
        except Exception: pass

# ── STATUS ──
if ss.get("saving", False):
    status_box.info("⏳ Saving your response… please wait. Do not refresh.")
else:
    if ss.submitted:
        pass  # el botón ya muestra el estado
    elif ss.status == "duplicate":
        status_box.info("You’ve already saved this exact configuration.")
        ss.status = "idle"
    elif ss.status == "cooldown":
        status_box.info("Please wait a moment before submitting again.")
        ss.status = "idle"
    elif ss.status == "error":
        status_box.error(f"Error saving your response. {ss.get('error_msg','')}")
        ss.status = "idle"
    else:
        status_box.empty()
