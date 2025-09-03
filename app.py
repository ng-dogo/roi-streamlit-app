# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib
from typing import Dict, List, Optional
import random
import threading
from queue import Queue, Empty
import os, psutil
import csv
import traceback

from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError

# ───────── CONFIG ─────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18); }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width:860px}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:1rem 0}
.name{font-weight:600;margin:.35rem 0 .25rem}
.rowbox{padding:.45rem .5rem;border-radius:12px;border:1px solid var(--border);}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.45rem .9rem}
.stButton>button:hover{filter:brightness(0.95)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default;}
.center input[type=number]{text-align:center;font-weight:600}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--border);font-size:.9rem;color:var(--muted)}
.kpis{display:flex;gap:1rem;align-items:center}
.kpis .strong{font-weight:700}
.rank {width:100%;border-collapse:collapse;font-size:.95rem;}
.rank th, .rank td {padding:.35rem .5rem;border-bottom:1px solid var(--border);}
.rank th {font-weight:600;color:var(--muted);text-align:center;}
.rank td {text-align:left;}
.rank td:first-child, .rank td:last-child {text-align:center;}
.name.center {text-align:center;}
.small-note{font-size:.9rem;color:var(--muted);margin:.25rem 0 0}
.soft-divider{height:0;border-top:1px solid var(--border);margin:.5rem 0 1rem}
.hud {position: fixed; left: 12px; bottom: 12px; width: 65vw; max-width: 720px; background: rgba(255,255,255,.9);
  backdrop-filter: blur(6px); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.08);
  padding: .5rem .75rem; z-index: 9999;}
.dark .hud { background: rgba(28,28,28,.85) }
.hud-row{ display:flex; align-items:center; gap:.75rem }
.hud-mono{ font-variant-numeric: tabular-nums; font-weight:600 }
.hud-spacer{ flex:1 }
.hud-bar{ position:relative; height: 8px; background: rgba(127,127,127,.18); border-radius: 999px; overflow: hidden; width: 52%;}
.hud-fill{ position:absolute; left:0; top:0; bottom:0; background: var(--brand); width: 0%;}
@media (hover:hover){ .hud:hover{ box-shadow: 0 8px 26px rgba(0,0,0,.12) } }
@media (max-width: 480px){ .hud { bottom: 8px; padding: .45rem .6rem } }
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
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 1.5
EPS = 1e-6  # tolerancia numérica

# Batching / backoff (ajustables por env)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))                 # cantidad gatillo
FLUSH_INTERVAL = float(os.getenv("FLUSH_INTERVAL", "1.25"))     # seg: gatillo temporal
MAX_BATCH_PER_FLUSH = int(os.getenv("MAX_BATCH_PER_FLUSH", "120"))
BASE_DELAY = float(os.getenv("BACKOFF_BASE_DELAY", "0.5"))      # seg, backoff base
MAX_BACKOFF_SLEEP = float(os.getenv("MAX_BACKOFF_SLEEP", "5.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
FALLBACK_CSV = os.getenv("FALLBACK_CSV", "/tmp/bap_fallback.csv")  # opcional

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
if "thanks_ts" not in st.session_state:
    st.session_state.thanks_ts = 0.0

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
    out["avg_weight"] = (
        pd.to_numeric(out["avg_weight"], errors="coerce")
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )
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

def make_on_change(comp: str):
    def _cb():
        cur = float(st.session_state.weights[comp])
        new_val = float(st.session_state[f"num_{comp}"])
        delta = new_val - cur
        if delta > 0:
            allowed = min(delta, max(0.0, remaining_points(st.session_state.weights)))
            st.session_state.weights[comp] = min(1.0, cur + allowed)
        else:
            st.session_state.weights[comp] = max(0.0, new_val)
        st.session_state.weights[comp] = float(np.round(st.session_state.weights[comp] + 1e-9, 2))
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
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    return sh

def _ensure_header_once(ws, headers: List[str]) -> None:
    """Escribe encabezado solo si hace falta (tolerante a concurrencia)."""
    try:
        first_row = ws.row_values(1)
        if first_row and first_row[:len(headers)] == headers:
            return
    except Exception:
        pass
    try:
        ws.update("A1", [headers])
    except Exception:
        pass

class BatchAppender:
    """Acumula filas y hace append en lote con backoff + jitter. Compartido entre sesiones."""
    def __init__(
        self,
        worksheet,
        headers: List[str],
        batch_size: int = BATCH_SIZE,
        flush_interval: float = FLUSH_INTERVAL,
        max_batch_per_flush: int = MAX_BATCH_PER_FLUSH,
        base_delay: float = BASE_DELAY,
        max_backoff_sleep: float = MAX_BACKOFF_SLEEP,
        max_retries: int = MAX_RETRIES,
        fallback_csv: Optional[str] = FALLBACK_CSV,
    ):
        self.ws = worksheet
        self.headers = headers
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_batch_per_flush = max_batch_per_flush
        self.base_delay = base_delay
        self.max_backoff_sleep = max_backoff_sleep
        self.max_retries = max_retries
        self.fallback_csv = fallback_csv

        _ensure_header_once(self.ws, self.headers)

        self.q: Queue = Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, row: List):
        self.q.put(row)

    def stop(self):
        self._stop.set()
        try:
            self._worker.join(timeout=2.0)
        except Exception:
            pass

    def _flush_batch(self, batch: List[List]):
        # Backoff exponencial con jitter a nivel de lote
        for attempt in range(1, self.max_retries + 1):
            try:
                # Jitter pequeño para desincronizar ráfagas
                time.sleep(random.uniform(0.0, 0.25))
                # gspread worksheet.append_rows usa la Values API 'append'
                self.ws.append_rows(batch, value_input_option="RAW")
                return True
            except APIError:
                if attempt == self.max_retries:
                    break
                sleep_s = min(self.max_backoff_sleep, self.base_delay * (2 ** (attempt - 1)))
                time.sleep(sleep_s + random.uniform(0.0, 0.25))
            except Exception:
                # Errores no-API: también reintentar
                if attempt == self.max_retries:
                    break
                sleep_s = min(self.max_backoff_sleep, self.base_delay * (2 ** (attempt - 1)))
                time.sleep(sleep_s + random.uniform(0.0, 0.25))

        # Fallback local si no logramos escribir tras reintentos
        try:
            if self.fallback_csv:
                os.makedirs(os.path.dirname(self.fallback_csv), exist_ok=True)
                write_header = not os.path.exists(self.fallback_csv)
                with open(self.fallback_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(self.headers)
                    for r in batch:
                        w.writerow(r)
        except Exception:
            # como último recurso, al menos log
            traceback.print_exc()
        return False

    def _run(self):
        """Worker: agrupa por tiempo o tamaño y escribe."""
        buf: List[List] = []
        last_push = time.time()
        while not self._stop.is_set():
            timeout = max(0.05, self.flush_interval - (time.time() - last_push))
            try:
                row = self.q.get(timeout=timeout)
                buf.append(row)
                # Si llenamos tamaño de lote, disparamos flush
                if len(buf) >= self.batch_size:
                    batch = buf[:self.max_batch_per_flush]
                    buf = buf[self.max_batch_per_flush:]
                    self._flush_batch(batch)
                    last_push = time.time()
            except Empty:
                # Tiempo cumplido: si hay acumulado, flush
                if buf:
                    batch = buf[:self.max_batch_per_flush]
                    buf = buf[self.max_batch_per_flush:]
                    self._flush_batch(batch)
                    last_push = time.time()

@st.cache_resource(show_spinner=False)
def get_batch_appender(headers_tuple: tuple):
    ws = get_worksheet()
    return BatchAppender(ws, headers=list(headers_tuple))

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

# Email
st.session_state.email = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# Reset
right_align = st.columns([3,1])[1]
with right_align:
    if st.button("Reset to averages", disabled=st.session_state.saving):
        st.session_state.weights = dict(st.session_state.defaults)
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
    st.number_input(
        label="",
        key=f"num_{comp}",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        on_change=make_on_change(comp),
        disabled=st.session_state.saving
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ───────── LIVE RANKING ─────────
def render_ranking_html(weights: Dict[str, float]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
    rows = []
    rank = 1
    for name, pts in ordered:
        rows.append(f"<tr><td>{rank}</td><td>{name}</td><td class='r'>{float(pts):.2f}</td></tr>")
        rank += 1
    table_html = f"""
    <div class='rowbox'>
      <div class='name center'>Ranking</div>
      <table class="rank">
        <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
render_ranking_html(st.session_state.weights)

mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
st.caption(f"RAM usada por el proceso: {mem_mb:.1f} MB")

# ───────── HUD FLOTANTE ─────────
def render_floating_hud(used: float, rem: float, pct_used: float):
    pct = max(0.0, min(1.0, pct_used)) * 100.0
    st.markdown(f"""
    <div class="hud">
      <div class="hud-row">
        <div class="hud-mono">{used:.2f}/1.00</div>
        <div class="hud-spacer"></div>
        <div class="hud-bar">
          <div class="hud-fill" style="width:{pct:.2f}%"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

used = float(sum(st.session_state.weights.values()))
rem = remaining_points(st.session_state.weights)
pct_used = used / TOTAL_POINTS if TOTAL_POINTS else 0.0
render_floating_hud(used, rem, pct_used)

# ───────── WARNING SI NO SUMA 1 ─────────
used = float(sum(st.session_state.weights.values()))
rem = remaining_points(st.session_state.weights)
if abs(rem) > EPS:
    tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
    st.markdown(
        f"""
        <div style="
            margin:.75rem 0;
            padding:.6rem .9rem;
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
    or (abs(remaining_points(st.session_state.weights)) > EPS)  # debe sumar 1.00
    or st.session_state.saving
    or cooling
)

status_box = st.empty()

left, right = st.columns([1,1])
with left:
    submit_label = "Submit" if not st.session_state.submitted else "✅ Submitted — Thank you!"
    if st.button(submit_label, disabled=disabled_submit):
        now2 = time.time()
        if (now2 - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC:
            st.session_state.status = "cooldown"
        else:
            # Construcción robusta de row (redondeo a centésimas y suma=1.00 garantizada)
            indicators = list(st.session_state.weights.keys())
            # Normalizamos a centésimas manteniendo suma exacta 1.00
            w_norm = round_to_cents_preserve_total(st.session_state.weights)
            ph = payload_hash(st.session_state.email, indicators, w_norm)
            if st.session_state.inflight_payload_hash == ph or st.session_state.last_payload_hash == ph:
                st.session_state.status = "duplicate"
            else:
                st.session_state.inflight_payload_hash = ph
                st.session_state.saving = True
                st.session_state.status = "queueing"
                try:
                    headers = ["timestamp","email","session_id"] + indicators + ["total","submission_hash"]
                    # Inicializa (o reutiliza) appender compartido
                    appender = get_batch_appender(tuple(headers))
                    row = (
                        [dt.datetime.now().isoformat(), email_norm, st.session_state.session_id]
                        + [float(np.round(w_norm[k], 2)) for k in indicators]
                        + [float(np.round(sum(w_norm.values()), 2))]
                        + [ph[:12]]  # id corto visible
                    )
                    # Encola (respuesta inmediata al usuario)
                    appender.enqueue(row)

                    st.session_state.last_payload_hash = ph
                    st.session_state.submitted = True
                    st.session_state.status = "queued"
                    st.session_state.thanks_ts = time.time()
                    st.toast("✅ Received — your allocation has been queued for logging.", icon="✅")
                except Exception as e:
                    st.session_state.status = "error"
                    st.session_state.error_msg = str(e)
                finally:
                    st.session_state.saving = False
                    st.session_state.inflight_payload_hash = ""
                    st.session_state.last_submit_ts = time.time()

with right:
    pass

# ───────── STATUS ─────────
if st.session_state.get("saving", False):
    status_box.info("⏳ Saving your response…")
else:
    if st.session_state.submitted:
        # Botón ya indica éxito; nota opcional
        status_box.success("Your response was received and will appear in the log shortly.")
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
