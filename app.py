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

/* Fuente y contenedor más compacto */
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.main .block-container{max-width:860px; padding-top: 0.6rem;}

/* Evitar scroll horizontal en cualquier caso */
html, body{ overflow-x: hidden; }

/* Divisores y títulos con menos margen vertical */
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:.5rem 0}
h1, h2, h3{ margin: .25rem 0 .5rem; }
.stSubtitle, .stHeader{ margin: .25rem 0 .5rem !important; }

/* Botón */
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;
  padding:.45rem .9rem}
.stButton>button:hover{filter:brightness(0.95)}
/* Verde oscuro cuando ya se envió */
.stButton>button:disabled{
  background:#0b6b59;
  color:#fff; 
  opacity:1;
  cursor:default;
}

/* Campos numéricos centrados y compactos */
.center input[type=number]{text-align:center;font-weight:600}

/* Badges y kpis (si se usan) */
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--border);
  font-size:.9rem;color:var(--muted)}
.kpis{display:flex;gap:1rem;align-items:center}
.kpis .strong{font-weight:700}

/* Caja genérica (se sigue usando en Ranking) */
.rowbox{padding:.45rem .5rem;border-radius:12px;border:1px solid var(--border);}

/* Tabla ranking más cerrada */
.rank { width:100%; border-collapse:collapse; font-size:.95rem; }
.rank th, .rank td { padding:.25rem .4rem; border-bottom:1px solid var(--border); }
.rank th { font-weight:600; color:var(--muted); text-align:center; }
.rank td { text-align:left; }
.rank td:first-child, .rank td:last-child { text-align:center; }
.name.center { text-align:center; }

/* Notas pequeñas */
.small-note{font-size:.9rem;color:var(--muted);margin:.15rem 0 0}
.soft-divider{height:0;border-top:1px solid var(--border);margin:.35rem 0 .75rem}

/* — HUD flotante inferior — */
.hud {
  position: fixed;
  left: 12px;
  bottom: 12px;
  width: 65vw;
  max-width: 720px;
  background: rgba(255,255,255,.9);
  backdrop-filter: blur(6px);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0,0,0,.08);
  padding: .5rem .75rem;
  z-index: 9999;
}
.dark .hud { background: rgba(28,28,28,.85) }
.hud-row{ display:flex; align-items:center; gap:.75rem }
.hud-mono{ font-variant-numeric: tabular-nums; font-weight:600 }
.hud-spacer{ flex:1 }
.hud-bar{
  position:relative;
  height: 8px;
  background: rgba(127,127,127,.18);
  border-radius: 999px;
  overflow: hidden;
  width: 52%;
}
.hud-fill{
  position:absolute; left:0; top:0; bottom:0;
  background: var(--brand);
  width: 0%;
}
@media (hover:hover){ .hud:hover{ box-shadow: 0 8px 26px rgba(0,0,0,.12) } }
@media (max-width: 480px){ .hud { bottom: 8px; padding: .45rem .6rem } }
@media (prefers-color-scheme: dark){
  .hud{ background: rgba(18,18,18,.85); border-color: rgba(255,255,255,.12); }
  .hud-mono{ color: rgba(255,255,255,.92); }
  .hud-bar{ background: rgba(255,255,255,.15); }
}

/* ============ ALLOCATION ULTRA-COMPACT ============ */
/* Contenedor responsivo en grilla: jamás produce scroll horizontal */
.alloc{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: .5rem .75rem;
  align-items: start;
}

/* Tarjeta/row por indicador: nombre + input en una sola línea */
.alloc-item{
  display: grid;
  grid-template-columns: 1fr 96px;  /* etiqueta flexible + input fijo */
  align-items: center;
  gap: .25rem .5rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .35rem .5rem;
  background: rgba(127,127,127,.04);
}

/* Nombre del indicador */
.alloc-name{
  font-weight:600; 
  margin: 0; 
  line-height:1.2;
  word-break: break-word;
  color: rgba(0,0,0,.88);
}

/* El widget numérico sin márgenes extra y más bajo */
.alloc-item [data-testid="stNumberInput"]{ margin: 0 !important; }
.alloc-item [data-testid="stNumberInput"] > label{ display:none !important; }
.alloc-item input[type=number]{
  text-align:center;
  font-weight:600;
  padding:.25rem .4rem;
  height: 2.0rem;
}

/* En pantallas muy estrechas, input ocupa el ancho disponible sin desbordar */
@media (max-width: 360px){
  .alloc-item{ grid-template-columns: 1fr 88px; }
}

/* Ajustes sutiles en mensajes/alertas para ahorrar vertical */
.stAlert{ padding: .5rem .75rem; }
.stAlert p{ margin: 0; }

/* Ajuste de espacios en campos de texto (email) */
[data-testid="stTextInput"]{ margin-bottom: .35rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 1.0  # pesos suman 1.00
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
THANKS_VISIBLE_SEC = 3.0
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
if "thanks_expire" not in st.session_state:
    st.session_state.thanks_expire = 0.0

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
    # CSV ya en [0,1]; limpiamos y acotamos por las dudas
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

# *** CAMBIO CLAVE: callback sin tope global (permite pasar de 1) ***
def make_on_change(comp: str):
    def _cb():
        new_val = float(st.session_state.get(f"num_{comp}", 0.0))
        # clamp individual en [0,1] y redondeo a 2 decimales
        new_val = float(np.round(min(max(new_val, 0.0), 1.0) + 1e-9, 2))
        st.session_state.weights[comp] = new_val
        st.session_state[f"num_{comp}"] = new_val
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
        time.sleep(random.uniform(0.0, 0.35))  # jitter
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError as e:
            if attempt == attempts:
                raise RuntimeError(
                    "No pudimos guardar en Google Sheets tras varios intentos. "
                    "Por favor, esperá unos segundos y hacé un único reintento."
                ) from e
            sleep_s = min(4.0, base_delay * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

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

# ───────── AVISO (rojo/verde) SEGÚN SUMA ─────────
used = float(sum(st.session_state.weights.values()))
rem = remaining_points(st.session_state.weights)

if abs(rem) > EPS:
    tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
    st.markdown(
        f"""
        <div style="
            margin:.5rem 0;
            padding:.55rem .8rem;
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
else:
    st.markdown(
        """
        <div style="
            margin:.5rem 0;
            padding:.55rem .8rem;
            border:1px solid rgba(16,127,70,.35);
            background:rgba(16,127,70,.08);
            border-radius:8px;
            font-size:.95rem;
            color:#0E7C66;">
            ✅ The weights sum to <b>1.00</b>. You can submit.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Inicializar inputs una sola vez
if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

# ============ NUEVO: grilla compacta de asignación ============
st.markdown("<div class='alloc'>", unsafe_allow_html=True)
for comp in indicators:
    st.markdown("<div class='alloc-item'>", unsafe_allow_html=True)
    st.markdown(f"<div class='alloc-name'>{comp}</div>", unsafe_allow_html=True)
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
st.markdown("</div>", unsafe_allow_html=True)
# ===============================================================

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
        <div class="hud-bar"><div class="hud-fill" style="width:{pct:.2f}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

used = float(sum(st.session_state.weights.values()))
rem = remaining_points(st.session_state.weights)
pct_used = used / TOTAL_POINTS if TOTAL_POINTS else 0.0
render_floating_hud(used, rem, pct_used)

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

left, right = st.columns([1,1])
with left:
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

with right:
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
