# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib, random
from typing import Dict, List
from threading import Lock

from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="centered")

CSS = """
<style>
:root{ --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18); }
html, body, [class*="css"]{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.25}
.main .block-container{max-width:840px; padding-top:.6rem; padding-bottom:.6rem}
h1,h2,h3,h4{margin:.25rem 0 .35rem}
hr{border:none;border-top:1px solid rgba(127,127,127,.22);margin:.5rem 0}
.stMarkdown{margin:.15rem 0}
.name{font-weight:600;margin:.2rem 0 .1rem}
.rowbox{padding:.25rem .5rem;border-radius:10px;border:1px solid var(--border);margin:.25rem 0}
.center input[type=number]{text-align:center;font-weight:600}
.small-note{font-size:.9rem;color:var(--muted);margin:.1rem 0 0}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.35rem .75rem;min-height:0}
.stButton>button:hover{filter:brightness(.95)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default}
.badge{display:inline-block;padding:.1rem .4rem;border-radius:999px;border:1px solid var(--border);font-size:.85rem;color:var(--muted)}
.kpis{display:flex;gap:.6rem;align-items:center}
.kpis .strong{font-weight:700}

/* Inputs más bajos/compactos */
input[type="number"], input[type="text"], input[type="email"]{
  height:34px; padding:.1rem .5rem; font-size:1rem;
}

/* Tabla ranking minimalista y compacta */
.rank{width:100%; border-collapse:collapse; font-size:.94rem; margin:.2rem 0}
.rank th, .rank td{padding:.2rem .4rem; border-bottom:1px solid var(--border)}
.rank th{font-weight:600;color:var(--muted);text-align:center}
.rank td{text-align:left}
.rank td:first-child, .rank td:last-child{text-align:center}
.name.center{text-align:center}

/* HUD flotante (compacto) */
.hud{position:fixed;left:12px;bottom:10px;width:62vw;max-width:680px;background:rgba(255,255,255,.9);
  backdrop-filter:blur(6px);border:1px solid var(--border);border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,.08);
  padding:.35rem .6rem; z-index:9999}
.dark .hud{background:rgba(28,28,28,.82)}
.hud-row{display:flex;align-items:center;gap:.6rem}
.hud-mono{font-variant-numeric:tabular-nums;font-weight:600}
.hud-spacer{flex:1}
.hud-bar{position:relative;height:8px;background:rgba(127,127,127,.18);border-radius:999px;overflow:hidden;width:52%}
.hud-fill{position:absolute;left:0;top:0;bottom:0;background:var(--brand);width:0%}
@media (hover:hover){ .hud:hover{box-shadow:0 8px 24px rgba(0,0,0,.12)} }
@media (max-width:480px){ .hud{bottom:8px;padding:.3rem .5rem} }

/* Avisos compactos */
.alert{
  margin:.5rem 0; padding:.5rem .7rem; border-radius:8px; font-size:.95rem
}
.alert.red{ border:1px solid rgba(217,48,37,.35); background:rgba(217,48,37,.08); color:#b3261e }
.alert.green{ border:1px solid rgba(15,123,90,.35); background:rgba(15,123,90,.08); color:#0b6b59 }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────── CONSTANTES ─────────────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columnas: indicator, avg_weight
TOTAL_POINTS = 1.0  # la suma debe ser 1.00 para poder enviar
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
EPS = 1e-6  # tolerancia numérica para 'exactamente 1.00'

# ───────────────── STATE ─────────────────
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
for k, v in {
    "weights": {}, "defaults": {}, "email": "", "submitted": False, "_init_inputs": False,
    "saving": False, "last_submit_ts": 0.0, "last_payload_hash": "", "inflight_payload_hash": "",
    "status": "idle", "error_msg": ""
}.items():
    st.session_state.setdefault(k, v)

# ───────────────── HELPERS ─────────────────
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
    """Redondea a 0.01 manteniendo suma = 1.00 (en centésimas) si total>0; reparte si total<=0."""
    if not weights: return {}
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        cents_each = int(round(100 / n))
        cents = [cents_each]*n
        diff = 100 - sum(cents)
        for i in range(abs(diff)):
            cents[i % n] += 1 if diff > 0 else -1
        return {k: v/100.0 for k, v in zip(weights.keys(), cents)}
    scaled = {k: 100.0 * (v / total) for k, v in weights.items()}
    rounded = {k: int(np.floor(s + 0.5)) for k, s in scaled.items()}
    resid = {k: (scaled[k] - rounded[k]) for k in weights}
    diff = 100 - sum(rounded.values())
    if diff > 0:
        for k in sorted(weights.keys(), key=lambda k: resid[k], reverse=True)[:diff]: rounded[k] += 1
    elif diff < 0:
        for k in sorted(weights.keys(), key=lambda k: resid[k])[:abs(diff)]: rounded[k] -= 1
    return {k: rounded[k] / 100.0 for k in rounded}

def remaining_points(weights: Dict[str, float]) -> float:
    return float(TOTAL_POINTS - float(sum(weights.values())))

def make_on_change(comp: str):
    """Versión 'libre': permite pasarse de 1; sólo se clamp a [0,1] y redondea a 2 decimales."""
    def _cb():
        val = float(st.session_state.get(f"num_{comp}", 0.0))
        val = max(0.0, min(1.0, val))
        st.session_state.weights[comp] = float(np.round(val + 1e-9, 2))
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
        time.sleep(random.uniform(0.0, 0.35))  # jitter
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError as e:
            if attempt == attempts:
                raise RuntimeError(
                    "No pudimos guardar en Google Sheets tras varios intentos. "
                    "Por favor, intentá una sola vez más en unos segundos."
                ) from e
            time.sleep(min(4.0, base_delay * (2 ** (attempt - 1))))

# ───────────────── CARGA DE DEFAULTS ─────────────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    st.session_state.defaults = round_to_cents_preserve_total(defaults_raw)
    st.session_state.weights = dict(st.session_state.defaults)
    st.session_state._init_inputs = True
else:
    indicators = list(st.session_state.weights.keys())

# ───────────────── UI ─────────────────
st.title("RGI – Budget Allocation Points")

# Email (compacto)
st.session_state.email = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")

# Reset (alineado a la derecha, mismo bloque compacto)
right_align = st.columns([3,1])[1]
with right_align:
    if st.button("Reset to averages", disabled=st.session_state.saving):
        st.session_state.weights = dict(st.session_state.defaults)
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

# Inicializa números visibles con defaults una sola vez
if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

# Entradas (una debajo de la otra, más compactas)
for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
    st.number_input(
        label="", key=f"num_{comp}", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        label_visibility="collapsed", on_change=make_on_change(comp), disabled=st.session_state.saving
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────── RANKING LIVE ─────────────────
def render_ranking_html(weights: Dict[str, float]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
    rows = []
    for i, (name, pts) in enumerate(ordered, start=1):
        rows.append(f"<tr><td>{i}</td><td>{name}</td><td class='r'>{float(pts):.2f}</td></tr>")
    table_html = f"""
    <div class='rowbox'>
      <div class='name center'>Ranking</div>
      <table class="rank">
        <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
render_ranking_html(st.session_state.weights)

# ───────────────── HUD FLOTANTE (compacto) ─────────────────
def render_floating_hud(used: float, pct_used: float):
    pct = max(0.0, min(1.0, pct_used)) * 100.0  # clamp para barra
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
pct_used = used / TOTAL_POINTS if TOTAL_POINTS else 0.0
render_floating_hud(used, pct_used)

# ───────────────── WARNING / OK ─────────────────
rem = remaining_points(st.session_state.weights)
if abs(rem) > EPS:
    tip = f"Add {rem:.2f}" if rem > 0 else f"Remove {abs(rem):.2f}"
    st.markdown(f"<div class='alert red'>⚠️ The weights must sum to 1.00. {tip} to continue.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='alert green'>✅ Perfect — the weights sum to 1.00.</div>", unsafe_allow_html=True)

# ───────────────── SUBMIT ─────────────────
st.markdown("<hr/>", unsafe_allow_html=True)

email_norm = (st.session_state.email or "").strip()
ok_email = bool(EMAIL_RE.match(email_norm))
cooling = (time.time() - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC
disabled_submit = (
    (not ok_email)
    or st.session_state.submitted
    or (abs(remaining_points(st.session_state.weights)) > EPS)  # sólo si suma 1.00
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
            ph = payload_hash(email_norm, indicators, st.session_state.weights)
            if st.session_state.inflight_payload_hash in (ph, ) or st.session_state.last_payload_hash in (ph, ):
                st.session_state.status = "duplicate"
            else:
                st.session_state.inflight_payload_hash = ph
                st.session_state.saving = True
                st.session_state.status = "saving"
                try:
                    save_to_sheet(email_norm, st.session_state.weights, st.session_state.session_id, indicator_order=indicators)
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

# ───────────────── STATUS ─────────────────
if st.session_state.get("saving", False):
    status_box.info("⏳ Saving your response… please wait. Do not refresh.")
else:
    if st.session_state.submitted:
        pass  # el botón ya muestra el éxito
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
