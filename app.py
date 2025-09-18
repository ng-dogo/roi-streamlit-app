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
.main .block-container{max-width:860px}
hr{border:none;border-top:1px solid rgba(127,127,127,.25);margin:1rem 0}
.name{font-weight:600;margin:.35rem 0 .25rem}
.rowbox{padding:.45rem .5rem;border-radius:12px;border:1px solid var(--border);}
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.45rem .9rem}
.stButton>button:hover{filter:brightness(0.95)}
/* Estilo verde oscuro cuando ya se envió */
.stButton>button:disabled{
  background:#0b6b59;
  color:#fff; 
  opacity:1;
  cursor:default;
}
.center input[type=number]{text-align:center;font-weight:600}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--border);font-size:.9rem;color:var(--muted)}
.kpis{display:flex;gap:1rem;align-items:center}
.kpis .strong{font-weight:700}

/* Tabla ranking minimalista (no widgets) */
.rank { width:100%; border-collapse:collapse; font-size:.95rem; }
.rank th, .rank td { padding:.35rem .5rem; border-bottom:1px solid var(--border); }
.rank th { font-weight:600; color:var(--muted); text-align:center; }
.rank td { text-align:left; }
.rank td:first-child, .rank td:last-child { text-align:center; }
.name.center { text-align:center; }

.small-note{font-size:.9rem;color:var(--muted);margin:.25rem 0 0}
.soft-divider{height:0;border-top:1px solid var(--border);margin:.5rem 0 1rem}

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

/* Compact table custom styles */
.mini { font-size: .92rem; }
.mini .head { color: var(--muted); font-weight:600; border-bottom:1px solid var(--border); padding:.25rem .4rem; }
.mini .row  { border-bottom:1px solid var(--border); padding:.25rem .4rem; }
.mini .name { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mini .wgt  { text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }
.mini .btn  { text-align:right; }
.mini .stButton>button{ padding:.15rem .4rem; border-radius:8px; min-height:auto }
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
if "indicator_order" not in st.session_state:
    st.session_state.indicator_order: List[str] = []

# ───────── HELPERS ─────────
@st.cache_data(ttl=300)
def load_defaults_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el archivo de defaults en: {path}. "
            "Definí la variable de entorno RGI_DEFAULTS_CSV o subí el archivo."
        )

    ext = os.path.splitext(path)[1].lower()

    # 1) Excel (exportado desde Sheets/Excel)
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # 2) CSV: probamos codificaciones típicas y dejamos que infiera el delimitador
        enc_try = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
        last_err = None
        df = None
        for enc in enc_try:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    encoding_errors="replace",
                    sep=None,                   # infiere coma/; /tab
                    engine="python"             # requerido para sep=None
                )
                break
            except Exception as e:
                last_err = e
                continue
        if df is None:
            raise RuntimeError(
                f"No pude leer el archivo CSV con las codificaciones {enc_try}. "
                f"Último error: {last_err}"
            )

    # Normalización de columnas esperadas
    cols = [c.strip().lower() for c in df.columns]
    if not cols:
        raise ValueError("El archivo de defaults no tiene columnas.")

    df.columns = cols
    name_col = "indicator" if "indicator" in cols else cols[0]
    weight_col = "avg_weight" if "avg_weight" in cols else (cols[1] if len(cols) > 1 else None)

    if weight_col is None:
        raise ValueError("No se encontró columna de pesos. Esperaba 'indicator' y 'avg_weight'.")

    out = df[[name_col, weight_col]].copy()
    out.columns = ["indicator", "avg_weight"]
    out["indicator"] = out["indicator"].astype(str).str.strip()
    out["avg_weight"] = (
        pd.to_numeric(out["avg_weight"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    )
    return out

# --- util: redondear a centésimas preservando suma = 1.00 ---
def round_to_cents_preserve_total(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        cents_each = int(round(100 / n))
        cents = [cents_each] * n
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
        time.sleep(random.uniform(0.0, 0.35))
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
# ───────── LOAD DEFAULTS ─────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)

    # Orden original del CSV (lo usamos para guardar en Sheets)
    indicator_order = df["indicator"].tolist()
    st.session_state.indicator_order = indicator_order

    # Pesos EXACTOS tal cual vienen del CSV → para el ranking fijo
    defaults_csv = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    st.session_state.defaults_csv = dict(defaults_csv)

    # Pesos "editables" (parte interactiva) → redondeados a centésimas y suma = 1.00
    defaults_cents = round_to_cents_preserve_total(defaults_csv)
    st.session_state.defaults = dict(defaults_cents)
    st.session_state.weights = dict(defaults_cents)

    st.session_state._init_inputs = True
else:
    # Fallbacks por si venís de una sesión vieja
    if "indicator_order" not in st.session_state or not st.session_state.indicator_order:
        st.session_state.indicator_order = list(st.session_state.weights.keys())
    if "defaults_csv" not in st.session_state or not st.session_state.defaults_csv:
        st.session_state.defaults_csv = dict(st.session_state.defaults)

# Alias local para usar en el guardado
indicators = st.session_state.indicator_order



# ───────── UI ─────────
st.title("RGI – Budget Allocation Points")

# Email
st.session_state.email = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# (Se eliminó el botón "Reset to averages" como pediste)

# ───────── AVISO (rojo/verde) SEGÚN SUMA ─────────
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
else:
    st.markdown(
        """
        <div style="
            margin:.75rem 0;
            padding:.6rem .9rem;
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
#left, mid, right = st.columns([1, 2, 1])
#with mid:
    #st.subheader("Allocation")

# ───────── COMPACT TABLE WITH +/- ─────────
STEP = 0.01
def _clamp01(x: float) -> float:
    return float(np.round(min(max(x, 0.0), 1.0) + 1e-9, 2))

def _adjust_others(weights: Dict[str, float], target: str, delta: float) -> Dict[str, float]:
    w = dict(weights)
    total_before = sum(w.values())
    w[target] = _clamp01(w[target] + delta)
    diff = total_before - sum(w.values())
    if abs(diff) < 1e-9:
        return w
    others = [k for k in w.keys() if k != target]
    if not others:
        return w
    pool = sum(w[k] for k in others)
    if pool > 1e-12:
        for k in others:
            prop = (w[k] / pool) if pool > 0 else (1.0 / len(others))
            adj = diff * prop
            newv = _clamp01(w[k] + adj)
            diff -= (newv - w[k])
            w[k] = newv
    if abs(diff) >= (STEP/2):
        if diff > 0:
            order = sorted(others, key=lambda k: (1.0 - w[k]), reverse=True)
            step_sign = +STEP
            can_move = lambda k: w[k] < 1.0 - 1e-9
        else:
            order = sorted(others, key=lambda k: w[k], reverse=True)
            step_sign = -STEP
            can_move = lambda k: w[k] > 0.0 + 1e-9
        safety = 20000; i = 0
        while abs(diff) >= (STEP/2) and i < safety:
            moved = False
            for k in order:
                if abs(diff) < (STEP/2): break
                if not can_move(k): continue
                newv = _clamp01(w[k] + step_sign)
                if abs(newv - w[k]) >= 1e-12:
                    w[k] = newv
                    diff = total_before - sum(w.values())
                    moved = True
            if not moved: break
            i += 1
    return w

def render_compact_table(weights: Dict[str, float]) -> None:
    # Encabezados con mismas proporciones que las filas
    h1, h2, h3 = st.columns([6, 2, 3])
    with h1:
        st.markdown("<div class='mini head'>Indicator</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<div class='mini head' style='text-align:left'>Weight</div>", unsafe_allow_html=True)

    # Filas: (podés dejar el orden como tengas; acá lo dejo por peso desc)
    ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))

    for name, val in ordered:
        c1, c2, c3 = st.columns([6, 2, 3])
        with c1:
            st.markdown(f"<div class='mini row name'>{name}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='mini row wgt'>{float(val):.2f}</div>", unsafe_allow_html=True)
        with c3:
            b1, b2 = st.columns(2)
            dec = b1.button("−", key=f"dec_{name}", disabled=st.session_state.saving)
            inc = b2.button("＋", key=f"inc_{name}", disabled=st.session_state.saving)
            if dec or inc:
                STEP = 0.01
                delta = (-STEP if dec else STEP)
                # Sin lock: solo ajusta este indicador (la suma puede moverse)
                newv = float(np.round(min(max(st.session_state.weights[name] + delta, 0.0), 1.0) + 1e-9, 2))
                st.session_state.weights[name] = newv
                st.rerun()

    # --- Fila TOTAL (neutral si ==1.00; roja si != 1.00) ---
    used_total = float(sum(weights.values()))
    ok = abs(used_total - 1.0) <= EPS
    style = '' if ok else ' style="color:#b3261e;background:rgba(217,48,37,.08);"'

    t1, t2, t3 = st.columns([6, 2, 3])
    with t1:
        st.markdown(f"<div class='mini row name'{style}><b>Total</b></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='mini row wgt'{style}><b>{used_total:.2f}</b></div>", unsafe_allow_html=True)
    with t3:
        # celda vacía para mantener la grilla
        st.markdown(f"<div class='mini row'{style}></div>", unsafe_allow_html=True)


render_compact_table(st.session_state.weights)


# ───────── LIVE RANKING (FIJO SEGÚN CSV) ─────────
# ───────── LIVE RANKING (FIJO SEGÚN CSV) ─────────
# ───────── LIVE RANKING (FIJO SEGÚN CSV, ORDENADO DESC) ─────────
def render_ranking_html_fixed() -> None:
    base = st.session_state.defaults_csv  # pesos crudos del CSV
    # Orden: peso descendente y, ante empate, alfabético
    ordered = sorted(base.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))

    rows = []
    for idx, (name, pts) in enumerate(ordered, start=1):
        rows.append(f"<tr><td>{idx}</td><td>{name}</td><td class='r'>{pts:.2f}</td></tr>")

    total_csv = float(sum(base.values()))
    total_row_style = '' if abs(total_csv - 1.0) <= EPS else ' style="color:#b3261e;background:rgba(217,48,37,.08);"'
    rows.append(f"<tr{total_row_style}><td>—</td><td><b>Total</b></td><td class='r'><b>{total_csv:.2f}</b></td></tr>")

    table_html = f"""
    <div class='rowbox'>
      <div class='name center'>Ranking (guía según CSV)</div>
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
render_ranking_html_fixed()



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
    or (abs(remaining_points(st.session_state.weights)) > EPS)
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
                        st.session_state.thanks_expire = time.time() + THANKS_VISIBLE_SEC
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
now_show = time.time()
if now_show < st.session_state.get("thanks_expire", 0):
    status_box.success("✅ Submitted — Thank you!")
elif st.session_state.get("saving", False):
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
