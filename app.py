# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib
from typing import Dict, List
from threading import Lock

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

/* 🔧 Mejora focus/tap de los number inputs (evita "rojo") */
.center input[type=number]{
  outline:none !important; box-shadow:none !important; -webkit-tap-highlight-color: transparent;
}
.center input[type=number]:focus,
.center input[type=number]:active{
  outline:none !important;
  box-shadow:0 0 0 2px rgba(14,124,102,.15) !important;
}

/* 🔧 Spin buttons WebKit más cómodos al tacto */
.center input[type=number]::-webkit-inner-spin-button,
.center input[type=number]::-webkit-outer-spin-button{
  opacity:1; height:28px; width:28px; margin:0 2px; background:transparent;
}

.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--border);font-size:.9rem;color:var(--muted)}
.kpis{display:flex;gap:1rem;align-items:center}
.kpis .strong{font-weight:700}

/* Tabla ranking minimalista (no widgets) */
.rank{width:100%; border-collapse:collapse; font-size:.95rem}
.rank th, .rank td{padding:.35rem .5rem; border-bottom:1px solid var(--border)}
.rank th{font-weight:600; color:var(--muted); text-align:left}
.rank td.r{text-align:right}
.small-note{font-size:.9rem;color:var(--muted);margin:.25rem 0 0}

/* Divisor suave entre secciones superiores */
.soft-divider{height:0;border-top:1px solid var(--border);margin:.5rem 0 1rem}

/* — HUD flotante inferior (Opción A) — */
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
@media (hover:hover){
  .hud:hover{ box-shadow: 0 8px 26px rgba(0,0,0,.12) }
}
@media (max-width: 480px){
  .hud { bottom: 8px; padding: .45rem .6rem }
}
@media (prefers-color-scheme: dark){
  .hud{
    background: rgba(18,18,18,.85);
    border-color: rgba(255,255,255,.12);
  }
  .hud-mono{ color: rgba(255,255,255,.92); }
  .hud-bar{ background: rgba(255,255,255,.15); }
}

/* 🎯 Botones +/− específicos de cada fila (grandes y táctiles) */
.stepper { display:flex; align-items:center; justify-content:center }
.stepper .stButton>button{
  -webkit-appearance:none; appearance:none;
  border:1px solid var(--border); background:#fff;
  padding:.4rem .6rem; border-radius:12px; font-weight:800;
  min-width:44px; min-height:40px; line-height:1; text-align:center;
  transition: transform .04s ease, box-shadow .12s ease, background .12s ease;
  color:#0b3941;
}
.stepper .stButton>button:active{ transform: translateY(1px) scale(0.98); }
@media (prefers-color-scheme: dark){
  .stepper .stButton>button{ background:#121212; color:#f2f2f2; border-color:rgba(255,255,255,.14); }
  .stepper .stButton>button:hover{ filter:brightness(1.08) }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")
TOTAL_POINTS = 1.0
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
THANKS_VISIBLE_SEC = 3.0
EPS = 1e-6

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
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").clip(0.0, 1.0).fillna(0.0)
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

@st.cache_resource(show_spinner=False)
def get_submit_lock() -> Lock:
    return Lock()

def save_to_sheet(email: str, weights: Dict[str, float], session_id: str, indicator_order: List[str]):
    sh = get_worksheet()
    headers = ["timestamp","email","session_id"] + indicator_order + ["total"]
    sh.update("A1", [headers])
    row = [dt.datetime.now().isoformat(), email, session_id] + [float(np.round(weights[k], 2)) for k in indicator_order] + [float(np.round(sum(weights.values()), 2))]
    delay = 0.5
    for attempt in range(3):
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError:
            if attempt == 2:
                raise
            time.sleep(delay)
            delay *= 2

# ➕ Ajuste fino con paso 0.01 mediante botones +/−
def adjust_small(comp: str, delta: float = 0.01):
    cur = float(st.session_state.weights[comp])
    if delta > 0:
        allowed = min(delta, max(0.0, remaining_points(st.session_state.weights)))
        new_val = min(1.0, cur + allowed)
    else:
        new_val = max(0.0, cur + delta)
    new_val = float(np.round(new_val, 2))
    st.session_state.weights[comp] = new_val
    st.session_state[f"num_{comp}"] = new_val

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

# Línea suave
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# Reset arriba a la derecha
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
    # Layout: −  [ number_input ]  +
    c1, c2, c3 = st.columns([1,3,1])
    with c1:
        st.markdown('<div class="stepper">', unsafe_allow_html=True)
        st.button("−", key=f"minus_{comp}", on_click=lambda c=comp: adjust_small(c, -0.01), disabled=st.session_state.saving)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
        st.number_input(
            label="", key=f"num_{comp}",
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            label_visibility="collapsed",
            on_change=make_on_change(comp),
            disabled=st.session_state.saving
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stepper">', unsafe_allow_html=True)
        st.button("+", key=f"plus_{comp}", on_click=lambda c=comp: adjust_small(c, +0.01), disabled=st.session_state.saving)
        st.markdown('</div>', unsafe_allow_html=True)

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
      <div class='name'>Ranking</div>
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

# ───────── HUD FLOTANTE (Opción A) ─────────
def render_floating_hud(used: float, rem: float, pct_used: float):
    pct = max(0.0, min(1.0, pct_used)) * 100.0
    st.markdown(f"""
    <div class="hud">
      <div class="hud-row">
        <div class="hud-mono">Used {used:.2f}/{(used+(1-used)):.2f}</div>
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

# ───────── FOOTER / SUBMIT ─────────
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(st.session_state.email or ""))
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
                        st.session_state.thanks_expire = time.time() + THANKS_VISIBLE_SEC
                        st.toast("Saved. Thank you!", icon="✅")
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
if st.session_state.status == "saving":
    status_box.warning("Submission in progress. Please wait…")
    if not st.session_state.get("saving", False):
        st.session_state.status = "idle"
elif st.session_state.status == "saved":
    status_box.success("Saved. Thank you.")
    if time.time() >= st.session_state.thanks_expire:
        st.session_state.status = "idle"
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
