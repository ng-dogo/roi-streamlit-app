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
.stButton>button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:.6rem 1rem;min-height:44px}
.stButton>button:hover{filter:brightness(0.95)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default}
.center input[type=number]{text-align:center;font-weight:600;font-size:18px} /* evita zoom en iOS */
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--border);font-size:.9rem;color:var(--muted)}
.kpis{display:flex;gap:1rem;align-items:center}
.kpis .strong{font-weight:700}

/* Tabla ranking minimalista */
.rank{width:100%; border-collapse:collapse; font-size:.95rem}
.rank th, .rank td{padding:.35rem .5rem; border-bottom:1px solid var(--border)}
.rank th{font-weight:600; color:var(--muted); text-align:left}
.rank td.r{text-align:right}
.small-note{font-size:.9rem;color:var(--muted);margin:.25rem 0 0}

/* Divisor suave entre secciones superiores */
.soft-divider{height:0;border-top:1px solid var(--border);margin:.5rem 0 1rem}

/* ───── Responsivo ───── */
@media (max-width: 900px){
  .main .block-container{max-width:700px}
}
@media (max-width: 680px){
  .main .block-container{max-width:100%; padding:.6rem}
  /* Apilar columnas de Streamlit en móvil */
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"]{
    flex:1 1 100% !important; width:100% !important; padding-right:0 !important;
  }
  .stButton>button{width:100%; margin-top:.25rem}
  .rowbox{padding:.6rem .7rem}
  .rank{font-size:.9rem}
}
@media (max-width: 420px){
  .rank th, .rank td{padding:.25rem .35rem}
  .rank{font-size:.85rem}
  .center input[type=number]{font-size:17px}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 100
STEP_BIG = 10
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
THANKS_VISIBLE_SEC = 3.0

# ───────── STATE ─────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "weights" not in st.session_state:
    st.session_state.weights: Dict[str, int] = {}
if "defaults" not in st.session_state:
    st.session_state.defaults: Dict[str, int] = {}
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
    out["avg_weight"] = pd.to_numeric(out["avg_weight"], errors="coerce").fillna(0.0)
    return out

def normalize_to_100_ints(weights: Dict[str, float]) -> Dict[str, int]:
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        base = 100 // n
        res = 100 - base * n
        out = {k: base for k in weights}
        for k in list(out.keys())[:res]:
            out[k] += 1
        return out
    raw = {k: 100.0 * v / total for k, v in weights.items()}
    floors = {k: int(np.floor(x)) for k, x in raw.items()}
    leftover = 100 - sum(floors.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    out = floors.copy()
    for k in order[:leftover]:
        out[k] += 1
    return out

def remaining_points(weights: Dict[str, int]) -> int:
    return int(TOTAL_POINTS - int(sum(weights.values())))

def adjust(comp: str, delta: int):
    cur = int(st.session_state.weights[comp])
    if delta > 0:
        allowed = min(delta, remaining_points(st.session_state.weights))
        new_val = min(100, cur + allowed)
    else:
        new_val = max(0, cur + delta)
    st.session_state.weights[comp] = int(new_val)
    st.session_state[f"num_{comp}"] = int(new_val)

def make_on_change(comp: str):
    def _cb():
        cur = int(st.session_state.weights[comp])
        new_val = int(st.session_state[f"num_{comp}"])
        delta = new_val - cur
        if delta > 0:
            allowed = min(delta, remaining_points(st.session_state.weights))
            st.session_state.weights[comp] = cur + allowed
        else:
            st.session_state.weights[comp] = max(0, new_val)
        st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])
    return _cb

def payload_hash(email: str, indicators: List[str], weights: Dict[str, int]) -> str:
    tpl = (email.strip().lower(), tuple(indicators), tuple(int(weights[k]) for k in indicators))
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

def save_to_sheet(email: str, weights: Dict[str, int], session_id: str, indicator_order: List[str]):
    sh = get_worksheet()
    headers = ["timestamp","email","session_id"] + indicator_order + ["total"]
    sh.update("A1", [headers])
    row = [dt.datetime.now().isoformat(), email, session_id] + [int(weights[k]) for k in indicator_order] + [int(sum(weights.values()))]
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

# ───────── LOAD DEFAULTS ─────────
if not st.session_state.weights:
    df = load_defaults_csv(CSV_PATH)
    indicators = df["indicator"].tolist()
    defaults_raw = {r.indicator: float(r.avg_weight) for r in df.itertuples()}
    defaults_int = normalize_to_100_ints(defaults_raw)
    st.session_state.defaults = defaults_int
    st.session_state.weights = dict(defaults_int)
    st.session_state._init_inputs = True
else:
    indicators = list(st.session_state.weights.keys())

# ───────── UI ─────────
st.title("RGI – Budget Allocation Points")

# Email (fila 1)
st.session_state.email = st.text_input("Email", value=st.session_state.email, placeholder="name@example.org")

# División suave
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# Progreso + Reset (fila 2)
col_prog, col_reset = st.columns([3, 1])
with col_prog:
    used = int(sum(st.session_state.weights.values()))
    rem = remaining_points(st.session_state.weights)
    pct_used = max(0, min(100, used)) / 100.0
    st.progress(pct_used, text=f"Used {used} • Remaining {rem}")
with col_reset:
    if st.button("Reset to averages", disabled=st.session_state.saving):
        st.session_state.weights = dict(st.session_state.defaults)
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Allocation")

if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = int(st.session_state.weights[comp])
    st.session_state._init_inputs = False

for comp in indicators:
    st.markdown(f"<div class='name'>{comp}</div>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 3, 1])
    with colL:
        cur = int(st.session_state.weights[comp])
        st.button("−10", key=f"m10_{comp}", on_click=lambda c=comp: adjust(c, -STEP_BIG),
                  disabled=(cur <= 0) or st.session_state.saving)
    with colC:
        st.markdown("<div class='rowbox center'>", unsafe_allow_html=True)
        # max_value fijo evita resets por cambios dinámicos de límites
        st.number_input(
            label="", key=f"num_{comp}", min_value=0, max_value=100,
            step=1, format="%d", label_visibility="collapsed",
            on_change=make_on_change(comp), disabled=st.session_state.saving
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        can_add = (remaining_points(st.session_state.weights) > 0) and (int(st.session_state.weights[comp]) < 100)
        st.button("+10", key=f"p10_{comp}", on_click=lambda c=comp: adjust(c, STEP_BIG),
                  disabled=(not can_add) or st.session_state.saving)

# ───────── LIVE RANKING (minimal, non-intrusive) ─────────
def render_ranking_html(weights: Dict[str, int]) -> None:
    ordered = sorted(weights.items(), key=lambda kv: (-int(kv[1]), kv[0].lower()))
    rows = []
    rank = 1
    for name, pts in ordered:
        rows.append(f"<tr><td>{rank}</td><td>{name}</td><td class='r'>{int(pts)}</td></tr>")
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

# ───────── FOOTER / SUBMIT ─────────
st.markdown("<hr/>", unsafe_allow_html=True)
ok_email = bool(EMAIL_RE.match(st.session_state.email or ""))
now = time.time()
cooling = (now - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC

disabled_submit = (
    (not ok_email)
    or st.session_state.submitted
    or (remaining_points(st.session_state.weights) != 0)
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
