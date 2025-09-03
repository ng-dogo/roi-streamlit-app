# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, os, time, hashlib, random
from typing import Dict, List
from threading import Lock
import psutil
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError

# ───────── CONFIG ─────────
st.set_page_config(page_title="RGI – Budget Allocation Points", page_icon="⚡", layout="wide")

CSS = """
<style>
:root{
  --brand:#0E7C66; --muted:rgba(128,128,128,.85); --border:rgba(127,127,127,.18);
  --ok:#0c8a4d; --bad:#b3261e; --ok-bg:rgba(12,138,77,.12); --bad-bg:rgba(217,48,37,.10);
}
html, body, [class*="css"]{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
/* Contenedor ancho y comprimido verticalmente */
.main .block-container{max-width:1180px; padding-top:.6rem; padding-bottom:.6rem}

/* Tipografías y densidad */
h1,h2,h3, .stMarkdown p{margin:.25rem 0}
small, .small-note, .caption{font-size:.88rem;color:var(--muted)}
.compact{margin:.35rem 0}

/* Botones */
.stButton>button{
  background:var(--brand);color:#fff;border:none;border-radius:10px;
  padding:.4rem .8rem; font-weight:600; height:2.2rem; min-height:2.2rem;
}
.stButton>button:hover{filter:brightness(.96)}
.stButton>button:disabled{background:#0b6b59;color:#fff;opacity:1;cursor:default}

/* Inputs densos */
input[type="number"], input[type="text"], textarea{
  height:2.2rem; min-height:2.2rem; padding:.2rem .5rem; font-weight:600
}
label, .stNumberInput label{margin-bottom:.15rem !important}

/* Tarjetas para cada indicador (compactas) */
.grid{margin-top:.2rem}
.card{
  padding:.4rem .5rem; border:1px solid var(--border); border-radius:10px;
  margin-bottom:.5rem; background:rgba(255,255,255,.6)
}
.card .lbl{font-size:.88rem;color:var(--muted);margin-bottom:.15rem}

/* Ranking a la derecha, tabla minimalista */
.rank {width:100%; border-collapse:collapse; font-size:.92rem}
.rank th, .rank td {padding:.28rem .4rem; border-bottom:1px solid var(--border)}
.rank th {font-weight:600; color:var(--muted); text-align:center}
.rank td:first-child, .rank td:last-child {text-align:center}

/* Píldoras de estado (suma) */
.pill{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.25rem .5rem; border-radius:999px; font-size:.9rem; font-weight:600;
  border:1px solid transparent; white-space:nowrap
}
.pill.ok{ color:var(--ok); background:var(--ok-bg); border-color:rgba(12,138,77,.25) }
.pill.bad{ color:var(--bad); background:var(--bad-bg); border-color:rgba(217,48,37,.28) }

/* HUD flotante inferior (compacto) */
.hud {
  position: fixed; left: 12px; bottom: 10px; width: 520px;
  background: rgba(255,255,255,.9); backdrop-filter: blur(6px);
  border: 1px solid var(--border); border-radius: 10px;
  box-shadow: 0 6px 20px rgba(0,0,0,.08); padding: .45rem .6rem; z-index: 9999;
}
.hud-row{ display:flex; align-items:center; gap:.5rem }
.hud-mono{ font-variant-numeric: tabular-nums; font-weight:700; font-size:.95rem }
.hud-spacer{ flex:1 }
.hud-bar{ position:relative; height:6px; background:rgba(127,127,127,.18); border-radius:999px; overflow:hidden; width:58% }
.hud-fill{ position:absolute; left:0; top:0; bottom:0; background:var(--brand); width:0% }

/* Fila superior compacta */
.toprow .element-container{margin-bottom:.35rem}

/* Dark mode tweaks */
@media (prefers-color-scheme: dark){
  .hud{ background: rgba(18,18,18,.85); border-color: rgba(255,255,255,.12) }
  .pill.ok{ background:rgba(12,138,77,.18) }
  .pill.bad{ background:rgba(217,48,37,.15) }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────── CONSTANTS ─────────
CSV_PATH = os.getenv("RGI_DEFAULTS_CSV", "rgi_bap_defaults.csv")  # columns: indicator, avg_weight
TOTAL_POINTS = 1.0  # pesos deben sumar 1.00
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_COOLDOWN_SEC = 2.0
EPS = 1e-6  # tolerancia numérica (usamos suma redondeada a 2 decimales para UI)

# ───────── STATE ─────────
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
for k, v in dict(weights={}, defaults={}, email="", submitted=False, _init_inputs=False,
                 saving=False, last_submit_ts=0.0, last_payload_hash="", inflight_payload_hash="",
                 status="idle", error_msg="").items():
    st.session_state.setdefault(k, v)

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
    """Redondea a 0.01 manteniendo suma exacta = 1.00 (en centésimas)."""
    if not weights: return {}
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(1, len(weights))
        cents_each = int(round(100 / n)); cents = [cents_each]*n
        diff = 100 - sum(cents)
        for i in range(abs(diff)):
            idx = i % n; cents[idx] += 1 if diff > 0 else -1
        return {k: v/100.0 for k, v in zip(weights.keys(), cents)}
    scaled = {k: 100.0 * (v / total) for k, v in weights.items()}
    rounded = {k: int(np.floor(s + 0.5)) for k, s in scaled.items()}
    resid = {k: (scaled[k] - rounded[k]) for k in weights}
    diff = 100 - sum(rounded.values())
    if diff > 0:
        for k in sorted(weights.keys(), key=lambda k: resid[k], reverse=True)[:diff]:
            rounded[k] += 1
    elif diff < 0:
        for k in sorted(weights.keys(), key=lambda k: resid[k])[:abs(diff)]:
            rounded[k] -= 1
    return {k: rounded[k] / 100.0 for k in rounded}

def sum_used(weights: Dict[str, float]) -> float:
    """Suma redondeada a 2 decimales (consistente con la UI)."""
    return float(np.round(sum(map(float, weights.values())) + 1e-9, 2))

def remaining_points(weights: Dict[str, float]) -> float:
    """Diferencia redondeada a 2 decimales (consistente con la UI)."""
    return float(np.round(TOTAL_POINTS - sum_used(weights), 2))

def make_on_change(comp: str):
    """Permite pasarse de 1 en la suma global; sólo acota cada input 0..1."""
    def _cb():
        val = float(st.session_state.get(f"num_{comp}", 0.0))
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
        + [float(np.round(sum_used(weights), 2))]
    )
    base_delay, attempts = 0.35, 5
    for attempt in range(1, attempts + 1):
        time.sleep(random.uniform(0.0, 0.3))  # jitter
        try:
            sh.append_row(row, value_input_option="RAW")
            return
        except APIError as e:
            if attempt == attempts:
                raise RuntimeError("No pudimos guardar en Google Sheets tras varios intentos. Volvé a intentar una vez más.") from e
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

# ───────── UI SUPER COMPACTA ─────────
# Título reducido
st.markdown("### RGI – Budget Allocation Points")

# Top row: Email + Reset + Estado suma + Submit
top_l, top_mid, top_state, top_r = st.columns([2.5, 1, 1.5, 1], gap="small")

with top_l:
    st.session_state.email = st.text_input(
        "Email", value=st.session_state.email, placeholder="name@example.org"
    )

with top_mid:
    if st.button("Reset to averages", key="reset_btn", use_container_width=True, disabled=st.session_state.saving):
        st.session_state.weights = dict(st.session_state.defaults)
        for comp in st.session_state.weights:
            st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
        st.rerun()

# Estado de suma (píldora compacta)
used = sum_used(st.session_state.weights)
rem = remaining_points(st.session_state.weights)
def sum_pill_html(used: float, rem: float) -> str:
    if abs(rem) <= 0.0 + EPS:
        return f'<span class="pill ok">✅ Suma 1.00 — listo</span>'
    tip = f"Faltan {abs(rem):.2f}" if rem > 0 else f"Sobran {abs(rem):.2f}"
    return f'<span class="pill bad">⚠️ {tip} para llegar a 1.00</span>'

with top_state:
    st.markdown(sum_pill_html(used, rem), unsafe_allow_html=True)

# Lógica de submit
email_raw = st.session_state.email or ""
email_norm = email_raw.strip()
ok_email = bool(EMAIL_RE.match(email_norm))
now = time.time()
cooling = (now - st.session_state.last_submit_ts) < SUBMISSION_COOLDOWN_SEC
can_submit_sum = (abs(rem) <= EPS)  # sólo si la suma es exactamente 1.00 (redondeo UI)
disabled_submit = (not ok_email) or st.session_state.submitted or (not can_submit_sum) or st.session_state.saving or cooling

status_box = st.empty()

with top_r:
    submit_label = "Submit" if not st.session_state.submitted else "✅ Submitted"
    if st.button(submit_label, key="submit_btn", use_container_width=True, disabled=disabled_submit):
        submit_lock = get_submit_lock()
        if not submit_lock.acquire(blocking=False):
            st.toast("Submission already in progress…", icon="⏳"); st.stop()
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
                        st.session_state.status = "error"; st.session_state.error_msg = str(e)
                    finally:
                        st.session_state.saving = False
                        st.session_state.inflight_payload_hash = ""
                        st.session_state.last_submit_ts = time.time()
        finally:
            try: submit_lock.release()
            except Exception: pass

# Línea final de estado (compacta)
if st.session_state.get("saving", False):
    status_box.info("⏳ Saving your response… please wait.")
else:
    if st.session_state.submitted:
        status_box.success("✅ Saved")
    elif st.session_state.status == "duplicate":
        status_box.info("Ya guardaste esta configuración exacta.")
        st.session_state.status = "idle"
    elif st.session_state.status == "cooldown":
        status_box.info("Esperá un instante antes de enviar otra vez.")
        st.session_state.status = "idle"
    elif st.session_state.status == "error":
        status_box.error(f"Error al guardar. {st.session_state.get('error_msg','')}")
        st.session_state.status = "idle"
    else:
        status_box.empty()

# ───────── GRID DE INDICADORES (4×2) + RANKING A LA DERECHA ─────────
if st.session_state.get("_init_inputs"):
    for comp in indicators:
        st.session_state[f"num_{comp}"] = float(st.session_state.weights[comp])
    st.session_state._init_inputs = False

left, right = st.columns([2.1, 1], gap="large")

with left:
    st.markdown("##### Allocation", help="Distribuí pesos (0.00–1.00) por indicador. La suma total debe ser 1.00.")
    # 4 columnas x 2 filas (8 indicadores)
    C = 4
    rows = [indicators[i:i+C] for i in range(0, len(indicators), C)]
    for row in rows:
        cols = st.columns(C, gap="small")
        for comp, c in zip(row, cols):
            with c:
                st.markdown(f"<div class='card'><div class='lbl'>{comp}</div>", unsafe_allow_html=True)
                st.number_input(
                    label="",
                    key=f"num_{comp}",
                    min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                    label_visibility="collapsed",
                    on_change=make_on_change(comp),
                    disabled=st.session_state.saving
                )
                st.markdown("</div>", unsafe_allow_html=True)

with right:
    # Ranking en paralelo (compacto)
    def render_ranking_html(weights: Dict[str, float]) -> None:
        ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0].lower()))
        rows = []
        for i, (name, pts) in enumerate(ordered, start=1):
            rows.append(f"<tr><td>{i}</td><td>{name}</td><td>{float(pts):.2f}</td></tr>")
        table_html = f"""
        <div class='card'>
          <div class='lbl' style="font-weight:700;color:inherit;margin-bottom:.25rem">Ranking</div>
          <table class="rank">
            <thead><tr><th>#</th><th>Indicator</th><th>Weight</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

    render_ranking_html(st.session_state.weights)

# ───────── HUD FLOTANTE (compacto) ─────────
def render_floating_hud(used: float):
    pct = max(0.0, min(1.0, (used / TOTAL_POINTS) if TOTAL_POINTS else 0.0)) * 100.0
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    st.markdown(f"""
    <div class="hud">
      <div class="hud-row">
        <div class="hud-mono">{used:.2f}/1.00</div>
        <div class="hud-spacer"></div>
        <div class="hud-bar"><div class="hud-fill" style="width:{pct:.2f}%"></div></div>
        <div class="small-note">RAM {mem_mb:.1f} MB</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

render_floating_hud(sum_used(st.session_state.weights))
