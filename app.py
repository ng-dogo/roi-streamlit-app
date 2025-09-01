# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, uuid, datetime as dt, time
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import APIError  # para manejar errores de cuota/servidor

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="RGI – Budget Allocation", page_icon="⚡", layout="centered")

MINI_CSS = """
<style>
:root{ --primary:#02593B; }
html, body, [class*="css"]{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
.stButton>button{background:var(--primary);color:#fff;border:none;border-radius:8px;padding:.5rem 1rem}
.stButton>button:disabled{background:#ccc;color:#666}
.main .block-container{max-width:900px}
hr{border:none;border-top:1px solid #e6e6e6;margin:1rem 0}
.segment-bar{display:flex;height:18px;border-radius:6px;overflow:hidden;background:#eee}
.segment{height:18px}
.legend{display:grid;grid-template-columns: 1fr auto auto auto;gap:.5rem .75rem;align-items:center}
.legend div.label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{padding:2px 8px;border-radius:999px;background:#f5f5f5;font-size:12px}
.lock{font-size:12px;color:#555}
</style>
"""
st.markdown(MINI_CSS, unsafe_allow_html=True)

# ─────────────── PARAMS ───────────────
RGI_COMPONENTS = [
    "Legal Framework",
    "Independence & Accountability",
    "Tariff Methodology",
    "Participation & Transparency",
    "Legal Mandate",
    "Clarity of Roles & Objectives",
    "Open Access to Information",
    "Transparency",
]
DEFAULTS_CSV_PATH = "rgi_defaults.csv"
REQUIRE_TOTAL_100 = True

# ─────────────── STATE ───────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = 1
if "email" not in st.session_state:
    st.session_state.email = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "saving" not in st.session_state:
    st.session_state.saving = False
if "weights" not in st.session_state:
    st.session_state.weights = {c: 0 for c in RGI_COMPONENTS}  # enteros 0..100
if "locks" not in st.session_state:
    st.session_state.locks = {c: False for c in RGI_COMPONENTS}
if "last_weights" not in st.session_state:
    st.session_state.last_weights = st.session_state.weights.copy()

# ─────────────── HELPERS (CSV / Defaults) ───────────────
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def ensure_headers(sh):
    headers = ["timestamp","email","session_id"] + RGI_COMPONENTS
    vals = sh.get_all_values()
    if not vals:
        sh.append_row(headers)

@st.cache_data(ttl=300)
def load_defaults_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.astype(str).str.strip()
        key_col = "email" if "email" in df.columns else ("name" if "name" in df.columns else None)
        if key_col is None:
            return pd.DataFrame()
        df[key_col] = df[key_col].astype(str).str.strip()
        df["__key__"] = df[key_col].str.casefold()
        return df
    except Exception:
        return pd.DataFrame()

def map_defaults_row_to_dict(row: pd.Series) -> dict:
    out = {}
    for comp in RGI_COMPONENTS:
        val = None
        c1, c2 = f"w::{comp}", comp
        if c1 in row and pd.notna(row[c1]):
            val = row[c1]
        elif c2 in row and pd.notna(row[c2]):
            val = row[c2]
        if isinstance(val, str):
            val = val.replace("%", "").strip()
        try:
            out[comp] = int(round(float(val))) if val not in (None, "") else 0
        except Exception:
            out[comp] = 0
    return out

def load_defaults_for_key(df: pd.DataFrame, key: str) -> dict | None:
    if df.empty:
        return None
    k = (key or "").strip().casefold()
    row = df.loc[df["__key__"] == k]
    if row.empty:
        return None
    return map_defaults_row_to_dict(row.iloc[0])

def autoload_defaults_by_email(email: str):
    df = load_defaults_csv(DEFAULTS_CSV_PATH)
    defaults = load_defaults_for_key(df, email)
    if defaults:
        # Si no suman 100, normalizamos a 100
        s = sum(defaults.values())
        if s <= 0:
            defaults = {k: (100 // len(RGI_COMPONENTS)) for k in RGI_COMPONENTS}
            defaults[RGI_COMPONENTS[0]] += 100 - sum(defaults.values())
        else:
            defaults = {k: int(round(v * 100.0 / s)) for k, v in defaults.items()}
            # Ajuste residuo por redondeo
            residue = 100 - sum(defaults.values())
            if residue != 0:
                # distribuir el residuo empezando por los mayores
                ordered = sorted(RGI_COMPONENTS, key=lambda c: -defaults[c])
                for i in range(abs(residue)):
                    idx = i % len(ordered)
                    c = ordered[idx]
                    defaults[c] += 1 if residue > 0 else -1
        st.session_state.weights = defaults
        st.session_state.last_weights = defaults.copy()
        st.success("Defaults loaded from CSV.")
    else:
        st.info("No defaults found for this email. You can allocate manually.")

# ─────────────── HELPERS (Sheets) ───────────────
@st.cache_resource
def get_sheet():
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    return client.open_by_key(st.secrets.sheet_id).sheet1

def save_to_sheet(email: str, weights: dict, session_id: str):
    sh = get_sheet()
    ensure_headers(sh)
    row = [dt.datetime.now().isoformat(), email, session_id] + [int(weights[c]) for c in RGI_COMPONENTS]
    # Reintentos con backoff exponencial ante cuota/errores transitorios
    max_tries, delay = 5, 0.5
    for attempt in range(1, max_tries + 1):
        try:
            sh.append_row(row)
            return
        except APIError as e:
            status = None
            try:
                status = e.response.status_code
            except Exception:
                try:
                    status = getattr(getattr(e, "response", None), "status", None)
                except Exception:
                    status = None
            if str(status) in {"429", "500", "502", "503", "504"} and attempt < max_tries:
                time.sleep(delay); delay *= 2
            else:
                raise
        except Exception:
            raise

# ─────────────── REBALANCE ENGINE (mantener 100) ───────────────
def _rebalance(weights: dict, changed: str, new_val: int, locks: dict) -> dict:
    """
    Ajusta proporcionalmente el resto (no bloqueados) para que la suma sea 100.
    Mantiene enteros y reparte el residuo por magnitud.
    """
    old_val = int(weights[changed])
    if new_val == old_val:
        return weights

    # Límite duro si todo lo demás está bloqueado o en 0 y se quiere subir
    other_keys = [c for c in RGI_COMPONENTS if c != changed and not locks.get(c, False)]
    if not other_keys:
        # Sin grados de libertad: cap al máximo posible para que total=100
        cap = 100 - sum(weights[c] for c in RGI_COMPONENTS if c != changed)
        new_val = max(0, min(int(new_val), int(cap)))
        w = weights.copy(); w[changed] = new_val
        return w

    # Calculamos delta y la masa de los otros (solo no bloqueados)
    delta = int(new_val) - int(old_val)
    others_sum = sum(int(weights[c]) for c in other_keys)

    w = weights.copy()
    w[changed] = int(new_val)

    if delta == 0:
        return w

    if delta > 0:
        # hay que restar 'delta' de los otros
        reducible = others_sum
        allowed = min(delta, reducible)
        # escala multiplicativa
        if others_sum > 0:
            factor = (others_sum - allowed) / float(others_sum)
            floats = {c: weights[c] * factor for c in other_keys}
        else:
            floats = {c: weights[c] for c in other_keys}
    else:
        # delta < 0: hay que sumar -delta a los otros
        add = -delta
        if others_sum > 0:
            # distribuir proporcional a sus pesos actuales
            factor = (others_sum + add) / float(others_sum)
            floats = {c: weights[c] * factor for c in other_keys}
        else:
            # si todos 0: distribuir equitativo entre no bloqueados
            base = add / float(len(other_keys))
            floats = {c: base for c in other_keys}

    # Redondeo a enteros + ajuste de residuo
    rounded = {c: int(round(floats[c])) for c in other_keys}
    # Ajustar residuo para que total=100 exacto
    w.update(rounded)
    total = sum(int(v) for v in w.values())
    residue = 100 - total
    if residue != 0:
        # Orden para aplicar residuo: por mayor fracción faltante respecto a float original
        diffs = sorted(other_keys, key=lambda c: (floats[c] - rounded[c]), reverse=(residue > 0))
        for i in range(abs(residue)):
            c = diffs[i % len(diffs)]
            w[c] += 1 if residue > 0 else -1

    # Cap de seguridad por si algún negativo cae por rounding
    for c in RGI_COMPONENTS:
        w[c] = max(0, min(100, int(w[c])))

    # Último ajuste fino si por límites quedó off
    s = sum(w.values())
    if s != 100:
        # mover la diferencia al componente más grande (si sobra) o al menor (si falta)
        target = max(w, key=w.get) if s > 100 else min(w, key=w.get)
        w[target] -= (s - 100)
    return w

def apply_rebalance_from_ui(new_ui_vals: dict):
    """
    Detecta qué componente cambió y aplica _rebalance() respetando locks.
    """
    old = st.session_state.last_weights
    changed = None
    for c in RGI_COMPONENTS:
        if int(new_ui_vals[c]) != int(old[c]):
            changed = c
            break
    if changed is None:
        return  # nada cambió
    # si el cambiado está locked, ignoramos cambio y restauramos UI
    if st.session_state.locks.get(changed, False):
        st.session_state.weights = old.copy()
        return
    rebalanced = _rebalance(old, changed, int(new_ui_vals[changed]), st.session_state.locks)
    st.session_state.weights = rebalanced
    st.session_state.last_weights = rebalanced.copy()

def reset_to_defaults():
    # si había defaults por email, reaplicamos; si no, reparto parejo
    df = load_defaults_csv(DEFAULTS_CSV_PATH)
    defaults = load_defaults_for_key(df, st.session_state.email)
    if defaults:
        st.session_state.weights = defaults
    else:
        equal = 100 // len(RGI_COMPONENTS)
        st.session_state.weights = {c: equal for c in RGI_COMPONENTS}
        st.session_state.weights[RGI_COMPONENTS[0]] += 100 - sum(st.session_state.weights.values())
    st.session_state.last_weights = st.session_state.weights.copy()

def equalize_unlocked():
    unlocked = [c for c in RGI_COMPONENTS if not st.session_state.locks.get(c, False)]
    if not unlocked:
        return
    locked_sum = sum(st.session_state.weights[c] for c in RGI_COMPONENTS if c not in unlocked)
    pool = 100 - locked_sum
    if pool < 0:
        pool = 0
    base = pool // len(unlocked)
    w = st.session_state.weights.copy()
    for c in unlocked:
        w[c] = base
    # residuo
    residue = pool - base * len(unlocked)
    for i in range(residue):
        w[unlocked[i % len(unlocked)]] += 1
    st.session_state.weights = w
    st.session_state.last_weights = w.copy()

# ─────────────── UI ───────────────
st.title("RGI – Budget Allocation")

# ====== STAGE 1: EMAIL ======
if st.session_state.stage == 1:
    st.subheader("Step 1 · Your email")
    email = st.text_input("Email", placeholder="name@example.org", value=st.session_state.email)

    can_continue = bool(EMAIL_RE.match(email))
    if st.button("Continue", disabled=not can_continue):
        st.session_state.email = email.strip()
        st.session_state.stage = 2
        autoload_defaults_by_email(st.session_state.email)
        st.rerun()

    st.caption("We use your email to match initial weights (if available) and to record your submission.")

# ====== STAGE 2: ALLOCATION ======
if st.session_state.stage == 2:
    st.markdown(f"**Email:** {st.session_state.email}")
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Step 2 · Allocate your 100 points")
    st.caption("Move any slider. The others adjust automatically to keep the total at 100. You can lock items you don't want to change.")

    # ── Controls: Reset / Equalize
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1,1,3])
    with c_ctrl1:
        if st.button("Reset to defaults"):
            reset_to_defaults()
    with c_ctrl2:
        if st.button("Equalize (unlocked)"):
            equalize_unlocked()

    # ── Stacked 100% bar (visual)
    total_now = sum(st.session_state.weights.values())
    widths = [max(0, st.session_state.weights[c]) for c in RGI_COMPONENTS]
    # Evitar división por cero
    widths = [w if total_now > 0 else 0 for w in widths]

    bar_html = ['<div class="segment-bar">']
    for i, c in enumerate(RGI_COMPONENTS):
        pct = 0 if total_now == 0 else (widths[i] / total_now * 100.0)
        bar_html.append(f'<div class="segment" title="{c}: {st.session_state.weights[c]}%" style="width:{pct}%; background:rgba(2,89,59,{0.35 + 0.05*i});"></div>')
    bar_html.append("</div>")
    st.markdown("".join(bar_html), unsafe_allow_html=True)

    st.write("")  # pequeño espacio

    # ── Lista de sliders + locks
    new_ui_vals = {}
    for comp in RGI_COMPONENTS:
        cols = st.columns([6, 2, 1, 1])
        with cols[0]:
            new_ui_vals[comp] = st.slider(comp, 0, 100, int(st.session_state.weights.get(comp, 0)), key=f"sl_{comp}")
        with cols[1]:
            st.write(f"<span class='badge'>{st.session_state.weights.get(comp, 0)}%</span>", unsafe_allow_html=True)
        with cols[2]:
            st.session_state.locks[comp] = st.checkbox("Lock", value=st.session_state.locks.get(comp, False), key=f"lk_{comp}")
        with cols[3]:
            st.write("")  # spacer

    # Aplicar rebalance si cambió uno
    apply_rebalance_from_ui(new_ui_vals)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write(f"**Total allocated:** {sum(st.session_state.weights.values())} / 100")

    # Submit
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted or
        not st.session_state.email or
        (REQUIRE_TOTAL_100 and sum(st.session_state.weights.values()) != 100)
    )

    # 🔒 Anti multi-submit
    if st.button("Submit", disabled=disabled_submit):
        st.session_state.submitted = True
        st.session_state.saving = True
        try:
            save_to_sheet(st.session_state.email, st.session_state.weights, st.session_state.session_id)
            st.success("Saved. Thank you.")
        except Exception as e:
            st.session_state.submitted = False
            st.error(f"Error saving your response. {e}")
        finally:
            st.session_state.saving = False

    if st.session_state.submitted:
        st.caption("Response already submitted for this session.")
