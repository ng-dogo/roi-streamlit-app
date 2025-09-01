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

/* Badge de valor: alto contraste para light/dark */
.badge{
  padding:2px 10px;border-radius:999px;
  background:#f8f8f8;color:#111;border:1px solid rgba(0,0,0,.15);
  font-size:12px; display:inline-block; min-width:46px; text-align:center;
}

/* Lista ordenada compacta */
.ol-compact {counter-reset:item; margin:0; padding-left:0;}
.ol-compact li{
  list-style:none; counter-increment:item;
  display:flex; justify-content:space-between; align-items:center;
  padding:.25rem .5rem; border-bottom:1px dashed #e9e9e9;
}
.ol-compact li::before{
  content: counter(item) ".";
  margin-right:.5rem; color:#444; width:1.5rem; text-align:right;
}
.comp-name{font-weight:600; color:#111;}
.comp-pct{font-variant-numeric:tabular-nums;}
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
if "last_weights" not in st.session_state:
    st.session_state.last_weights = st.session_state.weights.copy()
if "_prog_update" not in st.session_state:
    st.session_state._prog_update = False  # bandera anti-bucle tras sync programático

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
        if c1 in row and pd.notna(row[c1]): val = row[c1]
        elif c2 in row and pd.notna(row[c2]): val = row[c2]
        if isinstance(val, str): val = val.replace("%", "").strip()
        try: out[comp] = int(round(float(val))) if val not in (None, "") else 0
        except Exception: out[comp] = 0
    return out

def load_defaults_for_key(df: pd.DataFrame, key: str) -> dict | None:
    if df.empty: return None
    k = (key or "").strip().casefold()
    row = df.loc[df["__key__"] == k]
    if row.empty: return None
    return map_defaults_row_to_dict(row.iloc[0])

def autoload_defaults_by_email(email: str):
    df = load_defaults_csv(DEFAULTS_CSV_PATH)
    defaults = load_defaults_for_key(df, email)
    if defaults:
        s = sum(defaults.values())
        if s <= 0:
            defaults = {k: (100 // len(RGI_COMPONENTS)) for k in RGI_COMPONENTS}
            defaults[RGI_COMPONENTS[0]] += 100 - sum(defaults.values())
        else:
            defaults = {k: int(round(v * 100.0 / s)) for k, v in defaults.items()}
            residue = 100 - sum(defaults.values())
            if residue != 0:
                ordered = sorted(RGI_COMPONENTS, key=lambda c: -defaults[c])
                for i in range(abs(residue)):
                    defaults[ordered[i % len(ordered)]] += 1 if residue > 0 else -1
        st.session_state.weights = defaults
        st.session_state.last_weights = defaults.copy()
        # sync widgets
        for c in RGI_COMPONENTS[:-1]:
            st.session_state[f"sl_{c}"] = defaults[c]
        st.session_state._prog_update = True
        st.rerun()
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
    max_tries, delay = 5, 0.5
    for attempt in range(1, max_tries + 1):
        try:
            sh.append_row(row); return
        except APIError as e:
            status = None
            try: status = e.response.status_code
            except Exception:
                try: status = getattr(getattr(e, "response", None), "status", None)
                except Exception: status = None
            if str(status) in {"429","500","502","503","504"} and attempt < max_tries:
                time.sleep(delay); delay *= 2
            else:
                raise
        except Exception: raise

# ─────────────── LÓGICA DE TOPES Y REBALANCE ───────────────
def slider_bounds_by_trailing(weights: dict) -> dict:
    """
    Para cada componente i (excepto el último), calcula:
      - max_i = w[i] + suma(tail)   (lo que se puede quitar a los de abajo)
      - min_i = w[i] - suma(100 - w_tail) (lo que se puede poner a los de abajo)
    Clipa 0..100. El último se auto-calcula (sin slider).
    """
    comps = RGI_COMPONENTS
    n = len(comps)
    w = {c: int(weights[c]) for c in comps}
    bounds = {}
    for i, c in enumerate(comps):
        if i == n - 1:
            # último = lectura auto (sin slider)
            bounds[c] = (w[c], w[c])
            continue
        tail = comps[i+1:]
        reducible = sum(w[t] for t in tail)
        addable = sum(100 - w[t] for t in tail)
        max_i = min(100, w[c] + reducible)
        min_i = max(0, w[c] - addable)
        bounds[c] = (int(min_i), int(max_i))
    return bounds

def trailing_rebalance(weights: dict, changed_idx: int, new_val: int) -> dict:
    """
    Ajusta SOLO la cola (de atrás hacia adelante) para cerrar en 100.
    Si la cola no alcanza, capa el nuevo valor.
    """
    comps = RGI_COMPONENTS
    w = weights.copy()
    changed = comps[changed_idx]
    old_val = int(w[changed])
    new_val = max(0, min(100, int(new_val)))
    if new_val == old_val: return w

    tail = comps[changed_idx+1:]
    if not tail:
        w[changed] = new_val
        sum_others = sum(w[c] for c in comps[:-1])
        w[changed] = max(0, min(100, 100 - sum_others))
        return w

    delta = new_val - old_val
    if delta > 0:
        reducible = sum(int(w[c]) for c in tail)
        allowed_inc = min(delta, reducible)
        new_val = old_val + allowed_inc
        w[changed] = new_val
        remaining = allowed_inc
        for c in reversed(tail):
            take = min(remaining, int(w[c]))
            w[c] = int(w[c]) - take
            remaining -= take
            if remaining <= 0: break
    else:
        add = -delta
        addable = sum(100 - int(w[c]) for c in tail)
        allowed_dec = min(add, addable)
        new_val = old_val - allowed_dec
        w[changed] = new_val
        remaining = allowed_dec
        for c in reversed(tail):
            room = 100 - int(w[c])
            put = min(remaining, room)
            w[c] = int(w[c]) + put
            remaining -= put
            if remaining <= 0: break

    # Sanitizar + cerrar exacto 100
    for c in comps: w[c] = max(0, min(100, int(w[c])))
    s = sum(w.values())
    if s != 100:
        w[comps[-1]] += (100 - s)
        w[comps[-1]] = max(0, min(100, int(w[comps[-1]])))
    return w

def detect_changed_component(new_ui_vals: dict, old: dict) -> int | None:
    for idx, c in enumerate(RGI_COMPONENTS[:-1]):  # último no editable
        if int(new_ui_vals[c]) != int(old[c]):
            return idx
    return None

def reset_to_defaults():
    df = load_defaults_csv(DEFAULTS_CSV_PATH)
    defaults = load_defaults_for_key(df, st.session_state.email)
    if defaults:
        w = defaults
    else:
        equal = 100 // len(RGI_COMPONENTS)
        w = {c: equal for c in RGI_COMPONENTS}
        w[RGI_COMPONENTS[0]] += 100 - sum(w.values())
    st.session_state.weights = w
    st.session_state.last_weights = w.copy()
    # sync sliders
    for c in RGI_COMPONENTS[:-1]:
        st.session_state[f"sl_{c}"] = w[c]
    st.session_state._prog_update = True
    st.rerun()

def equalize_all():
    equal = 100 // len(RGI_COMPONENTS)
    w = {c: equal for c in RGI_COMPONENTS}
    w[RGI_COMPONENTS[0]] += 100 - sum(w.values())
    st.session_state.weights = w
    st.session_state.last_weights = w.copy()
    for c in RGI_COMPONENTS[:-1]:
        st.session_state[f"sl_{c}"] = w[c]
    st.session_state._prog_update = True
    st.rerun()

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
        # autoload_defaults hace rerun; si no hay defaults:
        if st.session_state.stage == 2: st.rerun()

    st.caption("We use your email to match initial weights (if available) and to record your submission.")

# ====== STAGE 2: ALLOCATION ======
if st.session_state.stage == 2:
    st.markdown(f"**Email:** {st.session_state.email}")
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Step 2 · Allocate your 100 points")
    st.caption("Move any slider. Only the last components adjust automatically so the total stays at 100. What you already set stays put. The last item is auto-calculated.")

    # Controles: Reset / Equalize
    c1, c2, _ = st.columns([1,1,4])
    with c1:
        if st.button("Reset to defaults"): reset_to_defaults()
    with c2:
        if st.button("Equalize all"): equalize_all()

    # TOPES dinámicos según el estado actual (último es lectura)
    bounds = slider_bounds_by_trailing(st.session_state.last_weights)

    # Sliders para todos menos el último
    new_ui_vals = {}
    for i, comp in enumerate(RGI_COMPONENTS):
        cols = st.columns([6, 2])
        if i < len(RGI_COMPONENTS) - 1:
            mn, mx = bounds[comp]
            with cols[0]:
                new_ui_vals[comp] = st.slider(
                    comp, mn, mx,
                    int(st.session_state.weights.get(comp, 0)),
                    key=f"sl_{comp}"
                )
            with cols[1]:
                st.write(f"<span class='badge'>{st.session_state.weights.get(comp, 0)}%</span>", unsafe_allow_html=True)
        else:
            # Último: auto, sin slider (lectura)
            with cols[0]:
                st.write(f"**{comp}** (auto)")
            with cols[1]:
                st.write(f"<span class='badge'>{st.session_state.weights.get(comp, 0)}%</span>", unsafe_allow_html=True)

    # Evitar bucle tras un update programático
    if st.session_state._prog_update:
        st.session_state._prog_update = False
    else:
        # Detectar cambio y aplicar rebalance de cola
        idx = detect_changed_component(new_ui_vals, st.session_state.last_weights)
        if idx is not None:
            reb = trailing_rebalance(st.session_state.last_weights, idx, int(new_ui_vals[RGI_COMPONENTS[idx]]))
            st.session_state.weights = reb
            st.session_state.last_weights = reb.copy()
            # Sync widgets (para que no quede un slider en un valor imposible)
            for c in RGI_COMPONENTS[:-1]:
                st.session_state[f"sl_{c}"] = reb[c]
            st.session_state._prog_update = True
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write(f"**Total allocated:** {sum(st.session_state.weights.values())} / 100")

    # Vista “Distribution overview” ordenada (texto claro)
    st.markdown("### Distribution overview")
    sorted_items = sorted(st.session_state.weights.items(), key=lambda kv: kv[1], reverse=True)
    items_html = ['<ol class="ol-compact">']
    for name, pct in sorted_items:
        items_html.append(
            f'<li><span class="comp-name">{name}</span>'
            f'<span class="comp-pct"><span class="badge">{pct}%</span></span></li>'
        )
    items_html.append("</ol>")
    st.markdown("".join(items_html), unsafe_allow_html=True)

    # Submit
    disabled_submit = (
        st.session_state.saving or st.session_state.submitted or
        not st.session_state.email or
        (REQUIRE_TOTAL_100 and sum(st.session_state.weights.values()) != 100)
    )

    # Anti multi-submit
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
