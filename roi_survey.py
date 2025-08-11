# app.py
import streamlit as st
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
import gspread, datetime as dt
import uuid

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="ROI Weighting Prototype", page_icon="⚡", layout="centered")

PRIMARY = "#02593B"
FONT_URL = "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap"

st.markdown(f"""
<style>
@import url('{FONT_URL}');
html, body, [class*="css"]{{font-family:'Montserrat',sans-serif;background:#FFFFFF;color:#1D1D1B}}
:root{{ --primary:{PRIMARY};}}
/* Sliders */
div[data-baseweb="slider"] > div > div{{background:var(--primary);}}
div[data-baseweb="slider"] > div > div > div{{background:var(--primary);}}
span[data-baseweb="tooltip"]{{background:var(--primary)!important;color:#fff!important;border:none!important}}
div[data-baseweb="slider"] span{{color:#fff!important}}
/* Buttons */
.stButton>button{{background:var(--primary);color:#FFF;border:none;border-radius:10px;padding:.5rem 1rem}}
.stButton>button:disabled{{background:#CCCCCC;color:#666}}
/* Cards */
.block{{border:1px solid #eee;border-radius:14px;padding:16px;margin:6px 0}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:999px;background:#EEF6F0;color:{PRIMARY};font-weight:600}}
.kpi{{background:#f8faf9;border:1px solid #eef3f0;border-radius:12px;padding:12px;text-align:center}}
/* Container */
.main .block-container{{padding-top:1.2rem;max-width:880px;margin:auto}}
/* Hide anchor icons */
a.anchor-link{{display:none}}
/* Center logo */
img[alt="ad_hoc_logo.png"]{{display:block;margin-left:auto;margin-right:auto}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── SESSION STATE ───────────────────────────
def _init_state():
    defaults = {
        "started": False,
        "submitted": False,
        "saving": False,
        "session_id": str(uuid.uuid4()),
        "user": {"nombre":"", "apellido":"", "regulador":"", "pais":"", "email":""},
        "weights_bap": {"FPC":33, "QSD":33, "FEA":34},
        "weights_ahp": {"FPC":33.3, "QSD":33.3, "FEA":33.3, "CR": None},
        "method_choice": "BAP (Presupuesto)",
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

_init_state()

# ─────────────────────────── DATA PERSISTENCE ───────────────────────────
def save_to_sheet(payload: dict):
    """
    Escribe una fila en la primera hoja del Google Sheet.
    Requiere en st.secrets: gs_email, gs_key, sheet_id
    """
    creds = {
        "type": "service_account",
        "client_email": st.secrets.gs_email,
        "private_key": st.secrets.gs_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(Credentials.from_service_account_info(creds, scopes=scope))
    sh = client.open_by_key(st.secrets.sheet_id).sheet1
    sh.append_row([
        dt.datetime.now().isoformat(),
        payload.get("nombre",""),
        payload.get("apellido",""),
        payload.get("regulador",""),
        payload.get("pais",""),
        payload.get("email",""),
        int(round(payload.get("FPC",0))),
        int(round(payload.get("QSD",0))),
        int(round(payload.get("FEA",0))),
        payload.get("method",""),
        payload.get("session_id","")
    ])

# ─────────────────────────── HELPERS ───────────────────────────
CRITERIA = ["FPC","QSD","FEA"]

def show_logo():
    try:
        st.image("ad_hoc_logo.png", width=150)
    except:
        st.markdown(f"<span class='badge'>ROI Prototype</span>", unsafe_allow_html=True)

def stepper(current_idx:int):
    labels = ["Datos", "Presupuesto", "AHP", "Sandbox", "Revisión", "Enviar"]
    cols = st.columns(len(labels))
    for i, c in enumerate(cols):
        with c:
            dot = "🟢" if i <= current_idx else "⚪"
            st.markdown(f"**{dot} {labels[i]}**")

def as_df(weights: dict) -> pd.DataFrame:
    return pd.DataFrame({"Subíndice": CRITERIA, "Peso (%)":[weights["FPC"],weights["QSD"],weights["FEA"]]}).set_index("Subíndice")

def bar(weights: dict, height=220):
    df = as_df(weights).sort_values("Peso (%)", ascending=False)
    st.bar_chart(df, height=height)

def doughnut(weights: dict):
    # simple textual doughnut replacement with KPIs (evita dependencias)
    c1,c2,c3 = st.columns(3)
    for (label,val),cx in zip(weights.items(), [c1,c2,c3]):
        with cx:
            st.markdown(f"<div class='kpi'><div style='font-size:28px;font-weight:700'>{int(round(val))}%</div><div style='opacity:.8'>{label}</div></div>", unsafe_allow_html=True)

def sum_to_100(a,b):
    a = max(0, min(100, a))
    b = max(0, min(100-a, b))
    c = 100 - (a+b)
    return a,b,c

def ahp_weights(r12, r13, r23):
    """
    AHP 3x3 por medias geométricas (aprox. del autovector principal).
    r12 = FPC vs QSD ; r13 = FPC vs FEA ; r23 = QSD vs FEA (escala Saaty, reciprocal matrix)
    """
    A = np.array([
        [1,   r12, r13],
        [1/r12, 1, r23],
        [1/r13, 1/r23, 1]
    ], dtype=float)
    # Pesos por medias geométricas
    gm = np.prod(A, axis=1)**(1/3)
    w = gm/np.sum(gm)
    # lambda_max para CI/CR
    Aw = A.dot(w)
    lam = np.sum(Aw / w) / 3.0
    CI = (lam - 3) / (3 - 1)  # n=3
    RI = 0.58                 # Random Index para n=3
    CR = CI/RI if RI > 0 else None
    return {"FPC": float(w[0]*100), "QSD": float(w[1]*100), "FEA": float(w[2]*100), "CR": float(CR)}

def saaty_select(label_left, label_right, default="Igual (1)"):
    """
    Selector simétrico: 1/9 .. 1/7 .. 1/5 .. 1/3 .. 1 .. 3 .. 5 .. 7 .. 9
    Devuelve el valor >0 a aplicar en la comparación (lado-izq sobre lado-der).
    """
    opts = [
        ("{} ≪ {}".format(label_left,label_right), 1/9),
        ("{} ≪ {}".format(label_left,label_right), 1/7),
        ("{} < {}".format(label_left,label_right), 1/5),
        ("{} < {}".format(label_left,label_right), 1/3),
        ("Igual (1)", 1),
        ("{} > {}".format(label_left,label_right), 3),
        ("{} > {}".format(label_left,label_right), 5),
        ("{} ≫ {}".format(label_left,label_right), 7),
        ("{} ≫ {}".format(label_left,label_right), 9),
    ]
    labels = [t[0] if i!=4 else "Igual (1)" for i,t in enumerate(opts)]
    values = [t[1] for t in opts]
    label = st.select_slider("", options=labels, value=default)
    val = values[labels.index(label)]
    return val, label

def lock_weights(method_name, weights_dict):
    st.session_state.method_choice = method_name
    # redondeo amable y normalización por si acaso
    total = sum([weights_dict[k] for k in CRITERIA])
    if total <= 0: total = 1
    for k in CRITERIA:
        weights_dict[k] = max(0, float(weights_dict[k])) * 100.0/total
    # asigno a sesión por método
    if "BAP" in method_name:
        st.session_state.weights_bap = {k: float(weights_dict[k]) for k in CRITERIA}
    elif "AHP" in method_name:
        st.session_state.weights_ahp = {k: float(weights_dict[k]) for k in CRITERIA} | {"CR": weights_dict.get("CR")}

# ─────────────────────────── PAGES ───────────────────────────
def page_intro():
    show_logo()
    st.title("Regulatory Outcome Index (ROI) – Prototipo de Pesos")
    st.markdown("Completá tus datos para comenzar. Tus preferencias **se visualizan en vivo** y se guardan al enviar.")
    with st.form("intro"):
        c1,c2 = st.columns(2)
        with c1:
            nombre = st.text_input("Nombre")
            regulador = st.text_input("Entidad reguladora")
        with c2:
            apellido = st.text_input("Apellido")
            pais = st.text_input("País")
        email = st.text_input("Email (opcional)")
        consent = st.checkbox("Acepto que se almacenen estos datos con mis ponderaciones de ROI.")
        start = st.form_submit_button("🚀 Empezar", disabled=not(consent and nombre and apellido and regulador and pais))
    if start:
        st.session_state.user = {"nombre":nombre.strip(), "apellido":apellido.strip(), "regulador":regulador.strip(), "pais":pais.strip(), "email":email.strip()}
        st.session_state.started = True
        st.success("Listo. Ya podés pasar a **Presupuesto** en la barra lateral.")

    st.markdown("---")
    with st.expander("ℹ️ ¿Qué mide el ROI?"):
        st.markdown("""
- **FPC** – *Financial Performance & Competitiveness*  
- **QSD** – *Quality of Service Delivery*  
- **FEA** – *Facilitating Electricity Access*
        """)

def page_bap():
    show_logo()
    stepper(1)
    st.header("⚖️ Presupuesto (BAP): repartí **100%** entre los subíndices")
    a = st.slider("FPC – Finanzas & Competitividad", 0, 100, int(round(st.session_state.weights_bap["FPC"])))
    b_max = 100 - a
    b = st.slider("QSD – Calidad del Servicio", 0, b_max, min(int(round(st.session_state.weights_bap["QSD"])), b_max), key=f"qsd_{b_max}")
    a,b,c = sum_to_100(a,b)
    st.info(f"FEA – Acceso se ajusta automáticamente a **{c}%** para cerrar 100%.")
    weights = {"FPC":a, "QSD":b, "FEA":c}
    st.markdown("#### Distribución")
    bar(weights)
    st.markdown("#### Resumen")
    doughnut(weights)
    st.button("✅ Usar estos pesos (BAP)", on_click=lock_weights, args=("BAP (Presupuesto)", weights))

def page_ahp():
    show_logo()
    stepper(2)
    st.header("🔢 AHP: Comparaciones pareadas (escala Saaty)")
    st.caption("Elegí la **intensidad de preferencia** entre pares. Calculamos pesos y **Consistence Ratio (CR)** en vivo.")
    st.markdown("**FPC vs QSD**")
    r12, lbl12 = saaty_select("FPC","QSD", default="Igual (1)")
    st.markdown("**FPC vs FEA**")
    r13, lbl13 = saaty_select("FPC","FEA", default="Igual (1)")
    st.markdown("**QSD vs FEA**")
    r23, lbl23 = saaty_select("QSD","FEA", default="Igual (1)")
    w = ahp_weights(r12, r13, r23)
    st.markdown("#### Pesos resultantes")
    bar({"FPC":w["FPC"], "QSD":w["QSD"], "FEA":w["FEA"]})
    st.markdown(f"**CR (Consistency Ratio):** `{w['CR']:.3f}`  " + ("✅ Aceptable (≤ 0.10)" if w["CR"] <= 0.10 else "⚠️ Revisá tus comparaciones (ideal ≤ 0.10)"))
    st.button("✅ Usar estos pesos (AHP)", disabled=(w["CR"]>0.10), on_click=lock_weights, args=("AHP (Consistente)", w))

def page_sandbox():
    show_logo()
    stepper(3)
    st.header("🧪 Sandbox: probá métodos y visualización en vivo")
    method = st.radio("Método a visualizar", ["BAP (Presupuesto)","AHP (Consistente)","Iguales (33/33/34)"], horizontal=True)
    if method == "BAP (Presupuesto)":
        weights = st.session_state.weights_bap
    elif method == "AHP (Consistente)":
        weights = st.session_state.weights_ahp
    else:
        weights = {"FPC":33, "QSD":33, "FEA":34}
    st.subheader("📊 Barras ordenadas")
    bar(weights, height=260)
    st.subheader("🔘 KPIs rápidos")
    doughnut(weights)

    st.markdown("---")
    st.subheader("✍️ Ajuste fino (opcional)")
    st.caption("Si querés microajustar, mantenemos el total en 100 automáticamente.")
    c1,c2 = st.columns(2)
    with c1:
        fpc = st.number_input("FPC (%)", 0, 100, int(round(weights["FPC"])))
    with c2:
        qsd = st.number_input("QSD (%)", 0, 100-int(round(fpc)), int(round(weights["QSD"])))
    fpc,qsd,fea = sum_to_100(fpc,qsd)
    w2 = {"FPC":fpc,"QSD":qsd,"FEA":fea}
    st.markdown("#### Vista previa ajustada")
    bar(w2)
    if st.button("✅ Fijar estos pesos en mi sesión"):
        lock_weights("BAP (Presupuesto)" if method!="AHP (Consistente)" else "AHP (Consistente)", w2)
        st.success("Pesos actualizados en tu sesión.")

def page_review():
    show_logo()
    stepper(4)
    st.header("📝 Revisión")
    st.markdown("Confirmá tus datos y la distribución final antes de enviar.")
    u = st.session_state.user
    w = st.session_state.weights_ahp if "AHP" in st.session_state.method_choice else st.session_state.weights_bap
    st.markdown("#### Tus datos")
    c1,c2 = st.columns(2)
    with c1:
        st.write(f"**Nombre:** {u['nombre']} {u['apellido']}")
        st.write(f"**Regulador:** {u['regulador']}")
        st.write(f"**Email:** {u['email'] or '—'}")
    with c2:
        st.write(f"**País:** {u['pais']}")
        st.write(f"**Método elegido:** {st.session_state.method_choice}")
    st.markdown("#### Tus pesos")
    bar(w)
    doughnut(w)

def page_submit():
    show_logo()
    stepper(5)
    st.header("📤 Enviar respuesta")
    if st.session_state.submitted:
        st.success("✅ ¡Gracias! Ya registramos tu respuesta para esta sesión.")
        st.caption("Si necesitás enviar de nuevo, recargá la app para una nueva sesión.")
        return
    st.markdown("Al enviar, se guardan tus datos y ponderaciones.")
    w = st.session_state.weights_ahp if "AHP" in st.session_state.method_choice else st.session_state.weights_bap
    disabled = st.session_state.saving or not st.session_state.started
    if st.button("Enviar ahora", disabled=disabled):
        st.session_state.saving = True
        try:
            payload = st.session_state.user | w | {
                "method": st.session_state.method_choice,
                "session_id": st.session_state.session_id
            }
            save_to_sheet(payload)
            st.session_state.submitted = True
            st.success("✅ Thank you for your response. ¡Gracias por tu respuesta!")
            st.balloons()
        except Exception as e:
            st.error("⚠️ Hubo un error al guardar. Verificá la conexión o credenciales y reintentá.")
        finally:
            st.session_state.saving = False

# ─────────────────────────── ROUTER ───────────────────────────
def sidebar_nav():
    st.sidebar.markdown("### 🧭 Navegación")
    pages = {
        "🏁 Inicio": page_intro,
        "⚖️ Presupuesto (BAP)": page_bap,
        "🔢 AHP": page_ahp,
        "🧪 Sandbox": page_sandbox,
        "📝 Revisión": page_review,
        "📤 Enviar": page_submit,
    }
    # bloqueo suave si no comenzó
    default = "🏁 Inicio" if not st.session_state.started else "⚖️ Presupuesto (BAP)"
    choice = st.sidebar.radio("", list(pages.keys()), index=list(pages.keys()).index(default))
    # info de estado
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Sesión:** `{st.session_state.session_id[:8]}`")
    if st.session_state.submitted:
        st.sidebar.success("Respuesta enviada ✅")
    return pages[choice]

# ─────────────────────────── RENDER ───────────────────────────
Page = sidebar_nav()
Page()
