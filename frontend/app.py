import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ----------------- Config -----------------

def obtener_url_api() -> str:
    return os.getenv("API_URL", "http://localhost:8000/ask")


# ----------------- Data helpers -----------------

def solicitar_respuesta(url_api: str, pregunta: str, cantidad_fragmentos: int) -> dict[str, Any]:
    respuesta = requests.post(
        url_api,
        json={"query": pregunta, "top_k": cantidad_fragmentos},
        timeout=120,
    )
    respuesta.raise_for_status()
    cuerpo = respuesta.json()
    if not isinstance(cuerpo, dict):
        raise ValueError("Invalid API response format.")
    return cuerpo


def tabla_fuentes_html(elementos: list[dict[str, Any]]) -> str:
    if not elementos:
        return ""
    filas = []
    for indice, elemento in enumerate(elementos, start=1):
        fuente = str(elemento.get("source", "unknown"))
        pagina = elemento.get("page", "?")
        puntuacion = float(elemento.get("score", 0.0))
        filas.append(
            f"<tr><td>Source {indice}</td><td>{fuente}</td><td>{pagina}</td><td>{puntuacion:.3f}</td></tr>"
        )
    cuerpo = "".join(filas)
    return f"""
    <div class='tabla-fuentes'>
      <div class='tabla-titulo'>Fuentes</div>
      <table>
        <thead><tr><th>Ref</th><th>Archivo</th><th>Página</th><th>Score</th></tr></thead>
        <tbody>{cuerpo}</tbody>
      </table>
    </div>
    """


# ----------------- Estilos globales -----------------

ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --background: #ffffff;
  --foreground: #000000;
  --muted: #f5f5f5;
  --muted-foreground: #4a4a4a;
  --border: #000000;
  --border-light: #d9d9d9;
}

html, body, [class^="st"], .stApp {
  font-family: 'Source Serif 4', Georgia, serif;
  background: var(--background);
  color: var(--foreground);
}

.stApp {
  background:
    repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 3px),
    #fefefe;
}

.main-block {padding: 0 2rem 3rem; max-width: 72rem; margin: 0 auto;}
.hero-wrap {border-bottom: 6px solid var(--foreground); padding: 3rem 0 2rem;}
.hero-title {font-family: 'Playfair Display', Georgia, serif; font-weight: 700; font-size: clamp(3.2rem, 8vw, 6rem); letter-spacing: -0.035em; margin: 0 0 0.75rem; line-height: 0.98;}
.hero-sub {font-size: 1.05rem; color: var(--muted-foreground); margin-top: 0; max-width: 48rem;}
.section-rule {border-top: 4px solid var(--foreground); margin: 2.5rem 0;}

.block-container {padding-top: 0;}
.sidebar .sidebar-content {padding: 2rem 1.5rem 3rem; border-right: 4px solid var(--foreground);}

input[type="text"], textarea, .stTextInput input {
  background: var(--background);
  border: 0;
  border-bottom: 2px solid var(--border);
  border-radius: 0;
  padding: 0.45rem 0.25rem;
  color: var(--foreground);
}
input[type="text"]:focus, textarea:focus, .stTextInput input:focus {
  outline: none;
  border-bottom: 4px solid var(--foreground);
}
textarea {min-height: 62px;}

.stSlider > div {padding: 0 !important;}
.stSlider [role="slider"] {border: 2px solid var(--foreground); box-shadow: none;}
.stSlider [role="slider"]:focus-visible {outline: 3px solid var(--foreground); outline-offset: 3px;}

button[kind="primary"], .stButton > button {
  background: var(--foreground);
  color: var(--background);
  border: 2px solid var(--foreground);
  border-radius: 0;
  padding: 0.8rem 1.6rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  transition: none;
}
button[kind="primary"]:hover, .stButton > button:hover {background: var(--background); color: var(--foreground);}
button:focus-visible {outline: 3px solid var(--foreground) !important; outline-offset: 3px;}

.btn-ghost {background: transparent !important; color: var(--foreground) !important; border: 2px solid var(--foreground) !important; border-radius: 0 !important;}
.btn-ghost:hover {background: var(--foreground) !important; color: var(--background) !important;}

.chat-wrap {display: grid; gap: 1.5rem;}
.stChatMessage {border: 2px solid var(--foreground); border-radius: 0; padding: 1rem; position: relative; background: var(--background);}
.stChatMessage[data-testid="stChatMessage"] {box-shadow: none;}
.stChatMessage.user {background: #ffffff; color: var(--foreground);}
.stChatMessage.assistant {background: var(--foreground); color: var(--background);}
.stChatMessage.assistant code, .stChatMessage.assistant pre {color: var(--background);}
.role-bar {position: absolute; left: -6px; top: 0; bottom: 0; width: 4px; background: var(--foreground);}
.stChatMessage.assistant .role-bar {background: var(--background);}

.tabla-fuentes {margin-top: 1rem; border: 2px solid var(--foreground); padding: 0;}
.tabla-fuentes .tabla-titulo {background: var(--foreground); color: var(--background); padding: 0.5rem 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.85rem; font-weight: 700;}
.tabla-fuentes table {width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace;}
.tabla-fuentes th, .tabla-fuentes td {border: 1px solid var(--foreground); padding: 0.55rem 0.75rem; text-align: left;}
.tabla-fuentes tr:hover {background: var(--foreground); color: var(--background);}

.status-text {font-size: 0.9rem; color: var(--muted-foreground); margin-top: 0.5rem;}
[data-testid="stMarkdownContainer"] p {margin-bottom: 0.6rem;}
[class*="shadow"], .stProgress, .stAlert {box-shadow: none !important;}
</style>
"""

# ----------------- UI -----------------

st.set_page_config(page_title="Buscador semántico RAG", page_icon=":black_large_square:", layout="wide")
st.markdown(ESTILOS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Controles", unsafe_allow_html=True)
    st.text_input("Endpoint API", value=obtener_url_api(), key="api_url")
    cantidad_fragmentos = st.slider("Top K", min_value=1, max_value=20, value=5)

st.markdown("""
<div class='main-block'>
  <div class='hero-wrap'>
    <h1 class='hero-title'>Buscador semántico RAG</h1>
    <p class='hero-sub'>Haz preguntas sobre tus PDFs, obtén respuestas citadas y mantiene el control en un panel limpio.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
st.markdown("<div class='main-block'><div class='section-rule'></div><div class='chat-wrap'>", unsafe_allow_html=True)
for mensaje in st.session_state.messages:
    role = mensaje.get("role", "assistant")
    with st.chat_message(role):
        st.markdown("<div class='role-bar'></div>", unsafe_allow_html=True)
        st.markdown(mensaje["content"])
        tabla_html = tabla_fuentes_html(mensaje.get("sources", []))
        if tabla_html:
            st.markdown(tabla_html, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Input
pregunta_usuario = st.chat_input("Ejemplo: ¿Qué dice el informe sobre inflación?")
if pregunta_usuario:
    st.session_state.messages.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown("<div class='role-bar'></div>", unsafe_allow_html=True)
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant"):
        st.markdown("<div class='role-bar'></div>", unsafe_allow_html=True)
        with st.spinner("Recuperando contexto y generando respuesta..."):
            try:
                cuerpo_respuesta = solicitar_respuesta(
                    url_api=st.session_state.api_url,
                    pregunta=pregunta_usuario,
                    cantidad_fragmentos=cantidad_fragmentos,
                )
                respuesta_textual = str(cuerpo_respuesta.get("answer", "No answer returned."))
                fuentes = cuerpo_respuesta.get("sources", [])
                st.markdown(respuesta_textual)
                tabla_html = tabla_fuentes_html(fuentes)
                if tabla_html:
                    st.markdown(tabla_html, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": respuesta_textual, "sources": fuentes}
                )
            except Exception as exc:
                mensaje_error = f"API request failed: {exc}"
                st.error(mensaje_error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": mensaje_error, "sources": []}
                )

