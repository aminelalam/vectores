import os
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI


def check_openai() -> None:
    clave_api = os.getenv("OPENAI_API_KEY", "").strip()
    modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not clave_api:
        raise ValueError("OPENAI_API_KEY is missing.")

    cliente = OpenAI(api_key=clave_api)
    respuesta = cliente.chat.completions.create(
        model=modelo,
        temperature=0,
        messages=[{"role": "user", "content": "Responde solo: ok"}],
        timeout=30,
    )
    texto = (respuesta.choices[0].message.content or "").strip()
    print(f"OpenAI OK. Model={modelo}. Response={texto}")


def check_ollama() -> None:
    url_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    modelo = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
    respuesta = requests.get(f"{url_base}/api/tags", timeout=15)
    respuesta.raise_for_status()
    datos = respuesta.json()
    modelos_disponibles = [elemento.get("name", "") for elemento in datos.get("models", [])]
    if modelo and not any(nombre.startswith(modelo) for nombre in modelos_disponibles):
        print(f"Ollama reachable, but model '{modelo}' not found. Available: {modelos_disponibles}")
    else:
        print(f"Ollama OK. Model={modelo}")


def main() -> int:
    load_dotenv()
    proveedor = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    try:
        if proveedor == "openai":
            check_openai()
        elif proveedor == "ollama":
            check_ollama()
        else:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'ollama'.")
        return 0
    except Exception as exc:
        print(f"LLM check failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
