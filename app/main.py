# app/main.py
from __future__ import annotations

import os
import sys
import traceback
from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.extract import extract_email_text
from app.services.llm_openai import analyse_with_openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MailSense AI")

# Mount static directory relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

if not os.path.isdir(static_dir):
    # evitar erro no deploy se caminho errado
    static_dir = os.path.join(os.getcwd(), "app", "static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Página inicial com formulário simples."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyse", response_class=HTMLResponse)
async def analyse(request: Request, email_text: str = Form(None), file: UploadFile | None = File(None)):
    """
    Endpoint principal:
    - aceita campo de texto 'email_text' ou arquivo 'file'
    - extrai texto via extract_email_text
    - chama analyse_with_openai (que internamente faz fallback heurístico)
    - devolve template com resultado
    """
    try:
        email_raw = await extract_email_text(raw_text=email_text, uploaded_file=file)
        if not email_raw:
            return templates.TemplateResponse(
                "result.html",
                {"request": request, "error": "Nenhum conteúdo extraído do email.", "result": None, "email_preview": ""},
            )
        result = analyse_with_openai(email_raw)
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": None, "result": result, "email_preview": email_raw[:1500]},
        )
    except Exception as exc:
        # registrar stacktrace nos logs (importante para debug no Vercel)
        tb = traceback.format_exc()
        print("Erro em /analyse:", tb, file=sys.stderr)
        # retornar resposta amigável sem crashar a função
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": "Erro interno ao processar (veja logs).", "result": None, "email_preview": ""},
        )

# rota healthcheck simples
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
