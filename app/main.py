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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🔹 Função interna reutilizável (boa prática)
async def _handle_analyse(request: Request, email_text: str | None, file: UploadFile | None):
    try:
        email_raw = await extract_email_text(raw_text=email_text, uploaded_file=file)

        if not email_raw:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "error": "Nenhum conteúdo extraído do email.",
                    "result": None,
                    "email_preview": "",
                },
            )

        result = analyse_with_openai(email_raw)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": None,
                "result": result,
                "email_preview": email_raw[:1500],
            },
        )

    except Exception:
        tb = traceback.format_exc()
        print("Erro em análise:", tb, file=sys.stderr)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": "Erro interno ao processar (veja logs).",
                "result": None,
                "email_preview": "",
            },
        )


# 🔹 Aceita as duas rotas (analyze e analyse)
@app.post("/analyse", response_class=HTMLResponse)
async def analyse_br(
    request: Request,
    email_text: str = Form(None),
    file: UploadFile | None = File(None),
):
    return await _handle_analyse(request, email_text, file)


@app.post("/analyze", response_class=HTMLResponse)
async def analyse_us(
    request: Request,
    email_text: str = Form(None),
    file: UploadFile | None = File(None),
):
    return await _handle_analyse(request, email_text, file)


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
