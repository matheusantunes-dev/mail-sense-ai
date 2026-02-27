# app/services/extract.py
from __future__ import annotations

from typing import Optional
from fastapi import UploadFile
import mimetypes

MAX_CHARS_DEFAULT = 20_000

def _clip(text: str, max_chars: int = MAX_CHARS_DEFAULT) -> str:
    return (text or "").strip()[:max_chars]

async def extract_email_text(raw_text: Optional[str] = None, uploaded_file: UploadFile | None = None, max_chars: int = MAX_CHARS_DEFAULT) -> str:
    """
    Extrai texto do corpo ou de um arquivo enviado.
    - Se raw_text for fornecido (string), retorna ele cortado.
    - Se uploaded_file for fornecido, tenta extrair texto conforme tipo (txt, eml, pdf).
    - Caso não consiga, retorna string vazia.
    """
    if raw_text:
        return _clip(raw_text, max_chars)

    if not uploaded_file:
        return ""

    filename = uploaded_file.filename or ""
    content = await uploaded_file.read()
    content_type = uploaded_file.content_type or mimetypes.guess_type(filename)[0] or ""

    # se for pdf, usar pdfplumber se disponível
    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        try:
            import pdfplumber
            from io import BytesIO
            text_parts = []
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
            return _clip("\n".join(text_parts), max_chars)
        except Exception:
            # não conseguiu extrair pdf -> fallback
            pass

    # se for texto simples ou .eml, decodificar como utf-8 (fallback latin1)
    try:
        text = content.decode("utf-8")
    except Exception:
        try:
            text = content.decode("latin-1")
        except Exception:
            text = ""

    return _clip(text, max_chars)
