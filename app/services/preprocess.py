# app/services/preprocess.py
from __future__ import annotations

import re
import unicodedata

_whitespace_re = re.compile(r"\s+")
_non_word_re = re.compile(r"[^\wÀ-ÿ]+", re.UNICODE)

def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def preprocess_pt(text: str) -> str:
    """
    Pré-processamento simples e robusto sem dependências externas:
    - normaliza para lower
    - remove excesso de whitespace
    - remove caracteres não-palavra mantendo letras acentuadas
    """
    if not text:
        return ""
    s = text.lower()
    s = _whitespace_re.sub(" ", s).strip()
    s = _non_word_re.sub(" ", s)
    s = _whitespace_re.sub(" ", s).strip()
    return s
