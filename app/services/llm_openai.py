# app/services/llm_openai.py
from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

# try to import OpenAI client; keep optional for environments without the package or key
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class EmailAIResult(BaseModel):
    category: Literal["Produtivo", "Improdutivo"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str
    suggested_reply: str

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")

def _heuristic_classifier(text: str) -> EmailAIResult:
    productive = [
        "preciso", "solicito", "solicitação", "por favor", "favor",
        "possível", "poderia", "gostaria", "urgente", "prazo", "entrega",
        "reunião", "confirma", "confirmar", "anexo", "corrigir", "ajuda",
        "erro", "problema", "falha", "reembolso", "pagamento", "vencido",
        "orcamento", "cotação", "contrato", "assinatura", "aceite", "aceitar",
    ]

    unproductive = [
        "newsletter", "promoção", "oferta", "parabéns", "congrat",
        "resumo", "relatório", "informe", "divulgação", "evento", "spam",
    ]

    txt = text.lower()
    score = 0
    productive_hits = 0
    unproductive_hits = 0

    for w in productive:
        if w in txt:
            productive_hits += 1
            score += 2 if w in ("urgente", "prazo", "erro", "problema") else 1

    for w in unproductive:
        if w in txt:
            unproductive_hits += 1
            score -= 1

    if score >= 2:
        category = "Produtivo"
        reason = "O conteúdo indica necessidade de ação, resposta ou acompanhamento."
    elif score <= -1:
        category = "Improdutivo"
        reason = "O conteúdo aparenta ser informativo ou promocional, sem exigir ação direta."
    else:
        category = "Improdutivo"
        reason = "Não foram identificados indícios claros de solicitação ou demanda objetiva."

    conf = max(0.55, min(0.9, 0.6 + abs(score) * 0.1))

    suggested = (
        "Olá! Recebemos sua mensagem e estamos analisando a solicitação. Em breve retornaremos."
        if category == "Produtivo"
        else "Mensagem registrada para acompanhamento. Nenhuma ação imediata identificada."
    )

    return EmailAIResult(
        category=category,
        confidence=round(conf, 2),
        short_reason=reason,
        suggested_reply=suggested
    )

def _fallback_parse_error() -> EmailAIResult:
    # Em caso de erro no parsing da resposta da API, optamos por marcar como Improdutivo
    return EmailAIResult(
        category="Improdutivo",
        confidence=0.45,
        short_reason="Falha ao interpretar resposta da API; classificado como improdutivo por segurança.",
        suggested_reply="Obrigado pelo contato. Se precisar de ajuda, por favor, nos envie mais detalhes."
    )

def _parse_openai_json(raw: str) -> EmailAIResult:
    """
    Tenta extrair JSON estrito da resposta do LLM. Aceitamos que o modelo
    possa circundar o JSON com texto — buscamos o primeiro {...} válido.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("Nenhum JSON encontrado na resposta da API")
    payload = match.group(0)
    data = json.loads(payload)
    return EmailAIResult(**data)

def analyse_with_openai(email_text: str) -> EmailAIResult:
    """
    Função de entrada única para classificação. Tenta usar a API OpenAI se disponível
    (variável OPENAI_API_KEY presente e cliente importado), e em caso de erro recorre
    ao classificador heurístico local para garantir funcionamento sem IA.
    """
    text = (email_text or "").strip()
    if not text:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.0,
            short_reason="Texto vazio",
            suggested_reply="Nenhum conteúdo encontrado no email."
        )

    # 1) tentamos a rota OpenAI (se cliente e chave existirem)
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = (
                "Classifique o texto do email abaixo em uma das categorias: 'Produtivo' ou 'Improdutivo'.\n"
                "Retorne apenas um objeto JSON com as chaves: category, confidence, short_reason, suggested_reply.\n"
                "- category: 'Produtivo' ou 'Improdutivo'\n"
                "- confidence: número entre 0.0 e 1.0\n"
                "- short_reason: uma frase curta explicando a decisão\n"
                "- suggested_reply: sugestão de resposta breve\n\n"
                "Email:\n"
                f"{text}\n\n"
                "JSON:"
            )
            # chamada defensiva: SDKs/versões podem variar; usamos responses.create se disponível
            resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=400)
            raw = ""
            if hasattr(resp, "output") and resp.output:
                if isinstance(resp.output, list):
                    for item in resp.output:
                        if isinstance(item, dict) and "content" in item:
                            for c in item["content"]:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    raw += c.get("text", "")
                                elif isinstance(c, str):
                                    raw += c
                else:
                    raw = str(resp.output)
            else:
                raw = str(resp)

            try:
                result = _parse_openai_json(raw)
                # segurança: se confiança muito baixa, combinamos com heurística
                if result.confidence < 0.5:
                    heur = _heuristic_classifier(text)
                    heur.short_reason = f"Resposta da API com baixa confiança ({result.confidence}); combinado com heurística."
                    if heur.confidence >= 0.6 and heur.category == "Produtivo":
                        return heur
                    result.short_reason += " (confiança baixa — revisar)"
                    return result
                return result
            except Exception:
                return _fallback_parse_error()
        except Exception:
            # qualquer falha na API -> fallback heurístico
            return _heuristic_classifier(text)

    # 2) rota offline / fallback
    return _heuristic_classifier(text)
