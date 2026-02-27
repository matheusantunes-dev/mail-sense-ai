from __future__ import annotations

import json
import os
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# ==============================
# CONTRATO DE RESPOSTA
# ==============================
class EmailAIResult(BaseModel):
    """
    Modelo fixo que garante consistência no retorno para o resto do sistema.
    Nunca confiamos direto na IA — sempre validamos aqui.
    """

    category: Literal["Produtivo", "Improdutivo"]
    confidence: float = Field(ge=0.0, le=1.0)
    short_reason: str
    suggested_reply: str


# ==============================
# CONFIGURAÇÃO DO SISTEMA
# ==============================
_SYSTEM = (
    "Você é um assistente corporativo do setor financeiro responsável "
    "por classificar e-mails como Produtivo ou Improdutivo e sugerir resposta objetiva."
)


# ==============================
# FALLBACKS SEGUROS
# ==============================
def _fallback_no_key() -> EmailAIResult:
    return EmailAIResult(
        category="Produtivo",
        confidence=0.50,
        short_reason="API key não configurada; utilizando fallback seguro.",
        suggested_reply=(
            "Olá! Poderia detalhar melhor sua solicitação ou informar o número do protocolo, por favor?"
        ),
    )


def _fallback_parse_error() -> EmailAIResult:
    return EmailAIResult(
        category="Produtivo",
        confidence=0.55,
        short_reason="Falha ao interpretar resposta da IA; fallback aplicado.",
        suggested_reply=(
            "Olá! Para prosseguir, poderia confirmar o objetivo do e-mail "
            "e informar o número do protocolo, se houver?"
        ),
    )


# ==============================
# HEURÍSTICA RÁPIDA (ANTES DA IA)
# ==============================
def _quick_heuristic(email_text: str) -> Optional[EmailAIResult]:
    """
    Regras simples e determinísticas.
    Reduz custo, latência e erro de classificação.
    """

    text = email_text.strip().lower()

    # Mensagem vazia ou muito curta
    if len(text) < 5:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.95,
            short_reason="Mensagem vazia ou extremamente curta.",
            suggested_reply="",
        )

    # Indícios claros de spam/marketing
    spam_keywords = ["unsubscribe", "promoção", "oferta", "black friday"]

    if any(word in text for word in spam_keywords):
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.90,
            short_reason="Indícios de spam ou conteúdo promocional.",
            suggested_reply="",
        )

    return None


# ==============================
# FUNÇÃO PRINCIPAL
# ==============================
def analyse_with_openai(email_text: str) -> EmailAIResult:
    """
    Fluxo robusto de classificação:

    1. Aplica heurísticas locais
    2. Valida API key
    3. Chama modelo com JSON forçado
    4. Valida com Pydantic
    5. Aplica regra defensiva de confiança
    """

    # 1️⃣ Heurística antes da IA
    heuristic_result = _quick_heuristic(email_text)
    if heuristic_result:
        return heuristic_result

    # 2️⃣ Lê ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return _fallback_no_key()

    client = OpenAI(api_key=api_key)

    # 3️⃣ Prompt estruturado e hierárquico
    user_prompt = f"""
Classifique o e-mail conforme as regras:

- Produtivo → existe pedido claro, solicitação objetiva, protocolo (#123),
  necessidade de suporte com ação concreta.
- Improdutivo → spam, propaganda, reclamação vaga sem pedido específico,
  mensagem genérica sem ação necessária.

IMPORTANTE:
1. A palavra "suporte" sozinha NÃO torna produtivo.
2. Só classifique como Produtivo se houver necessidade real de resposta.
3. Se faltar informação para executar a ação, continue classificando como Produtivo,
   mas peça os dados necessários.

Responda APENAS com JSON válido:

{{
  "category": "Produtivo" ou "Improdutivo",
  "confidence": número entre 0 e 1,
  "short_reason": "Justificativa objetiva em 1 frase.",
  "suggested_reply": "Resposta curta e profissional."
}}

E-mail:
{email_text}
""".strip()

    try:
        # 4️⃣ Chamada com JSON forçado
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,  # Mais determinístico
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)

        result = EmailAIResult.model_validate(data)

        # 5️⃣ Regra defensiva de confiança
        if result.confidence < 0.60:
            return EmailAIResult(
                category="Produtivo",
                confidence=result.confidence,
                short_reason="Confiança baixa na classificação; tratado como produtivo por segurança.",
                suggested_reply=(
                    "Olá! Poderia detalhar melhor sua solicitação ou informar o número do protocolo?"
                ),
            )

        return result

    except (json.JSONDecodeError, ValidationError, KeyError, IndexError):
        return _fallback_parse_error()
    except Exception:
        return _fallback_parse_error()
