# app/services/llm_openai.py
from __future__ import annotations

import json
import os
import re
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
def _fallback_parse_error() -> EmailAIResult:
    return EmailAIResult(
        category="Improdutivo",
        confidence=0.55,
        short_reason="Falha ao interpretar resposta da IA; fallback aplicado.",
        suggested_reply=(
            "Olá! Não consegui entender completamente sua mensagem. "
            "Poderia confirmar o objetivo do e-mail e fornecer mais detalhes?"
        ),
    )


# ==============================
# CLASSIFICADOR LOCAL (SEM IA)
# ==============================
def _local_rule_based_classify(email_text: str) -> EmailAIResult:
    """
    Classificador determinístico para quando a IA não estiver disponível.
    Regras simples, explicitas e conservadoras (prefere marcar como Improdutivo
    quando não houver um pedido claro).
    """
    text = (email_text or "").strip()
    low_text = text.lower()

    # Trivial checks
    if not text or len(low_text) < 5:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.95,
            short_reason="Mensagem vazia ou muito curta.",
            suggested_reply="",
        )

    # Spam / marketing
    spam_keywords = ["unsubscribe", "promoção", "oferta", "compre agora", "newsletter"]
    if any(k in low_text for k in spam_keywords):
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.90,
            short_reason="Indícios fortes de marketing/spam.",
            suggested_reply="",
        )

    # Protocol pattern like #12345 or protocolo 12345
    if re.search(r"#\d{2,}", text) or re.search(r"protocolo[:\s]*\d{2,}", low_text):
        return EmailAIResult(
            category="Produtivo",
            confidence=0.95,
            short_reason="Presença de protocolo identificado (#123 / protocolo 123).",
            suggested_reply=(
                "Obrigado pelo contato. Recebemos sua solicitação e vamos verificar. "
                "Se possível, poderia confirmar melhores horários para contato?"
            ),
        )

    # Words that often indicate a request/action
    request_keywords = [
        "solicit", "pedido", "precis", "gostaria", "poderia", "por favor", "favor",
        "necessito", "encaminhe", "anexo", "enviar", "agendar", "quando", "consulta",
        "abrir chamado", "abrir ticket", "abrir protocolo", "preciso de",
    ]
    has_request = any(k in low_text for k in request_keywords)

    # Support-related with clear action words
    support_words = ["suporte", "erro", "problema", "login", "acesso", "senha", "instal"]
    has_support = any(k in low_text for k in support_words)

    # Complaints / praise
    complaint_words = ["reclam", "insatisfeit", "não satisfaz", "não funciona", "lento"]
    has_complaint = any(k in low_text for k in complaint_words)

    # If there is a clear request or support plus action -> Produtivo
    if has_request:
        # If request words appear but the text is very short and only says "suporte", ask for details
        if has_support and len(low_text.split()) <= 6 and not re.search(r"\d", low_text):
            return EmailAIResult(
                category="Produtivo",
                confidence=0.70,
                short_reason="Pedido de suporte curto e sem detalhes; precisa de mais informações.",
                suggested_reply=(
                    "Olá! Para que possamos ajudar, poderia informar o número do protocolo (se houver), "
                    "o horário do problema e qualquer erro exibido?"
                ),
            )

        return EmailAIResult(
            category="Produtivo",
            confidence=0.90,
            short_reason="Presença de pedido ou ação solicitada.",
            suggested_reply=(
                "Olá! Obrigado pelo contato — vamos verificar sua solicitação. "
                "Poderia, por favor, confirmar o número do protocolo ou enviar mais detalhes?"
            ),
        )

    # Support words without clear request -> be conservative: Improdutivo but suggest clarifying question
    if has_support:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.65,
            short_reason="Menção a suporte sem pedido ou dados suficientes.",
            suggested_reply=(
                "Olá! Vi que mencionou suporte. Para ajudar melhor, poderia descrever o problema e "
                "incluir horários, mensagens de erro ou número do protocolo?"
            ),
        )

    # Complaints without action -> Improdutivo
    if has_complaint:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.80,
            short_reason="Reclamação observada sem pedido de ação clara.",
            suggested_reply=(
                "Lamentamos o transtorno. Para que possamos analisar, poderia informar mais detalhes "
                "e, se possível, o número do protocolo?"
            ),
        )

    # Default conservative choice
    return EmailAIResult(
        category="Improdutivo",
        confidence=0.50,
        short_reason="Sem evidência clara de solicitação; classificado como improdutivo por padrão conservador.",
        suggested_reply="",
    )


# ==============================
# HEURÍSTICA RÁPIDA (ANTES DA IA)
# ==============================
def _quick_heuristic(email_text: str) -> Optional[EmailAIResult]:
    """
    Regras simples e determinísticas.
    Reduz custo, latência e erro de classificação.
    """

    text = (email_text or "").strip().lower()

    # Mensagem vazia ou muito curta
    if len(text) < 5:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.95,
            short_reason="Mensagem vazia ou extremamente curta.",
            suggested_reply="",
        )

    # Indícios claros de spam/marketing
    spam_keywords = ["unsubscribe", "promoção", "oferta", "black friday", "newsletter"]
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
    3. Se não há API key -> usa classificador local
    4. Se há API key -> chama modelo com JSON forçado
    5. Valida com Pydantic
    6. Aplica regra defensiva de confiança
    """

    # 1️⃣ Heurística antes da IA
    heuristic_result = _quick_heuristic(email_text)
    if heuristic_result:
        return heuristic_result

    # 2️⃣ Lê ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 3️⃣ Se não tem chave, usa classificador local determinístico
    if not api_key:
        return _local_rule_based_classify(email_text)

    # 4️⃣ Chamada ao cliente OpenAI
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        # Se o import ou inicialização do cliente falhar, fallback para local
        return _local_rule_based_classify(email_text)

    # 5️⃣ Prompt estruturado e hierárquico
    user_prompt = f\"\"\"Classifique o e-mail conforme as regras:

- Produtivo -> existe pedido claro, solicitação objetiva, protocolo (#123), necessidade de suporte com ação concreta.
- Improdutivo -> spam, propaganda, reclamação vaga sem pedido específico, mensagem genérica sem ação necessária.

IMPORTANTE:
1. A palavra 'suporte' sozinha NÃO torna produtivo.
2. Só classifique como Produtivo se houver necessidade real de resposta.
3. Se faltar informação para executar a ação, continue classificando como Produtivo, mas peça os dados necessários.

Responda APENAS com JSON válido:

{{
  \"category\": \"Produtivo\" ou \"Improdutivo\",
  \"confidence\": número entre 0 e 1,
  \"short_reason\": \"Justificativa objetiva em 1 frase.\",
  \"suggested_reply\": \"Resposta curta e profissional.\"
}}

E-mail:
{email_text}
\"\"\".strip()

    try:
        # 6️⃣ Chamada com JSON forçado
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,  # Mais determinístico
            response_format={\"type\": \"json_object\"},
            messages=[
                {\"role\": \"system\", \"content\": _SYSTEM},
                {\"role\": \"user\", \"content\": user_prompt},
            ],
        )

        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)

        result = EmailAIResult.model_validate(data)

        # 7️⃣ Regra defensiva de confiança
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
