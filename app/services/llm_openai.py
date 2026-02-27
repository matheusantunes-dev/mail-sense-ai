# app/services/llm_openai.py
from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

# Import do SDK OpenAI feito de forma defensiva:
# se o pacote não existir ou houver erro, OpenAI ficará como None
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # tipo: ignore
    _OPENAI_AVAILABLE = False

from pydantic import BaseModel, Field, ValidationError


# ==============================
# CONTRATO DE RESPOSTA
# ==============================
class EmailAIResult(BaseModel):
    """
    Modelo fixo que garante consistência no retorno para o resto do sistema.
    """
    category: Literal["Produtivo", "Improdutivo"]
    confidence: float = Field(ge=0.0, le=1.0)
    short_reason: str
    suggested_reply: str


# ==============================
# MENSAGENS / SISTEMA
# ==============================
_SYSTEM = (
    "Você é um assistente corporativo do setor financeiro responsável "
    "por classificar e-mails como Produtivo ou Improdutivo e sugerir resposta objetiva."
)


# ==============================
# FALLBACKS SEGUROS
# ==============================
def _fallback_parse_error() -> EmailAIResult:
    """
    Quando a IA falha em retornar JSON ou algo inesperado acontece,
    somos conservadores: classificamos como Improdutivo e pedimos mais info.
    """
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
    Regras conservadoras para reduzir falsos-positivos (improdutivas marcadas
    como produtivas).
    """
    text = (email_text or "").strip()
    low_text = text.lower()

    # Mensagem vazia ou muito curta -> improdutivo
    if not text or len(low_text) < 5:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.95,
            short_reason="Mensagem vazia ou muito curta.",
            suggested_reply="",
        )

    # Spam / marketing detectado
    spam_keywords = ["unsubscribe", "promoção", "oferta", "compre agora", "newsletter", "black friday"]
    if any(k in low_text for k in spam_keywords):
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.90,
            short_reason="Indícios fortes de marketing/spam.",
            suggested_reply="",
        )

    # Protocol pattern like #12345 or "protocolo 12345"
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

    # Palavras que indicam pedido/ação
    request_keywords = [
        "solicit", "pedido", "precis", "gostaria", "poderia", "por favor", "favor",
        "necessito", "encaminhe", "anexo", "enviar", "agendar", "quando", "consulta",
        "abrir chamado", "abrir ticket", "abrir protocolo", "preciso de", "alterar",
        "cancelar", "reembolso", "cobrança", "pagamento"
    ]
    has_request = any(k in low_text for k in request_keywords)

    # Palavras de suporte técnico que precisam de contexto para serem produtivas
    support_words = ["suporte", "erro", "problema", "login", "acesso", "senha", "instal", "bug"]
    has_support = any(k in low_text for k in support_words)

    # Palavras que indicam reclamação sem pedido de ação
    complaint_words = ["reclam", "insatisfeit", "não satisfaz", "não funciona", "lento", "demora"]
    has_complaint = any(k in low_text for k in complaint_words)

    # Se há pedido claro -> Produtivo
    if has_request:
        # pedido curto com só "suporte" e sem números -> pedir detalhes
        if has_support and len(low_text.split()) <= 6 and not re.search(r"\d", low_text):
            return EmailAIResult(
                category="Produtivo",
                confidence=0.70,
                short_reason="Pedido de suporte curto e sem detalhes; precisa de mais informações.",
                suggested_reply=(
                    "Olá! Para que possamos ajudar, poderia informar o número do protocolo (se houver), "
                    "o horário do problema e qualquer mensagem de erro?"
                ),
            )

        return EmailAIResult(
            category="Produtivo",
            confidence=0.90,
            short_reason="Presença de pedido ou ação solicitada.",
            suggested_reply=(
                "Olá! Obrigado pelo contato — vamos verificar sua solicitação. "
                "Poderia confirmar o número do protocolo ou enviar mais detalhes?"
            ),
        )

    # Menção a suporte sem pedido claro -> improdutivo, mas pede informação
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

    # Reclamação sem pedido de ação -> improdutivo (sugere informações)
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

    # Default conservador
    return EmailAIResult(
        category="Improdutivo",
        confidence=0.50,
        short_reason="Sem evidência clara de solicitação; classificado como improdutivo por padrão conservador.",
        suggested_reply="",
    )


# ==============================
# HEURÍSTICA RÁPIDA (antes de tudo)
# ==============================
def _quick_heuristic(email_text: str) -> Optional[EmailAIResult]:
    """
    Regras bem rápidas para evitar chamadas desnecessárias.
    """
    text = (email_text or "").strip().lower()

    if len(text) < 5:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.95,
            short_reason="Mensagem vazia ou extremamente curta.",
            suggested_reply="",
        )

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
    Fluxo:
      1) heurística rápida
      2) se não tem chave ou SDK -> classificador local
      3) se tem OpenAI -> chamada ao modelo
      4) valida com Pydantic e aplica regra defensiva de confiança
    """
    # 1) heurística rápida
    heuristic_result = _quick_heuristic(email_text)
    if heuristic_result:
        return heuristic_result

    # 2) verifica ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 3) se SDK não disponível ou não há chave, usa classificador local determinístico
    if not _OPENAI_AVAILABLE or not api_key:
        return _local_rule_based_classify(email_text)

    # 4) tenta inicializar cliente de forma local (dentro da função)
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        # Se houver qualquer erro na inicialização do cliente, fallback para local
        return _local_rule_based_classify(email_text)

    # 5) prompt estruturado
    user_prompt = f\"\"\"Classifique o e-mail conforme as regras:

- Produtivo -> existe pedido claro, solicitação objetiva, protocolo (#123), necessidade de suporte com ação concreta.
- Improdutivo -> spam, propaganda, reclamação vaga sem pedido específico, mensagem genérica sem ação necessária.

IMPORTANTE:
1. A palavra 'suporte' sozinha NÃO torna produtivo.
2. Só classifique como Produtivo se houver necessidade real de resposta.
3. Se faltar informação para executar a ação, continue classificando como Produtivo, mas peça os dados necessários.

Responda APENAS com JSON válido:

{{
  "category": "Produtivo" ou "Improdutivo",
  "confidence": número entre 0 e 1,
  "short_reason": "Justificativa objetiva em 1 frase.",
  "suggested_reply": "Resposta curta e profissional."
}}

E-mail:
{email_text}
\"\"\".strip()

    try:
        # 6) Chamada com JSON forçado (se suportado pelo SDK/model)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        result = EmailAIResult.model_validate(data)

        # 7) Regra defensiva de confiança
        if result.confidence < 0.60:
            # comportamento de negócio: confiança baixa -> pedimos mais detalhes
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
