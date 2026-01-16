from __future__ import annotations

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class EmailAIResult(BaseModel):
    """
    Contrato (formato fixo) do que o backend devolve para o resto do sistema.
    Isso garante consistência, mesmo se a IA responder algo estranho.
    """

    category: Literal["Produtivo", "Improdutivo"]
    confidence: float = Field(ge=0.0, le=1.0)
    short_reason: str = Field(description="Justificativa curta (1 frase).")
    suggested_reply: str = Field(description="Resposta pronta para enviar.")


_SYSTEM = (
    "Você é um assistente corporativo de triagem de e-mails do setor financeiro. "
    "Classifique e gere resposta curta, educada e objetiva. "
    "Se faltar informação, peça de forma direta (sem enrolar)."
)


def _fallback_no_key() -> EmailAIResult:
    """
    Fallback usado quando não existe OPENAI_API_KEY.
    Isso evita erro 500 e deixa a demo funcionando no navegador.
    """
    return EmailAIResult(
        category="Produtivo",
        confidence=0.50,
        short_reason="Chave de API não configurada; usando fallback.",
        suggested_reply=(
            "Olá! Para dar sequência, poderia informar mais detalhes do pedido "
            "ou o número do protocolo, por favor?"
        ),
    )


def _fallback_parse_error() -> EmailAIResult:
    """
    Fallback usado quando a IA respondeu algo que não deu para interpretar/validar.
    """
    return EmailAIResult(
        category="Produtivo",
        confidence=0.55,
        short_reason="Não foi possível interpretar a saída da IA; usando fallback.",
        suggested_reply=(
            "Olá! Para prosseguir, poderia confirmar o objetivo do e-mail e "
            "informar o número do protocolo (se houver), por favor?"
        ),
    )


def analyse_with_openai(email_text: str) -> EmailAIResult:
    """
    1) Lê a API key do ambiente
    2) Chama OpenAI via Chat Completions
    3) Pede resposta em JSON
    4) Faz parse/validação com Pydantic
    5) Se der ruim, usa fallback seguro
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Se não tem chave, não tenta chamar a OpenAI (evita erro 401/500).
    if not api_key:
        return _fallback_no_key()

    # Cria o cliente OpenAI usando a chave do .env
    client = OpenAI(api_key=api_key)

    # A gente pede para a IA responder ESTRITAMENTE como JSON.
    # Isso facilita parse e deixa o sistema mais robusto.
    user_prompt = f"""
Você vai analisar o e-mail abaixo.
Também, caso houver algo sobre suporte, é classificado como "Produtivo", senão é "Improdutivo",
caso houver um email curto, sobre suporte, sem informações de horários, protocolos, classifique como "Produtivo" e
dê uma resposta breve perguntando horário, protocolo, etc, caso não houver estas informações no email.
Responda APENAS com um JSON válido (sem texto fora do JSON), seguindo este formato:
{{
  "category": "Produtivo" ou "Improdutivo",
  "confidence": número entre 0 e 1,
  "short_reason": "1 Resposta tamanho médio, com no final agradecimentos e o tempo de aguardo de  até 24h",
  "suggested_reply": "resposta curta, educada e pronta para enviar"
}}

E-mail:
{email_text}
""".strip()

    try:
        # Chamada ao endpoint "chat.completions" (compatível com o SDK 1.30.1)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Pega o texto retornado pelo modelo
        content = completion.choices[0].message.content or ""

        # Converte o JSON (string) em dict Python
        data = json.loads(content)

        # Valida e normaliza no nosso modelo (garante tipo/limites/valores)
        return EmailAIResult.model_validate(data)

    except (json.JSONDecodeError, ValidationError, KeyError, IndexError):
        # JSON inválido, campos faltando, ou formato inesperado
        return _fallback_parse_error()
    except Exception:
        # Qualquer erro inesperado (rede, timeout, etc.)
        return _fallback_parse_error()
