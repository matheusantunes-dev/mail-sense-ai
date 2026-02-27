# app/services/llm_openai.py
from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# pydantic not strictly required here, mas mantemos para validação simples em pontos críticos
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------
# Modelos de retorno
# -----------------------
class EmailAIResult(BaseModel):
    """
    Contrato de retorno usado pela aplicação.
    - category: Produtivo / Improdutivo
    - confidence: float 0.0-1.0
    - short_reason: texto para logs/debug (curto)
    - user_message: texto pronto para enviar/exibir ao cliente
    - used_strategy: string que informa se foi 'openai', 'rules' ou 'fallback'
    - matched_intents: lista de intents detectadas (útil para logs)
    """
    category: Literal["Produtivo", "Improdutivo"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str
    user_message: str
    used_strategy: str
    matched_intents: List[str] = Field(default_factory=list)


# -----------------------
# Engine de regras escalável
# -----------------------
@dataclass
class Rule:
    """
    Representa uma regra: nome, padrões (regex ou keywords), prioridade e meta-dados.
    priority: maior número = maior prioridade (executa primeiro).
    intent: nome lógico (ex: support, billing).
    category_override: opcional, força 'Produtivo'/'Improdutivo'.
    """
    name: str
    patterns: List[str]  # regex patterns (re.IGNORECASE)
    priority: int
    intent: str
    category_override: Optional[str] = None
    min_hits: int = 1  # quantas ocorrências precisam bater para acionar a regra
    meta: Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """
    Motor simples que aplica regras por prioridade e retorna intents encontradas.
    Arquitetura pensada para ser substituída por DB + admin UI no futuro.
    """
    def __init__(self, rules: List[Rule]):
        # ordenar regras por prioridade decrescente facilita encontrar intents críticas primeiro
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        # compila regex para performance
        self._compiled: List[Tuple[Rule, List[re.Pattern]]] = [
            (r, [re.compile(p, re.IGNORECASE | re.UNICODE) for p in r.patterns]) for r in self.rules
        ]

    def analyze(self, text: str) -> List[Tuple[Rule, int]]:
        """
        Retorna lista de (Rule, hits) que casaram com o texto.
        hits = número de correspondências encontradas (soma simples).
        """
        results: List[Tuple[Rule, int]] = []
        for rule, patterns in self._compiled:
            hits = 0
            for pat in patterns:
                matches = pat.findall(text)
                if matches:
                    hits += len(matches)
            if hits >= rule.min_hits:
                results.append((rule, hits))
        return results


# -----------------------
# Templates de resposta
# -----------------------
# Cada intent pode ter várias variações de template (formal, friendly, concise).
# Use chaves {…} para inserir dados.
TEMPLATES: Dict[str, Dict[str, str]] = {
    "support": {
        "friendly": "Olá! Obrigado por avisar — já abrimos um chamado para o time de suporte. Você pode nos enviar mais detalhes (prints ou passos para reproduzir)?",
        "formal": "Prezado(a), agradecemos o contato. Abriremos um chamado com o setor de suporte. Favor informar passos para reprodução e anexos relevantes.",
        "concise": "Chamado de suporte criado. Envie mais detalhes, por favor."
    },
    "billing": {
        "friendly": "Oi! Vi que é sobre cobrança — vou encaminhar para financeiro e voltamos com um retorno em breve. Pode enviar o comprovante se tiver?",
        "formal": "Recebemos sua solicitação referente à cobrança. Encaminharemos ao departamento financeiro. Solicitamos o envio de comprovantes quando aplicável.",
        "concise": "Encaminhado ao financeiro. Envie comprovante, se houver."
    },
    "meeting": {
        "friendly": "Ótimo — parece que você quer agendar/confirmar uma reunião. Quais são suas disponibilidades?",
        "formal": "Entendemos que deseja agendar uma reunião. Por favor, informe datas e horários de preferência.",
        "concise": "Precisa agendar reunião? Informe disponibilidade."
    },
    "marketing": {
        "friendly": "Obrigado pelo envio — registramos como material informativo/marketing. Nada de ação imediata necessária.",
        "formal": "Registramos o conteúdo como informativo/marketing. Sem ação requerida no momento.",
        "concise": "Material informativo recebido."
    },
    "attachment": {
        "friendly": "Notamos que você mencionou um anexo. Não chegamos a receber — poderia reenviar o arquivo?",
        "formal": "Foi mencionada a presença de anexo, porém não encontramos o mesmo. Solicitamos o reenvio do arquivo.",
        "concise": "Anexo não encontrado. Reenvie, por favor."
    },
    "general": {
        "friendly": "Obrigado pela mensagem! Vamos analisar e retornar assim que possível.",
        "formal": "Agradecemos o contato. Analisaremos e daremos retorno em breve.",
        "concise": "Mensagem recebida. Retornaremos em breve."
    }
}


# -----------------------
# Regras iniciais (extendíveis)
# -----------------------
DEFAULT_RULES: List[Rule] = [
    Rule(
        name="urgent_support",
        patterns=[r"\burgent", r"\burgente\b", r"\bimediato", r"\bcrítico\b", r"erro grave"],
        priority=100,
        intent="support",
        category_override="Produtivo",
    ),
    Rule(
        name="support_generic",
        patterns=[r"\bsuport", r"\bsuporte\b", r"\bajuda\b", r"\bproblema\b", r"\bbug\b", r"\berro\b"],
        priority=90,
        intent="support",
        category_override="Produtivo",
    ),
    Rule(
        name="billing_words",
        patterns=[r"\bfatura\b", r"\bpagamento\b", r"\bboleto\b", r"\breembolso\b", r"\bnota fiscal\b", r"\bchargeback\b"],
        priority=80,
        intent="billing",
        category_override="Produtivo",
    ),
    Rule(
        name="meeting_words",
        patterns=[r"\breuni.{1,3}\b", r"\bagendar\b", r"\bagendamento\b", r"\bdisponibilid"],
        priority=70,
        intent="meeting",
        category_override="Produtivo",
    ),
    Rule(
        name="attachment_words",
        patterns=[r"\banexo\b", r"\banexado\b", r"\banexos\b", r"\bsegue anexo\b", r"\barquivo em anexo\b"],
        priority=60,
        intent="attachment",
        category_override=None,  # não força produtivo — apenas flag
    ),
    Rule(
        name="marketing_words",
        patterns=[r"\bnewsletter\b", r"\bpromoç", r"\boferta\b", r"\bdivulgação\b", r"\bevento\b"],
        priority=30,
        intent="marketing",
        category_override="Improdutivo",
    ),
]


# instantiate engine (ponto único para trocar por DB no futuro)
_RULE_ENGINE = RuleEngine(DEFAULT_RULES)


# -----------------------
# Utilitários de template / tom
# -----------------------
def detect_tone(text: str) -> str:
    """
    Decide tom com base em pistas de linguagem.
    - frases muito formais -> 'formal'
    - saudação curta, emoticons -> 'friendly'
    - caso contrário -> 'concise' se for curto ou 'friendly' por default
    """
    text = (text or "").strip()
    lower = text.lower()
    # pistas de formalidade
    if re.search(r"\b(prezado|senhor|senhora|atenciosamente|cordialmente)\b", lower):
        return "formal"
    if re.search(r"^\s*(bom dia|boa tarde|boa noite)\b", lower):
        return "friendly"
    # se o texto for curto -> conciso
    if len(text) < 80:
        return "concise"
    return "friendly"


def render_template(intent: str, tone: str, extras: Optional[Dict[str, str]] = None) -> str:
    """
    Renderiza template com substituições simples.
    extras permite enviar dados como {name:.., date:..}, se quisermos enriquecer.
    """
    extras = extras or {}
    templates_for_intent = TEMPLATES.get(intent) or TEMPLATES["general"]
    template = templates_for_intent.get(tone) or templates_for_intent.get("friendly")
    # substitui chaves simples
    try:
        return template.format(**extras)
    except Exception:
        return template


# -----------------------
# Heurística leve (fallback)
# -----------------------
def simple_confidence_from_hits(hits: int, max_hits: int = 5) -> float:
    """
    Mapear hits (0..n) para confiança simples [0.5..0.95].
    """
    ratio = min(hits, max_hits) / max_hits
    return round(0.5 + ratio * 0.45, 2)


# -----------------------
# OpenAI integration (opcional) — tentativa, mas não obrigatória
# -----------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")


def _try_openai_classify(text: str) -> Optional[EmailAIResult]:
    """
    Chama OpenAI para obter uma classificação mais sofisticada.
    Se falhar por qualquer motivo, retorna None para que possamos usar regras locais.
    NOTA: mantemos a mesma estrutura de JSON esperada (category, confidence, short_reason, suggested_reply).
    """
    if not OPENAI_API_KEY or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Classifique o texto abaixo em category ('Produtivo' ou 'Improdutivo'), "
            "confidence (0.0-1.0), short_reason (uma frase), suggested_reply (mensagem curta).\n\n"
            "Retorne apenas um objeto JSON.\n\n"
            f"Email:\n{text}\n\nJSON:"
        )
        resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=300)
        # extrair texto da resposta (método defensivo porque SDKs mudam)
        raw = ""
        if hasattr(resp, "output") and resp.output:
            # tenta varrer estrutura
            if isinstance(resp.output, list):
                for item in resp.output:
                    if isinstance(item, dict) and "content" in item:
                        for c in item["content"]:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                raw += c.get("text", "")
                            elif isinstance(c, str):
                                raw += c
                    elif isinstance(item, str):
                        raw += item
            else:
                raw = str(resp.output)
        else:
            raw = str(resp)
        # extrai JSON da resposta
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        payload = json.loads(match.group(0))
        return EmailAIResult(
            category=payload.get("category", "Improdutivo"),
            confidence=float(payload.get("confidence", 0.5)),
            short_reason=payload.get("short_reason", ""),
            user_message=payload.get("suggested_reply", ""),
            used_strategy="openai",
            matched_intents=[]
        )
    except Exception:
        return None


# -----------------------
# Função pública principal
# -----------------------
def analyse_with_openai(email_text: str) -> EmailAIResult:
    """
    Pipeline principal:
    1) tenta OpenAI (se disponível) e valida resposta
    2) se falha, usa RuleEngine para detectar intents
    3) combina resultado e retorna EmailAIResult consistente
    """
    text = (email_text or "").strip()
    if not text:
        return EmailAIResult(
            category="Improdutivo",
            confidence=0.0,
            short_reason="Texto vazio",
            user_message="Nenhum conteúdo encontrado no email.",
            used_strategy="none",
            matched_intents=[]
        )

    # 1) tentar OpenAI (se configurado)
    ai_result = _try_openai_classify(text)
    if ai_result is not None:
        # se confiança alta, aceitamos; se baixa, combinamos com regras
        if ai_result.confidence >= 0.65:
            ai_result.used_strategy = "openai"
            return ai_result
        # se baixa confiança, deixamos passar para motor de regras para cross-check
        # mas preservamos a sugestão do LLM como fallback parcial
        openai_fallback = ai_result

    else:
        openai_fallback = None

    # 2) regras locais (determinísticas)
    rule_hits = _RULE_ENGINE.analyze(text)
    matched_intents = [r.intent for r, hits in rule_hits]

    # soma hits para cada intent (para gerar confiança)
    intent_hits_map: Dict[str, int] = {}
    for r, hits in rule_hits:
        intent_hits_map[r.intent] = intent_hits_map.get(r.intent, 0) + hits

    # escolher intent com maior prioridade/hits
    if rule_hits:
        # a lista já foi filtrada por prioridade; escolher a regra com maior prioridade e hits
        best_rule, best_hits = rule_hits[0]
        # se houver regras múltiplas com diferentes intents, priorizamos por priority (já ordenado)
        intent = best_rule.intent
        hits = intent_hits_map.get(intent, best_hits)
        confidence = simple_confidence_from_hits(hits)
        # category: considerar override da regra ou heurística padrão
        category = best_rule.category_override or ("Produtivo" if confidence >= 0.6 else "Improdutivo")
        tone = detect_tone(text)
        message = render_template(intent, tone)
        short_reason = f"Regra '{best_rule.name}' acionada (hits={hits})."
        return EmailAIResult(
            category=category,
            confidence=confidence,
            short_reason=short_reason,
            user_message=message,
            used_strategy="rules",
            matched_intents=matched_intents
        )

    # 3) se nenhuma regra casou: usar heurística leve (palavras genéricas)
    # isso evita respostas vazias/tecnicalidades no front
    # heurística simples: procurar verbos de solicitação
    if re.search(r"\b(preciso|gostaria|poderia|solicito|favor|por favor|pode me)\b", text, re.IGNORECASE):
        # pode ser produtivo mesmo sem regra específica
        message = render_template("general", detect_tone(text))
        return EmailAIResult(
            category="Produtivo",
            confidence=0.6,
            short_reason="Heurística simples detectou linguagem de solicitação.",
            user_message=message,
            used_strategy="heuristic",
            matched_intents=[]
        )

    # 4) fallback final: OpenAI (se tinha output) ou improdutivo genérico
    if openai_fallback:
        # mistura: manter a suggested_reply do LLM, mas ajustar a forma
        return EmailAIResult(
            category=openai_fallback.category,
            confidence=openai_fallback.confidence,
            short_reason="Resposta do LLM com baixa confiança; usada como fallback.",
            user_message=openai_fallback.user_message or render_template("general", detect_tone(text)),
            used_strategy="openai-fallback",
            matched_intents=[]
        )

    # default final
    return EmailAIResult(
        category="Improdutivo",
        confidence=0.55,
        short_reason="Nenhuma regra ou padrão de solicitação detectado.",
        user_message=render_template("general", detect_tone(text)),
        used_strategy="fallback",
        matched_intents=[]
    )


# -----------------------
# API de extensão (runtime)
# -----------------------
def add_rule(rule: Rule):
    """
    Interface para adicionar regra em runtime (útil para testes ou painel administrativo).
    Observação: atualmente adiciona apenas na instância em memória; para persistir, salve em DB.
    """
    _RULE_ENGINE.rules.append(rule)
    # recompila engine (simples e robusto para MVP)
    _RULE_ENGINE.__init__(_RULE_ENGINE.rules)


def add_template(intent: str, tone: str, template_text: str):
    """
    Adiciona/atualiza template para intent+tone.
    """
    if intent not in TEMPLATES:
        TEMPLATES[intent] = {}
    TEMPLATES[intent][tone] = template_text
