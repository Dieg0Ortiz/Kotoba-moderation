import json
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from groq import Groq

app = FastAPI(title="Kotoba Content Moderation Service")

openai_client = None
groq_client = None

BACKEND_URL = os.getenv("BACKEND_URL", "https://kotoba-back-production.up.railway.app")


def get_openai():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client


def get_groq():
    global groq_client
    if groq_client is None:
        groq_client = Groq()
    return groq_client


class ModerationRequest(BaseModel):
    text: str
    work_id: Optional[str] = None


class ModerationResponse(BaseModel):
    flagged: bool
    confidence: float
    reason: str
    categories: dict[str, bool]


HATE_VIOLENCE_CATEGORIES = [
    'hate', 'hate/threatening', 'harassment', 'harassment/threatening',
    'violence', 'violence/graphic', 'self-harm', 'self-harm/intent',
    'self-harm/instructions', 'illicit', 'illicit/violent',
]

MATURE_PROMPT = """Eres un moderador de contenido experto para Kotoba, una app de lectura/escritura de historias en español.

Esta historia está clasificada como CONTENIDO MADURO. El autor marcó que contiene material para adultos.

## POLÍTICA PARA CONTENIDO MADURO

### CONTENIDO PERMITIDO (NO BANDEREAR):
- Contenido sexual/NSFW entre adultos (consensuado)
- Escenas eróticas, románticas con contenido sexual explícito
- Violencia gráfica en contexto de ficción
- Lenguaje vulgar o soez
- Drogas, alcohol, tabaco
- Temas oscuros: muerte, trauma, pérdida

### CONTENIDO PROHIBIDO (SÍ BANDEREAR):
- Odio hacia grupos protegidos (raza, etnia, religión, orientación sexual, género, discapacidad, nacionalidad)
- Odio disfrazado / codificado (eufemismos, dog whistles, deshumanización)
- Llamados a la violencia o eliminación de grupos protegidos
- Promoción del odio, discriminación o exclusión de personas por su identidad
- Ideologías supremacistas, neonazis, o de odio

### REGLAS
- NO confundas ficción legítima con odio.
- Analiza la INTENCIÓN y el CONTEXTO.
- Si hay duda razonable, aprueba el contenido.

Responde SOLO con JSON válido:
{
  "is_hate_speech": true/false,
  "confidence": 0.0-1.0,
  "reason": "explicación breve en español",
  "categories": ["ej: racism", "xenophobia", "dehumanization"]
}"""

NOT_MATURE_PROMPT = """Eres un moderador de contenido experto para Kotoba, una app de lectura/escritura de historias en español.

Esta historia NO está clasificada como contenido maduro. El autor NO marcó que contiene material para adultos.

## POLÍTICA PARA CONTENIDO NO MADURO

### CONTENIDO PERMITIDO (NO BANDEREAR):
- Violencia ficticia leve en contexto de ficción
- Lenguaje vulgar moderado
- Temas oscuros: muerte, trauma, pérdida (sin explícito)
- Ficción con personajes que son villanos, criminales, etc.

### CONTENIDO PROHIBIDO (SÍ BANDEREAR):
- Cualquier contenido sexual explícito o NSFW
- Escenas eróticas o románticas con contenido sexual
- Nudidad explícita
- Odio hacia grupos protegidos
- Odio disfrazado / codificado
- Llamados a la violencia o eliminación de grupos protegidos
- Promoción del odio, discriminación o exclusión
- Ideologías supremacistas, neonazis, o de odio

### REGLAS
- Si el contenido es sexual y la historia NO es madura, flagged: true con razón "Contenido sexual no permitido en historias no clasificadas como maduras".
- NO confundas ficción legítima con odio.
- Analiza la INTENCIÓN y el CONTEXTO.
- Si hay duda razonable, aprueba el contenido (excepto contenido sexual).

Responde SOLO con JSON válido:
{
  "is_hate_speech": true/false,
  "confidence": 0.0-1.0,
  "reason": "explicación breve en español",
  "categories": ["ej: sexual_content", "racism", "xenophobia", "dehumanization"]
}"""


def get_work_maturity(work_id: str) -> bool:
    """Check if a work is marked as mature via the backend API."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{BACKEND_URL}/api/works/{work_id}")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("is_mature", False)
    except Exception:
        pass
    return False


@app.get("/health")
def health():
    return {"status": "ok", "service": "kotoba-moderation"}


@app.post("/moderate", response_model=ModerationResponse)
def moderate(req: ModerationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Determine if the work is mature
    is_mature = False
    if req.work_id:
        is_mature = get_work_maturity(req.work_id)

    # Select the right prompt based on maturity
    system_prompt = MATURE_PROMPT if is_mature else NOT_MATURE_PROMPT

    # ── Layer 1: OpenAI Moderation API (free, instant) ──
    filtered_flagged = False
    openai_succeeded = False
    result = None
    try:
        mod = get_openai().moderations.create(
            model="omni-moderation-latest",
            input=req.text,
        )
        result = mod.results[0]
        openai_succeeded = True
        filtered_flagged = any(result.categories.get(cat) for cat in HATE_VIOLENCE_CATEGORIES)

        # For non-mature works, also flag sexual content
        if not is_mature and not filtered_flagged:
            filtered_flagged = result.categories.get("sexual", False) or result.categories.get("sexual/minors", False)
    except Exception:
        pass

    # If OpenAI succeeded and didn't flag, we're done
    if openai_succeeded and not filtered_flagged:
        return ModerationResponse(
            flagged=False,
            confidence=0.5,
            reason="No se detectó violación de políticas.",
            categories={},
        )

    # ── Layer 2: Groq Llama-Guard-4 (free, fast moderation) ──
    try:
        guard = get_groq().chat.completions.create(
            model="meta-llama/Llama-Guard-4-12B",
            messages=[{"role": "user", "content": req.text}],
        )
        guard_response = guard.choices[0].message.content.strip()
        is_safe = guard_response.lower().startswith("safe")

        if is_safe and openai_succeeded:
            return ModerationResponse(
                flagged=False,
                confidence=0.8,
                reason="Llama-Guard determinó que el contenido es seguro.",
                categories={},
            )
    except Exception:
        pass

    # ── Layer 3: Groq Llama 3.3 70B (free, deep analysis) ──
    triggered = {}
    scores = {}
    if result is not None:
        all_flagged = {k: v for k, v in result.categories.items() if v}
        triggered = all_flagged
        scores = {k: round(v, 4) for k, v in result.category_scores.items() if v > 0.01}

    maturity_label = "MADURO" if is_mature else "NO MADURO"
    analysis_prompt = f"""Un texto fue flaggeado por detección automática.
Historia clasificada como: {maturity_label}

Categorías flaggeadas: {json.dumps(triggered)}
Puntajes: {json.dumps(scores)}
Texto: "{req.text}"

Analiza si es contenido prohibido considerando la clasificación de madurez. Responde SOLO con JSON válido."""

    try:
        chat = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        deep = json.loads(chat.choices[0].message.content)
    except Exception:
        deep = {"is_hate_speech": True, "reason": "Flaggeado por moderación automática.", "confidence": 0.7}

    is_hate = deep.get("is_hate_speech", True)
    confidence = deep.get("confidence", 0.7)
    reason = deep.get("reason", "Análisis automático.")
    categories = deep.get("categories", list(triggered.keys()))

    if isinstance(categories, list):
        categories = {c: True for c in categories}
    elif not isinstance(categories, dict):
        categories = {c: True for c in triggered.keys()}

    return ModerationResponse(
        flagged=is_hate,
        confidence=round(confidence, 4) if isinstance(confidence, (int, float)) else 0.7,
        reason=reason,
        categories=categories if is_hate else {},
    )
