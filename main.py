import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from groq import Groq

app = FastAPI(title="Kotoba Content Moderation Service")

openai_client = None
groq_client = None


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

MODERATION_PROMPT = """Eres un moderador de contenido experto para Kotoba, una app de lectura/escritura de historias en español.

## POLÍTICA DE CONTENIDO DE KOTOBA

### CONTENIDO PERMITIDO (NO BANDEREAR):
- Contenido sexual/NSFW entre adultos (consensuado)
- Escenas eróticas, románticas con contenido sexual explícito
- Violencia ficticia en contexto de ficción (guerra, fantasía, acción)
- Lenguaje vulgar o soez en contexto narrativo
- Drogas, alcohol, tabaco en contexto de historia
- Temas oscuros: muerte, trauma, pérdida
- Ficción con personajes que son villanos, criminales, etc.

### CONTENIDO PROHIBIDO (SÍ BANDEREAR):
- Odio hacia grupos protegidos (raza, etnia, religión, orientación sexual, género, discapacidad, nacionalidad)
- Odio disfrazado / codificado:
  * Eufemismos para referirse a grupos (ej: "cucarachas", "plagas", "parásitos" para personas de cierta raza)
  * Dog whistles: lenguaje que suena inocente pero tiene significado de odio oculto
  * Metáforas que deshumanizan grupos de personas
  * Sarcasmo que esconde odio real
- Llamados a la violencia o eliminación de grupos protegidos
- Promoción del odio, discriminación o exclusión de personas por su identidad
- Contenido que promueve ideologías supremacistas, neonazis, o de odio

## REGLAS DE ANÁLISIS
- NO confundas ficción legítima con odio. Un personaje racista en una historia NO es odio si el autor no promueve esas ideas.
- Analiza la INTENCIÓN y el CONTEXTO, no solo palabras sueltas.
- Si hay duda razonable, aprueba el contenido.
- Si el contenido es sexual pero NO contiene odio, responde con categorías vacías.

Responde SOLO con JSON válido:
{
  "is_hate_speech": true/false,
  "confidence": 0.0-1.0,
  "reason": "explicación breve en español",
  "categories": ["ej: racism", "xenophobia", "dehumanization"]
}"""


@app.get("/health")
def health():
    return {"status": "ok", "service": "kotoba-moderation"}


@app.post("/moderate", response_model=ModerationResponse)
def moderate(req: ModerationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

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
    except Exception:
        pass  # If OpenAI fails (rate limit, etc), skip to Layer 2

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

        if is_safe:
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
        triggered = {k: v for k, v in result.categories.items() if v and k in HATE_VIOLENCE_CATEGORIES}
        scores = {k: round(v, 4) for k, v in result.category_scores.items() if v > 0.01}

    analysis_prompt = f"""Un texto fue flaggeado por detección automática. Analízalo considerando la política de Kotoba.

Categorías flaggeadas: {json.dumps(triggered)}
Puntajes: {json.dumps(scores)}
Texto: "{req.text}"

Analiza si es odio real o contenido permitido. Responde SOLO con JSON válido."""

    try:
        chat = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": MODERATION_PROMPT},
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
