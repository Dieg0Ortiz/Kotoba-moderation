import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Kotoba Content Moderation Service")
client = None


def get_client():
    global client
    if client is None:
        client = OpenAI()
    return client


class ModerationRequest(BaseModel):
    text: str


class ModerationResponse(BaseModel):
    flagged: bool
    confidence: float
    reason: str
    categories: dict[str, bool]


MODERATION_PROMPT = """Eres un sistema de moderación de contenido para Kotoba, una aplicación de lectura/escritura de historias en español.

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
  * Eufemismos para referirse a grupos (ej: "cucarachas", "plagas", "parásitos" para referirse a personas de cierta raza/etnia)
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
- Si el contenido es sexual pero NO contiene odio, responde "flagged: false".
- Piensa en español ya que el contenido es en español.

Responde SOLO con JSON válido:
{
  "reason": "explicación breve en español",
  "severity": "low/medium/high",
  "categories": ["ej: racism", "xenophobia", "dehumanization", etc.]
}"""


@app.get("/health")
def health():
    return {"status": "ok", "service": "kotoba-moderation"}


@app.post("/moderate", response_model=ModerationResponse)
def moderate(req: ModerationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Layer 1: OpenAI Moderation API (free, instant)
    mod = get_client().moderations.create(
        model="omni-moderation-latest",
        input=req.text,
    )
    result = mod.results[0]

    # Ignore 'sexual' and 'sexual/minors' categories — Kotoba allows NSFW content.
    # Only flag hate/violence/harassment categories.
    HATE_VIOLENCE_CATEGORIES = [
        'hate', 'hate/threatening', 'harassment', 'harassment/threatening',
        'violence', 'violence/graphic', 'self-harm', 'self-harm/intent',
        'self-harm/instructions', 'illicit', 'illicit/violent',
    ]
    filtered_flagged = any(result.categories.get(cat) for cat in HATE_VIOLENCE_CATEGORIES)

    # If not flagged by the fast filter, return clean
    if not filtered_flagged:
        top_score = max(result.category_scores.values())
        return ModerationResponse(
            flagged=False,
            confidence=round(top_score, 4),
            reason="No se detectó violación de políticas.",
            categories={k: v for k, v in result.categories.items() if v},
        )

    # Layer 2: GPT-4o-mini deep analysis (catches disguised/coded hate speech)
    triggered = {k: v for k, v in result.categories.items() if v and k in HATE_VIOLENCE_CATEGORIES}
    scores = {k: round(v, 4) for k, v in result.category_scores.items() if v > 0.01}

    analysis_prompt = f"""Un texto fue flaggeado por detección automática. Analízalo considerando la política de Kotoba:

POLÍTICA DE KOTOBA:
- CONTENIDO PERMITIDO: Sexual/NSFW, violencia ficticia, lenguaje vulgar, temas oscuros
- CONTENIDO PROHIBIDO: Odio, discriminación, dehumanización de grupos protegidos

Analiza el texto y retorna JSON con:
- "reason": explicación breve (1-2 oraciones)
- "severity": "low", "medium", o "high"
- "categories": qué categorías de odio aplican (SOLO si es odio real)

Categorías flaggeadas: {json.dumps(triggered)}
Puntajes: {json.dumps(scores)}
Texto: "{req.text}"

Si el contenido es sexual pero NO contiene odio, retorna "reason" explicando que es contenido permitido y las categorías vacías.
Retorna SOLO JSON válido."""

    try:
        chat = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown."},
                {"role": "user", "content": analysis_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        deep = json.loads(chat.choices[0].message.content)
    except Exception:
        deep = {"reason": "Flaggeado por moderación automática.", "severity": "medium"}

    max_score = max(result.category_scores.values())

    return ModerationResponse(
        flagged=True,
        confidence=round(max_score, 4),
        reason=deep.get("reason", "Flaggeado por moderación automática."),
        categories=triggered,
    )
