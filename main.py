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


MODERATION_PROMPT = """Eres un sistema de moderación de contenido para una aplicación de lectura/escritura de historias en español (Kotoba).

Tu tarea es detectar odio, discriminación o discurso de odio en el texto, incluyendo:

1. **Odio directo**: Racismo, xenofobia, clasismo, sexismo, homofobia, discriminación por discapacidad, etc.
2. **Odio disfrazado / codificado**:
   - Eufemismos para referirse a grupos (ej: "cucarachas", "plagas", "parásitos" para referirse a personas de cierta raza/etnia)
   - Lenguaje de perro (dog whistles) que suena inocente pero tiene significado oculto
   - Metáforas que deshumanizan grupos de personas
   - Sarcasmo que esconde odio real
3. **Violencia contra grupos**: Llamados a la violencia, eliminación o exclusión de grupos protegidos
4. **Deshumanización**: Comparar personas con animales, objetos, enfermedades, etc.

IMPORTANTE:
- NO confundas ficción legítima con odio. Una historia que retrata un personaje racista NO es odio si el autor no promueve esas ideas.
- Analiza la INTENCIÓN y el CONTEXTO, no solo palabras sueltas.
- Si hay duda razonable, aprueba el contenido.
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

    # If not flagged by the fast filter, return clean
    if not result.flagged:
        top_score = max(result.category_scores.values())
        return ModerationResponse(
            flagged=False,
            confidence=round(top_score, 4),
            reason="No se detectó violación de políticas.",
            categories={k: v for k, v in result.categories.items() if v},
        )

    # Layer 2: GPT-4o-mini deep analysis (catches disguised/coded hate speech)
    triggered = {k: v for k, v in result.categories.items() if v}
    scores = {k: round(v, 4) for k, v in result.category_scores.items() if v > 0.01}

    analysis_prompt = f"""Un texto fue flaggeado por detección automática. Analízalo y retorna JSON con:
- "reason": explicación breve (1-2 oraciones)
- "severity": "low", "medium", o "high"
- "categories": qué categorías de odio aplican

Categorías flaggeadas: {json.dumps(triggered)}
Puntajes: {json.dumps(scores)}
Texto: "{req.text}"

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
