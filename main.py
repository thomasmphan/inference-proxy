import logging
import time

from anthropic import APIError, Anthropic
from anthropic.types import Usage
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
client = Anthropic()


MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}

# Cost per million tokens (in USD)
COST_PER_MILLION_TOKENS = {
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
}

# Caching for exact input messages
cache = {}


class ChatRequest(BaseModel):
    message: str
    model: str = "haiku"


def calculate_cost(model_id: str, usage: Usage) -> float:
    costs = COST_PER_MILLION_TOKENS[model_id]
    estimated_cost_usd = (
        usage.input_tokens / 1_000_000 * costs["input"]
        + usage.output_tokens / 1_000_000 * costs["output"]
    )
    return estimated_cost_usd


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {"models": list(MODELS.keys())}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:

    model_id = MODELS.get(request.model)
    if model_id is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{request.model}'. Available: {list(MODELS.keys())}")

    if (request.message, model_id) in cache:
        logger.info(f"Cache hit model={request.model}")
        return StreamingResponse(
            iter([cache[(request.message, model_id)]]),
            media_type="text/plain",
            headers={"X-Cache": "HIT"},
        )

    def generate():
        start = time.time()
        try:
            with client.messages.stream(
                model=model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": request.message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

                final_message = stream.get_final_message()
                cache[(request.message, model_id)] = final_message.content[0].text
                usage = final_message.usage
                cost = round(calculate_cost(model_id, usage), 6)
                duration_ms = round((time.time() - start) * 1000)

                logger.info(
                    f"stream response model={request.model} "
                    f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens} "
                    f"cost_usd={cost} duration_ms={duration_ms}"
                )

                yield (
                    f"\n\n[input tokens: {usage.input_tokens}, "
                    f"output tokens: {usage.output_tokens}, "
                    f"estimated cost: ${cost}]"
                )
        except APIError as e:
            logger.error(f"upstream error model={request.model} error={e}")
            raise HTTPException(status_code=503, detail=str(e))

    return StreamingResponse(generate(), media_type="text/plain", headers={"X-Cache": "MISS"})


@app.post("/chat")
def chat(request: ChatRequest) -> dict:

    model_id = MODELS.get(request.model)
    if model_id is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{request.model}'. Available: {list(MODELS.keys())}")

    if (request.message, model_id) in cache:
        logger.info(f"Cache hit model={request.model}")
        return {
            "response": cache[(request.message, model_id)],
            "model": request.model,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "estimated_cost_usd": 0.0,
            "cached": True,
        }

    start = time.time()
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": request.message}],
        )
        cache[(request.message, model_id)] = response.content[0].text
    except APIError as e:
        logger.error(f"upstream error model={request.model} error={e}")
        raise HTTPException(status_code=503, detail=str(e))
    
    cost = round(calculate_cost(model_id, response.usage), 6)
    duration_ms = round((time.time() - start) * 1000)
    logger.info(
        f"chat response model={request.model} "
        f"input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens} "
        f"cost_usd={cost} duration_ms={duration_ms}"
    )

    return {
        "response": response.content[0].text,
        "model": request.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "estimated_cost_usd": cost,
        "cached": False,
    }
