import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import APITimeoutError, OpenAI, OpenAIError

from src.llm.schema import TriageInput, TriageOutput

load_dotenv()

app = FastAPI(title="Support Triage API")
logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "triage-v1.md"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    error = exc.errors()[0]
    location = error.get("loc", ())
    field = (
        str(location[-1])
        if len(location) > 1
        and location[0] == "body"
        and isinstance(location[-1], str)
        else "text"
    )
    message = error.get("msg", "Invalid value")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"field": field, "message": f"Invalid field '{field}': {message}"},
    )


@app.post("/triage", response_model=None)
def triage(payload: TriageInput) -> TriageOutput | PlainTextResponse:
    if os.getenv("LLM_STUB") == "1":
        return TriageOutput(
            category="other",
            urgency="normal",
            confidence=0.5,
            reason="Stub response for local testing.",
        )

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not base_url or not api_key or not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL to enable model calls.",
        )

    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as error:
        logger.exception("Could not load system prompt from %s", PROMPT_PATH)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The triage system prompt could not be loaded.",
        ) from error

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=2)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload.text)},
            ],
            temperature=0.2,
        )
    except APITimeoutError as error:
        logger.exception("The LLM provider request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM provider request timed out.",
        ) from error
    except OpenAIError as error:
        logger.exception("The LLM provider request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM provider request failed.",
        ) from error

    content = response.choices[0].message.content
    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM provider returned an empty response.",
        )
    return PlainTextResponse(content=content)
