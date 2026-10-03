import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APITimeoutError, OpenAI, OpenAIError
from pydantic import ValidationError

from src.llm.schema import Category, TriageInput, TriageOutput, Urgency

load_dotenv()

app = FastAPI(title="Support Triage API")
logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "triage-v1.md"
PROMPT_VERSION = "v1"
QUARANTINE_PATH = Path(__file__).resolve().parent.parent / "logs" / "quarantine.jsonl"


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


def parse_model_output(raw_output: str) -> TriageOutput:
    unfenced_output = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw_output,
        flags=re.IGNORECASE,
    )
    decoder = json.JSONDecoder()
    for start, character in enumerate(unfenced_output):
        if character != "{":
            continue
        try:
            parsed: Any
            parsed, end = decoder.raw_decode(unfenced_output, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        json_object = unfenced_output[start : start + end]
        return TriageOutput.model_validate_json(json_object, strict=True)

    raise ValueError("Response did not contain a valid JSON object.")


def call_model(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
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

    if not response.choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM provider returned no choices.",
        )
    return response.choices[0].message.content or ""


def quarantine_invalid_output(
    input_text: str,
    initial_raw_output: str,
    repaired_raw_output: str,
    validation_error: str,
) -> None:
    record = {
        "input": input_text,
        "initial_raw_output": initial_raw_output,
        "repaired_raw_output": repaired_raw_output,
        "error": validation_error,
        "prompt_version": PROMPT_VERSION,
    }
    try:
        QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUARANTINE_PATH.open("a", encoding="utf-8") as quarantine_file:
            quarantine_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as error:
        logger.exception("Could not write invalid model output to quarantine")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The invalid model response could not be quarantined.",
        ) from error


@app.post("/triage", response_model=TriageOutput)
def triage(payload: TriageInput) -> TriageOutput:
    if os.getenv("LLM_STUB") == "1":
        return TriageOutput(
            category=Category.OTHER,
            urgency=Urgency.NORMAL,
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
    input_message = json.dumps(payload.text, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]

    raw_output = call_model(client, model, messages)
    try:
        return parse_model_output(raw_output)
    except (ValueError, ValidationError) as error:
        first_validation_error = str(error)

    repair_messages = [
        *messages,
        {"role": "assistant", "content": raw_output},
        {
            "role": "user",
            "content": (
                "Your previous answer was rejected for this reason:\n"
                f"{first_validation_error}\n"
                "Return only corrected JSON matching the schema."
            ),
        },
    ]
    repaired_output = call_model(client, model, repair_messages)
    try:
        return parse_model_output(repaired_output)
    except (ValueError, ValidationError) as error:
        final_validation_error = str(error)

    quarantine_invalid_output(
        input_text=payload.text,
        initial_raw_output=raw_output,
        repaired_raw_output=repaired_output,
        validation_error=final_validation_error,
    )
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            "The model response could not be parsed or validated after one repair "
            "attempt. The invalid response was quarantined."
        ),
    )
