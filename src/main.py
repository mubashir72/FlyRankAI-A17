import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APIStatusError, APITimeoutError, OpenAI, OpenAIError
from pydantic import ValidationError

from src.llm.schema import Category, TriageInput, TriageOutput, Urgency

load_dotenv()

app = FastAPI(title="Support Triage API")
logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "triage-v1.md"
PROMPT_VERSION = "v1"
QUARANTINE_PATH = Path(__file__).resolve().parent.parent / "logs" / "quarantine.jsonl"
MAX_PROVIDER_RETRIES = 3
RETRY_JITTER_SECONDS = 0.25


@dataclass(frozen=True)
class ModelCallResult:
    content: str
    input_tokens: int
    output_tokens: int
    duration_ms: int


def log_model_call(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    duration_ms: int,
    needed_repair: bool | None,
    outcome: str,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "llm_call",
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": duration_ms,
                "needed_repair": needed_repair,
                "outcome": outcome,
            },
            separators=(",", ":"),
        )
    )


def retry_after_seconds(error: APIStatusError) -> float | None:
    retry_after = error.response.headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (retry_at - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            logger.warning("Ignoring invalid Retry-After header from provider")
            return None


def is_retryable_status(error: APIStatusError) -> bool:
    return error.status_code == 429 or error.status_code >= 500


def retry_delay(retry_number: int, error: APIStatusError | None = None) -> float:
    if error is not None and error.status_code == 429:
        provider_delay = retry_after_seconds(error)
        if provider_delay is not None:
            return provider_delay
    return (2 ** (retry_number - 1)) + random.uniform(0.0, RETRY_JITTER_SECONDS)


def log_completed_model_call(
    result: ModelCallResult, model: str, needed_repair: bool
) -> None:
    log_model_call(
        model,
        result.input_tokens,
        result.output_tokens,
        result.duration_ms,
        needed_repair,
        "success",
    )


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


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    is_repair: bool,
) -> ModelCallResult:
    retries = 0
    while True:
        attempt_started_at = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            break
        except APITimeoutError as error:
            attempt_duration_ms = round(
                (time.perf_counter() - attempt_started_at) * 1000
            )
            if retries < MAX_PROVIDER_RETRIES:
                retries += 1
                delay = (2 ** (retries - 1)) + random.uniform(
                    0.0, RETRY_JITTER_SECONDS
                )
                log_model_call(
                    model,
                    None,
                    None,
                    attempt_duration_ms,
                    None,
                    "retry_timeout",
                )
                logger.warning(
                    "LLM provider timeout; retrying attempt %s/%s after %.2fs",
                    retries,
                    MAX_PROVIDER_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            log_model_call(
                model,
                None,
                None,
                attempt_duration_ms,
                True if is_repair else None,
                "timeout",
            )
            logger.exception("The LLM provider request timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The LLM provider request timed out after retries.",
            ) from error
        except APIStatusError as error:
            attempt_duration_ms = round(
                (time.perf_counter() - attempt_started_at) * 1000
            )
            if is_retryable_status(error) and retries < MAX_PROVIDER_RETRIES:
                retries += 1
                delay = retry_delay(retries, error)
                log_model_call(
                    model,
                    None,
                    None,
                    attempt_duration_ms,
                    None,
                    f"retry_http_{error.status_code}",
                )
                logger.warning(
                    "LLM provider returned HTTP %s; retrying attempt %s/%s after %.2fs",
                    error.status_code,
                    retries,
                    MAX_PROVIDER_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue

            outcome = f"http_{error.status_code}"
            log_model_call(
                model,
                None,
                None,
                attempt_duration_ms,
                True if is_repair else None,
                outcome,
            )
            if error.status_code in {400, 401, 403}:
                logger.warning("LLM provider rejected the request with HTTP %s", error.status_code)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        f"The LLM provider rejected the request with HTTP "
                        f"{error.status_code}; check the model configuration and API key."
                    ),
                ) from error
            if error.status_code == 429:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The LLM provider rate limit persisted after retries.",
                ) from error
            logger.exception("The LLM provider request failed with HTTP %s", error.status_code)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The LLM provider request failed after retries.",
            ) from error
        except OpenAIError as error:
            log_model_call(
                model,
                None,
                None,
                round((time.perf_counter() - attempt_started_at) * 1000),
                True if is_repair else None,
                "provider_error",
            )
            logger.exception("The LLM provider request failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The LLM provider request failed.",
            ) from error

    duration_ms = round((time.perf_counter() - attempt_started_at) * 1000)
    usage = getattr(response, "usage", None)
    content = response.choices[0].message.content if response.choices else ""

    return ModelCallResult(
        content=content or "",
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        duration_ms=duration_ms,
    )


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
    if os.getenv("LLM_ENABLED", "true").strip().lower() == "false":
        return fallback_response()

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

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
    input_message = json.dumps(payload.text, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]

    initial_call = call_model(client, model, messages, is_repair=False)
    try:
        parsed_response = parse_model_output(initial_call.content)
    except (ValueError, ValidationError) as error:
        first_validation_error = str(error)
        log_completed_model_call(initial_call, model, needed_repair=True)
    else:
        log_completed_model_call(initial_call, model, needed_repair=False)
        return parsed_response

    repair_messages = [
        *messages,
        {"role": "assistant", "content": initial_call.content},
        {
            "role": "user",
            "content": (
                "Your previous answer was rejected for this reason:\n"
                f"{first_validation_error}\n"
                "Return only corrected JSON matching the schema."
            ),
        },
    ]
    repair_call = call_model(client, model, repair_messages, is_repair=True)
    try:
        parsed_response = parse_model_output(repair_call.content)
    except (ValueError, ValidationError) as error:
        final_validation_error = str(error)
    else:
        log_completed_model_call(repair_call, model, needed_repair=True)
        return parsed_response

    log_completed_model_call(repair_call, model, needed_repair=True)
    quarantine_invalid_output(
        input_text=payload.text,
        initial_raw_output=initial_call.content,
        repaired_raw_output=repair_call.content,
        validation_error=final_validation_error,
    )
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            "The model response could not be parsed or validated after one repair "
            "attempt. The invalid response was quarantined."
        ),
    )


def fallback_response() -> TriageOutput:
    return TriageOutput(
        category=Category.OTHER,
        urgency=Urgency.NORMAL,
        confidence=0.0,
        reason="Automated triage is temporarily disabled.",
    )
