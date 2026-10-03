import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.llm.schema import TriageInput, TriageOutput

load_dotenv()

app = FastAPI(title="Support Triage API")


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


@app.post("/triage", response_model=TriageOutput)
def triage(payload: TriageInput) -> TriageOutput:
    if os.getenv("LLM_STUB") == "1":
        return TriageOutput(
            category="other",
            urgency="normal",
            confidence=0.5,
            reason="Stub response for local testing.",
        )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model calls are not implemented yet; set LLM_STUB=1 for local testing.",
    )
