# Week 6 — LLM integration

This project starts with a single support-message triage job. Its input and closed
output contract are recorded in [JOB-CARD.md](JOB-CARD.md).

The provider is configured only through environment variables. Changing the base
URL and model lets the same OpenAI-compatible client use a local model or a hosted
provider without hard-coding provider details.

## Stage 0: verify the model connection

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` if needed. Set `LLM_API_KEY` in `.env` to your
   Groq API key. The default `LLM_BASE_URL` is
   `https://api.groq.com/openai/v1` and `LLM_MODEL` is `openai/gpt-oss-20b`.

3. Run the one-shot connection check:

   ```powershell
   python src\llm\hello.py
   ```

The `.env` file is ignored by Git; never commit your API key.

## Stage 1: validated triage endpoint (stub mode)

Install dependencies as above, then start the API with `LLM_STUB=1` in `.env`:

```powershell
uvicorn src.main:app --reload
```

In another PowerShell window, send a valid request (piping the JSON avoids
PowerShell's native-command quoting differences):

```powershell
@'
{"text":"I was charged twice for my subscription."}
'@ | curl.exe -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'
```

The response has the closed fields `category`, `urgency`, `confidence`, and
`reason`. The stub skips all model calls. Invalid input is rejected with HTTP
400 and a JSON message naming the field, before any model call could occur.
For example, this deliberately omits the required `text` field:

```powershell
'{}' | curl.exe -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'
```

The endpoint currently returns HTTP 503 when stub mode is disabled; the real
model integration is a later stage.
