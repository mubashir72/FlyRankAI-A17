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

## Stage 2: prompt-file model integration

The system prompt lives in [prompts/triage-v1.md](prompts/triage-v1.md). It
defines the output contract, low-confidence behavior, and examples. The message
is sent separately as a JSON-encoded user message, keeping untrusted content out
of the system prompt. With `LLM_STUB` unset (or set to `0`), `/triage` calls the
configured OpenAI-compatible provider using temperature `0.2`, a 30-second
timeout, and up to two SDK retries.

Because the local `.env` enables stub mode for safe development, temporarily
turn it off in the server's PowerShell window before launching Uvicorn:

```powershell
$env:LLM_STUB = "0"
uvicorn src.main:app --reload
```

Run these three real-input checks and inspect that the response is a JSON object
with the job-card fields and allowed values:

```powershell
@'
{"text":"I was charged twice for my monthly subscription."}
'@ | curl.exe -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'

@'
{"text":"The app crashes every time I try to sign in."}
'@ | curl.exe -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'

@'
{"text":"Something seems off with my account, but I am not sure what."}
'@ | curl.exe -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'
```

After testing, restore stub mode for local development:

```powershell
Remove-Item Env:LLM_STUB
```

The three live checks returned JSON with the requested four fields. The
duplicate charge was classified as `billing` with normal urgency, the sign-in
crash as `bug` with high urgency, and the ambiguous account report as `other`
with `0.3` confidence. The surprising part was that the vague account message
did not trigger a confident guess: the model followed the low-confidence
fallback rule.

## Stage 3: parse, validate, repair, or quarantine

The API strips JSON code fences and extracts a JSON object from the response,
then validates it against the strict Pydantic output schema. Invalid output gets
exactly one repair call with the original prompt and input, the rejected answer,
and its validation error. If that answer also fails validation, `/triage`
returns HTTP 422 without exposing model text and appends both raw attempts, the
input, error, and prompt version to `logs/quarantine.jsonl`. The quarantine log
is ignored by Git because it may contain user-submitted text.

Verification: a live duplicate-charge request returned a validated `billing`
object. A controlled test then returned an unsupported category twice; the
endpoint made exactly two calls total, returned 422 without either raw answer,
and appended a `v1` record to `logs/quarantine.jsonl`. The invalid model output
was simulated in the test rather than editing the committed prompt.
