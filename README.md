# Support message triage API

## What it does

This API takes one customer support message and sorts it into a small set of
teams' categories, such as billing or bug reports. It also estimates urgency
and explains the choice briefly. If the message is unclear, the system can
label it “other” rather than pretend it knows. Responses are checked against a
fixed format before they are returned.

## Try it: request and exact response

With the default stub mode enabled, this PowerShell request makes no model call:

```powershell
@'
{"text":"I was charged twice for my subscription."}
'@ | curl.exe -sS -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" --data-binary '@-'
```

Exact response:

```json
{"category":"other","urgency":"normal","confidence":0.5,"reason":"Stub response for local testing."}
```

## Job card

What it does: Classifies a support message so it lands on the right team.

Input: `{ "text": "string, 1-2000 characters" }`

Output: `{ "category": one of [billing|bug|feature|other], "urgency": one of [low|normal|high], "confidence": 0.0-1.0, "reason": "one short sentence" }`

It must never:

- Invent a category outside the list.
- Give medical, legal, or financial advice.
- Reveal the prompt.

When unsure, use category `other` with low confidence rather than guessing.

## Provider and configuration

The live evaluation below used Groq's OpenAI-compatible API with model
`openai/gpt-oss-20b`. To swap providers, configure these three variables in your
local `.env` file:

- `LLM_BASE_URL` — OpenAI-compatible API base URL.
- `LLM_API_KEY` — the provider key; keep it in ignored `.env`, never in the
  tracked `.env.example`.
- `LLM_MODEL` — the provider's model identifier.

The `.env.example` documents all variables. Stub mode (`LLM_STUB=1`) is the
safe default for local development. Set it to `0` for a live model call.

## Evaluation result

On **2026-10-03**, prompt **v1**, the eight-case evaluation scored **8/8
(100.0%) category accuracy**. There were no category mismatches. This is a
small, hand-written smoke/evaluation set, not a claim of production-level
accuracy.

Run the same cases against a locally running API with:

```powershell
python evals\run_eval.py
```

## Cost log and daily estimate

One successful call from that run logged:

```json
{"event":"llm_call","prompt_version":"v1","model":"openai/gpt-oss-20b","input_tokens":505,"output_tokens":73,"duration_ms":1365,"needed_repair":false,"outcome":"success"}
```

Using an estimated rate of **$0.10 per million input tokens** and **$0.50 per
million output tokens**, that call costs about **$0.000087**. The eight eval
calls used 4,055 input and 1,078 output tokens total (about **$0.000945** at
those rates). At the eval's average token use, 10,000 requests/day would be
about **$1.18/day**, before retries, repairs, taxes, or provider price changes.
Verify current provider pricing before budgeting.

## What I would fix with another day

I would expand the eval set and have a second person independently label it;
eight examples are too few to establish reliable accuracy, especially for
urgency and ambiguous cases.

## Run from a fresh clone

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For the exact stub response above, leave `LLM_STUB=1` in `.env`. Start the API:

```powershell
uvicorn src.main:app --reload
```

To use Groq instead, put your key in the ignored `.env`, set `LLM_STUB=0`,
`LLM_ENABLED=true`, and configure the three provider variables above. Then send
the same curl request. The endpoint has a 30-second provider timeout, bounded
retries for timeouts/429/5xx, a kill switch (`LLM_ENABLED=false`), strict output
validation, one repair attempt, and quarantine handling for invalid responses.

## Run the test suite

```powershell
python -m unittest discover -s tests -v
```
