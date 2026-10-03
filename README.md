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
