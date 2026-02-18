# Simple FastAPI project

Quick start instructions using your existing virtual environment.

1. Activate your virtualenv (example if it's `.venv`):

```zsh
source .venv/bin/activate
```

Replace the path above with the path to your existing environment if different.

2. Install requirements:

```zsh
pip install -r requirements.txt
```

3. Run the app (from project root):

```zsh
uvicorn main:app --reload
```

4. Open the Swagger UI at: `http://127.0.0.1:8000/docs`

Using the LM Studio proxy
-------------------------

This project exposes a small proxy endpoint at `POST /ask` which will forward a question
to a locally running LM Studio (or other local LLM server).


Example JSON body (only `question` is required):

```json
{ "question": "What is the capital of France?" }
```

Example `curl` (the API uses fixed local LM Studio host/port/model from env vars or defaults):

```zsh
curl -X POST "http://127.0.0.1:8000/ask" \
	-H "Content-Type: application/json" \
	-d '{"question":"Who wrote \"Pride and Prejudice\"?"}'
```

Notes:
 - The client will first try an OpenAI-compatible `/v1/chat/completions` endpoint, then
	 fall back to a generic `/api/generate` text-generation endpoint if needed.
 - Host/port/model are read from environment variables `LM_STUDIO_HOST`, `LM_STUDIO_PORT`,
	 and `LM_STUDIO_MODEL`. Defaults are `127.0.0.1:8080` and `llama-2-7b-chat`.
