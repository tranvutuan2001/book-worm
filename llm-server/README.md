# Multi-Provider LLM Server

A minimalist, strictly-typed LLM server using FastAPI and dependency-injector.

## Features
- **Provider Agnostic**: Swap between OpenAI, Anthropic, and local MLX models via configuration.
- **Strictly Typed**: Full Pydantic and Python Protocol support.
- **Formal DI**: Managed by `dependency-injector`.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your keys and preferred backend
   ```

3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Usage

### Generate Completion
`POST /generate`
```json
{
  "messages": [
    {"role": "user", "content": "Tell me a joke about robots."}
  ],
  "max_tokens": 100
}
```

## Testing
```bash
pytest
```
