# LLM Server

A FastAPI-based server for serving local large language models using llama-cpp-python. Features include dynamic model loading, model caching, and support for Qwen3-style tool/function calling.

## Features

- **FastAPI** framework for high-performance API serving
- **llama-cpp-python** for efficient local model inference
- **Dynamic model loading** - Load any model from `/models` directory by specifying its name
- **Model caching** - Models are cached after first load for fast subsequent requests
- **Multiple model support** - Switch between different chat and embedding models
- **Model management** - List, download, and manage models via API
- **Qwen3 tool binding** - automatic conversion from OpenAI format to Qwen3 tool calling format
- **OpenAI-compatible API** - `/v1/chat/completions` and `/v1/embeddings` endpoints
- **CORS enabled** - ready for web applications
- **Structured & readable** - modular architecture with separate formatting logic

## Architecture

### Dynamic Model Loading

The server now supports loading models dynamically based on the request. Instead of hardcoding a single model at startup:

1. **List available models** using `/v1/models/chat` or `/v1/models/embeddings`
2. **Specify model name** in each request to `/v1/chat/completions` or `/v1/embeddings`
3. **Model validation** - Server checks if the model exists before attempting to load
4. **Automatic caching** - Once loaded, models stay in memory for fast subsequent requests
5. **Multiple models** - You can have multiple models in `/models` and switch between them

**Benefits:**
- No need to restart the server to use a different model
- Support for multiple specialized models (e.g., different sizes, quantizations)
- Efficient memory usage with caching
- Better error handling with model validation

### Request Flow

1. **Receive** OpenAI-format request at `/v1/chat/completions`
2. **Convert** messages and tools to Qwen3 format (with `<|im_start|>` tags and `<tools>` section)
3. **Generate** completion using the formatted prompt
4. **Parse** response to extract tool calls (from `<tool_call>` tags)
5. **Return** OpenAI-compatible response

### Qwen3 Format

The server automatically converts OpenAI format to Qwen3's expected format:

**Tools Section:**
```xml
<tools>
[
  {
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {...}
  }
]
</tools>
```

**Messages:**
```xml
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What's the weather in Tokyo?<|im_end|>
<|im_start|>assistant
```

**Tool Calls in Response:**
```xml
<tool_call>
{
  "name": "get_weather",
  "arguments": {"location": "Tokyo, Japan"}
}
</tool_call>
```

## Project Structure

```
llm-server/
├── main.py                 # FastAPI application with chat endpoint
├── model.py                # Singleton model loader
├── schemas.py              # Pydantic models for request/response
├── config.py               # Settings and configuration
├── prompt_formatters.py    # OpenAI to Qwen3 conversion logic
├── example_client.py       # Usage examples
├── requirements.in         # Dependency definitions
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Installation

### 1. Clone and Setup

**Important:** Always use the `.venv` virtual environment for package installation and running programs.

```bash
cd llm-server

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install pip-tools
.venv/bin/pip install pip-tools

# Compile requirements (optional, requirements.txt is committed)
.venv/bin/pip-compile requirements.in

# Install dependencies
.venv/bin/pip install -r requirements.txt
```

**Note:** All commands should be run with `.venv` activated or by using `.venv/bin/python` and `.venv/bin/pip` directly.

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
MODEL_PATH=/path/to/your/model.gguf
N_CTX=2048
N_GPU_LAYERS=0
N_THREADS=4
TEMPERATURE=0.7
MAX_TOKENS=512
```

**Important:** Download a `.gguf` model file and update `MODEL_PATH` to point to it.

Popular sources:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [TheBloke's models](https://huggingface.co/TheBloke)

## Usage

### Start the Server

**Always use `.venv` to run the server:**

```bash
# Activate virtual environment first
source .venv/bin/activate

# Then run the server
python run.py
```

Or without activating:

```bash
.venv/bin/python run.py
```

Or with uvicorn directly:

```bash
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### API Documentation

Interactive API documentation is automatically generated and available at:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with request/response examples
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation with a clean, responsive layout
- **OpenAPI Schema**: `http://localhost:8000/openapi.json` - Raw OpenAPI 3.0 specification

The Swagger UI allows you to:
- Browse all available endpoints with detailed descriptions
- View request/response schemas with examples
- Test API calls directly from your browser
- See authentication requirements (if configured)

### Example Requests

#### List Available Models

**List Chat Models:**
```bash
curl http://localhost:8000/v1/models/chat
```

**List Embedding Models:**
```bash
curl http://localhost:8000/v1/models/embeddings
```

#### Basic Chat Completion

**Note:** The `model` field is now **required**. Use the model name from `/v1/models/chat`.

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

#### Chat with Tool Binding

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

#### Embeddings

**Note:** The `model` field is now **required**. Use the model name from `/v1/models/embeddings`.

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-4B-Q4_K_M",
    "input": "This is a text to embed",
    "encoding_format": "float"
  }'
```

#### Python Client Example

```python
import requests

# First, list available models
models_response = requests.get("http://localhost:8000/v1/models/chat")
chat_models = models_response.json()
model_name = chat_models[0]["name"]  # Use the first available model

# Then, use it for chat completion
url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 200
}

response = requests.post(url, json=payload)
result = response.json()

print(result["choices"][0]["message"]["content"])
```

#### Testing the API

Run the included test script to verify dynamic model loading:

```bash
.venv/bin/python test_dynamic_loading.py
```

For detailed API usage examples, see [API_USAGE.md](./API_USAGE.md).

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to .gguf model file | Required |
| `N_CTX` | Context window size | 2048 |
| `N_GPU_LAYERS` | Number of layers to offload to GPU | 0 |
| `N_THREADS` | CPU threads for inference | 4 |
| `TEMPERATURE` | Default sampling temperature | 0.7 |
| `MAX_TOKENS` | Default max tokens to generate | 512 |
| `VERBOSE` | Enable verbose logging | False |

## Architecture

### Singleton Pattern

The `LLMModelSingleton` class ensures the model is loaded only once:

```python
from model import get_llm_model

# This will return the same model instance every time
model = get_llm_model()
```

Benefits:
- **Memory efficient** - only one model in memory
- **Fast response** - no reload on each request
- **Thread-safe** - shared across FastAPI workers

### Tool Binding

The server supports OpenAI-style function calling. When tools are provided:

1. Tool definitions are injected into the prompt
2. Model generates response (possibly with tool call)
3. Response is parsed for tool call JSON
4. Tool call is returned in structured format

## Performance Tips

1. **GPU Acceleration**: Set `N_GPU_LAYERS` > 0 if you have CUDA/Metal support
2. **Context Size**: Adjust `N_CTX` based on your model and use case
3. **Threads**: Set `N_THREADS` to match your CPU cores
4. **Model Size**: Smaller quantized models (Q4, Q5) offer better performance

## Troubleshooting

### Model fails to load

- Verify `MODEL_PATH` points to a valid `.gguf` file
- Check file permissions
- Ensure sufficient RAM for your model size

### Slow inference

- Increase `N_THREADS`
- Enable GPU layers with `N_GPU_LAYERS`
- Use a smaller/more quantized model

### CUDA errors

For GPU support, install with CUDA:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" .venv/bin/pip install llama-cpp-python --force-reinstall --no-cache-dir
```

For Metal (macOS):

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" .venv/bin/pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Testing

Run the test suite using pytest:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py


## Docker

- **Build image:**

```bash
docker build -t llm-server:latest .
```

- **Run container (recommended):** mount the host `models/` directory into the container so large model files are not baked into the image.

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/models" \
  -e PORT=8000 \
  -e MODEL_DIR=/models \
  --name llm-server \
  llm-server:latest
```

- **Notes:**
  - The Dockerfile builds Python wheels in a builder stage and keeps the final image minimal.
  - Do NOT copy your large `.gguf` model files into the image; mount them at `/models` as shown above.
  - Adjust `--workers` and `--threads` in the `Dockerfile` `CMD` if you need more concurrency.
  - For orchestrators, the container exposes a healthcheck at `/health`.
# Run with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src tests/
```

Or without activating the virtual environment:

```bash
.venv/bin/pytest
.venv/bin/pytest tests/test_integration.py -v
```

### Test Structure

- `tests/test_qwen3_format.py` - Unit tests for Qwen3 format conversion
- `tests/test_integration.py` - Integration tests with tool calling validation

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
