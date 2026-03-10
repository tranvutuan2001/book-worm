#!/usr/bin/env bash
# =============================================================================
# Book Worm API – end-to-end smoke test
# =============================================================================
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
DOCUMENT_NAME="The Simple Path to Wealth_ Your - J Collins_20260219_231152"

# Colours for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

separator() { echo -e "${CYAN}──────────────────────────────────────────────${NC}"; }

# jq is required for JSON parsing
if ! command -v jq &>/dev/null; then
  echo "Error: 'jq' is required but not installed. Install it with: brew install jq" >&2
  exit 1
fi

# =============================================================================
# STEP 0 – Unload all currently loaded models
# =============================================================================
separator
echo -e "${YELLOW}STEP 0: Unload all loaded models${NC}"
separator

echo -e "\n${GREEN}► GET /v1/models/loaded${NC}"
LOADED_MODELS=$(curl -sf "${BASE_URL}/v1/models/loaded")
echo "$LOADED_MODELS" | jq .

LOADED_COUNT=$(echo "$LOADED_MODELS" | jq 'length')
echo -e "\nFound ${LOADED_COUNT} loaded model(s)."

if [[ "$LOADED_COUNT" -gt 0 ]]; then
  echo "$LOADED_MODELS" | jq -c '.[]' | while read -r model; do
    MODEL_PATH=$(echo "$model" | jq -r '.model_path')
    MODEL_TYPE=$(echo "$model" | jq -r '.model_type')
    echo -e "\n${GREEN}► POST /v1/models/unload  (${MODEL_TYPE}: ${MODEL_PATH})${NC}"
    UNLOAD_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/models/unload" \
      -H "Content-Type: application/json" \
      -d "{\"model_path\": \"${MODEL_PATH}\", \"model_type\": \"${MODEL_TYPE}\"}")
    echo "$UNLOAD_RESPONSE" | jq .
  done
  echo -e "\n${GREEN}All loaded models unloaded.${NC}"
else
  echo -e "${GREEN}No models were loaded — nothing to unload.${NC}"
fi

# =============================================================================
# STEP 1 – List embedding models and chat models
# =============================================================================
separator
echo -e "${YELLOW}STEP 1: List available models${NC}"
separator

echo -e "\n${GREEN}► GET /v1/models/embeddings${NC}"
EMBEDDING_MODELS=$(curl -sf "${BASE_URL}/v1/models/embeddings")
echo "$EMBEDDING_MODELS" | jq .

echo -e "\n${GREEN}► GET /v1/models/chat${NC}"
CHAT_MODELS=$(curl -sf "${BASE_URL}/v1/models/chat")
echo "$CHAT_MODELS" | jq .

# =============================================================================
# STEP 2 – Pick first model of each type and load them
# =============================================================================
separator
echo -e "${YELLOW}STEP 2: Load one embedding model and one chat model${NC}"
separator

# Extract the first model from each list
EMBEDDING_MODEL_PATH=$(echo "$EMBEDDING_MODELS" | jq -r '.[0].path')
EMBEDDING_MODEL_NAME=$(echo "$EMBEDDING_MODELS" | jq -r '.[0].path')

CHAT_MODEL_PATH=$(echo "$CHAT_MODELS" | jq -r '.[0].path')
CHAT_MODEL_NAME=$(echo "$CHAT_MODELS" | jq -r '.[0].path')

echo -e "\nSelected embedding model : ${EMBEDDING_MODEL_NAME}  (path: ${EMBEDDING_MODEL_PATH})"
echo -e "Selected chat model      : ${CHAT_MODEL_NAME}  (path: ${CHAT_MODEL_PATH})"

echo -e "\n${GREEN}► POST /v1/models/load  (embedding)${NC}"
LOAD_EMBEDDING_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model_path\": \"${EMBEDDING_MODEL_PATH}\", \"model_type\": \"embedding\"}")
echo "$LOAD_EMBEDDING_RESPONSE" | jq .

echo -e "\n${GREEN}► POST /v1/models/load  (chat)${NC}"
LOAD_CHAT_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model_path\": \"${CHAT_MODEL_PATH}\", \"model_type\": \"chat\"}")
echo "$LOAD_CHAT_RESPONSE" | jq .

# =============================================================================
# STEP 3 – Ask a question about the document
# =============================================================================
separator
echo -e "${YELLOW}STEP 3: Ask a question about the document${NC}"
separator

TIMESTAMP=$(date +%s)
QUESTION="What is snp500"

PAYLOAD=$(jq -n \
  --arg doc   "$DOCUMENT_NAME" \
  --arg chat  "$CHAT_MODEL_NAME" \
  --arg emb   "$EMBEDDING_MODEL_NAME" \
  --argjson ts "$TIMESTAMP" \
  '{
    id: "conv_test_001",
    timestamp: $ts,
    document_name: $doc,
    chat_model: $chat,
    embedding_model: $emb,
    message_list: [
      {
        id: "msg_001",
        role: "user",
        content: "What is snp500",
        timestamp: $ts
      }
    ]
  }')

echo -e "\nRequest payload:"
echo "$PAYLOAD" | jq .

echo -e "\n${GREEN}► POST /ask  (waiting for response…)${NC}"
ASK_RESPONSE=$(curl -sf --max-time 300 -X POST "${BASE_URL}/ask" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo -e "\nFull response:"
echo "$ASK_RESPONSE" | jq .

echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━  AI Answer  ━━━━━━━━━━━━━━━━━━━${NC}"
echo "$ASK_RESPONSE" | jq -r '.message'
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

separator
echo -e "${GREEN}All steps completed successfully.${NC}"
separator
