#!/bin/bash

# Configuration
API_URL="http://localhost:8000/v1/chat/completions"
# Using the model found in your models directory
MODEL_NAME="Qwen3-4B-Instruct-2507-Q4_K_M"

echo "Sending chat completion request to $API_URL..."

curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "'"$MODEL_NAME"'",
       "messages": [
         {
           "role": "system",
           "content": "You are a helpful assistant."
         },
         {
           "role": "user",
           "content": "Hello! Can you briefly explain what a Large Language Model is?"
         }
       ],
       "temperature": 0.7,
       "max_tokens": 512
     }'