import pytest
from app.infrastructure.mlx_provider.mlx_response_parser import MLXResponseParser

def test_strip_thoughts():
    # Test DeepSeek/Qwen style
    text = "<think>Calculating...</think> The answer is 42."
    clean_text, _ = MLXResponseParser.parse(text, model_path="qwen")
    assert clean_text == "The answer is 42."

    # Test unfinished thought
    text = "Here is the result.\n<think>Let me think"
    clean_text, _ = MLXResponseParser.parse(text, model_path="qwen")
    assert clean_text == "Here is the result."

    # Test Gemma/Hermes style
    text = "<|channel>thought\nThinking...<channel|>\nResult is here."
    clean_text, _ = MLXResponseParser.parse(text, model_path="gemma")
    assert clean_text == "Result is here."

def test_parse_gemma_style_tool_call():
    text = "<|tool_call>call:get_weather(location='Paris')<tool_call|>"
    clean_text, tools = MLXResponseParser.parse(text, model_path="gemma-2-9b")
    
    assert clean_text is None
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"
    assert "Paris" in tools[0]["arguments"]

def test_parse_qwen_style_tool_call():
    text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"London\"}}\n</tool_call>"
    clean_text, tools = MLXResponseParser.parse(text, model_path="qwen2.5-7b")
    
    assert clean_text is None
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"
    assert "London" in tools[0]["arguments"]



def test_parse_mixed_content():
    text = "<think>I should call weather API</think>\nI will check the weather for you.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}\n</tool_call>"
    clean_text, tools = MLXResponseParser.parse(text, model_path="qwen2.5-7b")
    
    assert clean_text == "I will check the weather for you."
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"
    assert "Rome" in tools[0]["arguments"]


def test_parse_gemma_reasoning_tool_call():
    text = """<|channel>thought
I need to find info about Collins.
<channel|><|tool_call>call:get_the_most_relevant_chunks{question:<|"|>tell me about collins<|"|>}<tool_call|>"""
    
    clean_text, tools = MLXResponseParser.parse(text, model_path="gemma-2-9b")
    
    assert clean_text is None
    assert len(tools) == 1
    assert tools[0]["name"] == "get_the_most_relevant_chunks"
    assert "tell me about collins" in tools[0]["arguments"]

def test_parse_gemma_empty_braces():
    text = "<|tool_call>get_document_title{}<tool_call|>"
    clean_text, tools = MLXResponseParser.parse(text, model_path="gemma")
    
    assert len(tools) == 1
    assert tools[0]["name"] == "get_document_title"
    assert tools[0]["arguments"] == "{}"
