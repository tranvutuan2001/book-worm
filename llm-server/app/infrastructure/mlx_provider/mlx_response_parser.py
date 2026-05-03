import re
import ast
import json
import uuid
from typing import Any

class MLXResponseParser:
    """
    A flexible parser for handling various output formats from different MLX models.
    Supports Qwen and Gemma model families with distinct parsing strategies.
    """

    @classmethod
    def parse(cls, response_text: str, model_path: str | None = None) -> tuple[str | None, list[dict[str, str]]]:
        """
        Parses the raw response text based on the model family.
        Returns a tuple of (clean_content, list_of_tool_calls).
        """
        path = model_path.lower() if model_path else ""
        
        if "gemma" in path:
            return cls._parse_by_strategy(
                response_text,
                thought_patterns=[
                    r'<\|?channel\|?>?thought.*?<\|?/?channel\|?>?',
                    r'<\|?channel\|?>?thought.*'
                ],
                tool_patterns=[r'<\|?tool_call\|?>(.*?)<\|?/?tool_call\|?>?'],
                is_gemma=True
            )
        elif "qwen" in path:
            return cls._parse_by_strategy(
                response_text,
                thought_patterns=[
                    r'<think>.*?</think>', 
                    r'<think>.*',
                    r'.*?</think>' # Handle pre-filled thoughts ending with </think>
                ],
                tool_patterns=[
                    r'<tool_call>(.*?)</tool_call>',
                    r'(tool_call:[a-zA-Z0-9_:]+\{.*?\})' # New Qwen format
                ],
                is_gemma=False
            )
        
        from app.domain.exceptions.llm_exception import LLMGenerationException
        raise LLMGenerationException(
            message=f"Unsupported model family in path: {model_path}. MLX provider currently only supports Qwen and Gemma models for tool parsing.",
            provider="mlx"
        )

    @classmethod
    def _parse_by_strategy(
        cls, 
        text: str, 
        thought_patterns: list[str], 
        tool_patterns: list[str],
        is_gemma: bool | None
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Core parsing logic using provided patterns and strategy."""
        tool_calls: list[dict[str, str]] = []
        clean_text = text

        # 1. Extract tool calls FIRST to avoid them being stripped by thought patterns
        for pattern in tool_patterns:
            matches = list(re.finditer(pattern, clean_text, flags=re.DOTALL | re.IGNORECASE))
            for match in matches:
                # Use first capturing group if present, otherwise full match
                content = match.group(1).strip() if match.groups() else match.group(0).strip()
                extracted = cls._parse_call_content(content, is_gemma)
                
                if extracted:
                    if isinstance(extracted, list):
                        tool_calls.extend(extracted)
                    else:
                        tool_calls.append(extracted)
                    # Remove the matched block from clean_text
                    clean_text = clean_text.replace(match.group(0), '')

        # 2. Strip thoughts from the remaining text
        for pattern in thought_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        clean_text = clean_text.strip()
        if not clean_text and not tool_calls:
            return "", tool_calls
            
        return clean_text if clean_text else None, tool_calls

    @classmethod
    def _parse_call_content(cls, content: str, is_gemma: bool | None) -> dict[str, str] | list[dict[str, str]] | None:
        """Attempts to parse the inner content of a tool call block."""
        
        # Strategy A: JSON (Common for Qwen and some Gemma variants)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                return cls._format_tool_call(data["name"], data.get("arguments", {}))
            elif isinstance(data, list):
                results: list[dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        results.append(cls._format_tool_call(item["name"], item.get("arguments", {})))
                if results: return results
        except json.JSONDecodeError:
            pass

        # Strategy B: Tagged/Prefixed formats (e.g. 'call:func_name{...}' or 'tool_call:provider:name{...}')
        temp_content = content.strip()
        
        # Strip common prefixes
        if temp_content.lower().startswith("tool_call:"):
            temp_content = temp_content[10:].strip()
        elif temp_content.lower().startswith("call:"):
            temp_content = temp_content[5:].strip()
            
        # Match name and arguments: name{json_args}
        # Name can contain colons (e.g., default_api:summarize)
        match = re.match(r'^([a-zA-Z0-9_:]+)\s*(\{.*\})$', temp_content, re.DOTALL)
        if match:
            func_name = match.group(1).strip()
            # If name has provider prefix (e.g. default_api:summarize), take only the function name
            if ":" in func_name:
                func_name = func_name.split(":")[-1]
                
            args_json = match.group(2).strip()
            try:
                kwargs = json.loads(args_json)
                return cls._format_tool_call(func_name, kwargs)
            except json.JSONDecodeError:
                pass

        # Strategy C: Python AST (Fallback for legacy or mixed formats)
        try:
            temp_content = content.strip()
            if temp_content.lower().startswith("tool_call:"):
                temp_content = temp_content[10:].strip()
            elif temp_content.lower().startswith("call:"):
                temp_content = temp_content[5:].strip()
                
            tree = ast.parse(temp_content, mode='eval')
            if isinstance(tree.body, ast.Call) and hasattr(tree.body.func, 'id'):
                func_name = tree.body.func.id
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords if kw.arg}
                return cls._format_tool_call(func_name, kwargs)
        except Exception:
            pass

        return None

    @classmethod
    def _format_tool_call(cls, name: str, arguments: object) -> dict[str, str]:
        """Normalizes the parsed tool call into the expected dictionary format."""
        if isinstance(arguments, dict):
            args_str = json.dumps(arguments)
        else:
            args_str = str(arguments)
            
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": name,
            "arguments": args_str
        }
