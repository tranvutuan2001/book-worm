import re
import ast
import json
import uuid

class _QwenParser:
    """Private strategy for parsing Qwen model responses."""
    THOUGHT_PATTERNS = [
        r'<think>.*?</think>', 
        r'<think>.*',
        r'.*?</think>'
    ]
    TOOL_PATTERNS = [
        r'<tool_call>(.*?)</tool_call>',
        r'(tool_call:[a-zA-Z0-9_:]+\{.*?\})'
    ]

    @classmethod
    def parse(cls, text: str) -> tuple[str | None, list[dict[str, str]]]:
        tool_calls: list[dict[str, str]] = []
        clean_text = text

        # 1. Extract tool calls
        for pattern in cls.TOOL_PATTERNS:
            matches = list(re.finditer(pattern, clean_text, flags=re.DOTALL | re.IGNORECASE))
            for match in matches:
                # Use first capturing group if present, otherwise full match
                content = match.group(1).strip() if match.groups() else match.group(0).strip()
                extracted = cls._parse_call_content(content)
                
                if extracted:
                    if isinstance(extracted, list):
                        tool_calls.extend(extracted)
                    else:
                        tool_calls.append(extracted)
                    # Remove the matched block from clean_text
                    clean_text = clean_text.replace(match.group(0), '')

        # 2. Strip thoughts
        for pattern in cls.THOUGHT_PATTERNS:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        clean_text = clean_text.strip()
        return clean_text if clean_text else None, tool_calls

    @classmethod
    def _parse_call_content(cls, content: str) -> dict[str, str] | list[dict[str, str]] | None:
        # Strategy A: JSON (Standard Qwen format)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                return MLXResponseParser.format_tool_call(data["name"], data.get("arguments", {}))
            elif isinstance(data, list):
                results: list[dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        results.append(MLXResponseParser.format_tool_call(item["name"], item.get("arguments", {})))
                if results: return results
        except json.JSONDecodeError:
            pass

        # Strategy B: Tagged JSON (e.g. tool_call:name{...})
        temp_content = content.strip()
        if temp_content.lower().startswith("tool_call:"):
            temp_content = temp_content[10:].strip()
            
        match = re.match(r'^([a-zA-Z0-9_:]+)\s*(\{.*\})$', temp_content, re.DOTALL)
        if match:
            func_name = match.group(1).strip().split(":")[-1]
            try:
                kwargs = json.loads(match.group(2).strip())
                return MLXResponseParser.format_tool_call(func_name, kwargs)
            except json.JSONDecodeError:
                pass

        # Strategy D: XML-like format (used by some Qwen reasoning models)
        try:
            func_match = re.search(r'<function=(.*?)>', content)
            if func_match:
                func_name = func_match.group(1).strip()
                params = {}
                param_matches = re.finditer(r'<parameter=(.*?)>(.*?)</parameter>', content, re.DOTALL)
                for p_match in param_matches:
                    p_name = p_match.group(1).strip()
                    p_value = p_match.group(2).strip()
                    try:
                        params[p_name] = json.loads(p_value)
                    except json.JSONDecodeError:
                        params[p_name] = p_value
                return MLXResponseParser.format_tool_call(func_name, params)
        except Exception:
            pass

        return None

class _GemmaParser:
    """Private strategy for parsing Gemma model responses."""
    THOUGHT_PATTERNS = [
        r'<\|?channel\|?>?thought.*?<\|?/?channel\|?>?',
        r'<\|?channel\|?>?thought.*'
    ]
    TOOL_PATTERNS = [
        r'<\|?tool_call\|?>(.*?)<\|?/?tool_call\|?>(?:\n| |$)?',
        r'<\|?tool_call\|?>(.*)' 
    ]

    @classmethod
    def parse(cls, text: str) -> tuple[str | None, list[dict[str, str]]]:
        tool_calls: list[dict[str, str]] = []
        clean_text = text

        # 1. Extract tool calls
        for pattern in cls.TOOL_PATTERNS:
            matches = list(re.finditer(pattern, clean_text, flags=re.DOTALL | re.IGNORECASE))
            for match in matches:
                content = match.group(1).strip() if match.groups() else match.group(0).strip()
                extracted = cls._parse_call_content(content)
                
                if extracted:
                    tool_calls.append(extracted)
                    clean_text = clean_text.replace(match.group(0), '')

        # 2. Strip thoughts
        for pattern in cls.THOUGHT_PATTERNS:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        clean_text = clean_text.strip()
        return clean_text if clean_text else None, tool_calls

    @classmethod
    def _parse_call_content(cls, content: str) -> dict[str, str] | None:
        # 1. Pre-process: handle weird Gemma markers like <|"|> or <|'|>
        # These appear to be used by some Gemma models as special string delimiters
        temp_content = content.strip()
        temp_content = temp_content.replace('<|"|>', '"').replace("<|'|>", "'")
        
        # Strategy C: Python AST (Common for Gemma/Hermes 'call:func(args)' format)
        ast_content = temp_content
        if ast_content.lower().startswith("call:"):
            ast_content = ast_content[5:].strip()
            
        try:
            tree = ast.parse(ast_content, mode='eval')
            if isinstance(tree.body, ast.Call) and hasattr(tree.body.func, 'id'):
                func_name = tree.body.func.id
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords if kw.arg}
                return MLXResponseParser.format_tool_call(func_name, kwargs)
        except Exception:
            pass
            
        # Strategy B: Brace format (e.g. func{key:val} or call:func{key:val})
        # Common in some Gemma 2 reasoning variants
        brace_match = re.match(r'^(?:call:)?([a-zA-Z0-9_:]+)\s*\{(.*)\}$', temp_content, re.DOTALL | re.IGNORECASE)
        if brace_match:
            func_name = brace_match.group(1).strip().split(":")[-1]
            args_content = brace_match.group(2).strip()
            
            if not args_content:
                return MLXResponseParser.format_tool_call(func_name, {})

            # Try to parse args_content as JSON first
            try:
                data = json.loads("{" + args_content + "}")
                return MLXResponseParser.format_tool_call(func_name, data)
            except Exception:
                # Fallback: try to parse as a series of key:value pairs using regex
                # This handles unquoted keys like {question: "..."}
                params = {}
                # This pattern matches key: value, where value is a quoted string or a primitive
                kv_pairs = re.finditer(r'([a-zA-Z0-9_]+)\s*:\s*("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[a-zA-Z0-9_.-]+)', args_content)
                for kv in kv_pairs:
                    key = kv.group(1)
                    val_raw = kv.group(2)
                    try:
                        # Try to parse value as JSON (handles strings, numbers, booleans)
                        params[key] = json.loads(val_raw.replace("'", '"')) if val_raw.startswith("'") else json.loads(val_raw)
                    except Exception:
                        # Fallback for unquoted strings or complex values
                        params[key] = val_raw.strip("'\"")
                
                if params:
                    return MLXResponseParser.format_tool_call(func_name, params)
                elif not args_content: # Handle empty braces {}
                    return MLXResponseParser.format_tool_call(func_name, {})

        # Strategy A Fallback: Pure JSON
        try:
            data = json.loads(temp_content)
            if isinstance(data, dict) and "name" in data:
                return MLXResponseParser.format_tool_call(data["name"], data.get("arguments", {}))
        except json.JSONDecodeError:
            pass

        return None

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
            return _GemmaParser.parse(response_text)
        elif "qwen" in path or not path: # Default to Qwen if path is missing to support tests and generic usage
            return _QwenParser.parse(response_text)
        
        from app.domain.exceptions.llm_exception import LLMGenerationException
        raise LLMGenerationException(
            message=f"Unsupported model family in path: {model_path}. MLX provider currently only supports Qwen and Gemma models for tool parsing.",
            provider="mlx"
        )


    @staticmethod
    def format_tool_call(name: str, arguments: dict[str, object] | object) -> dict[str, str]:
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
