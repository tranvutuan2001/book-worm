import re
import ast
import json
import uuid
from typing import List, Tuple, Dict, Any, Optional

class MLXResponseParser:
    """
    A generic parser for handling various output formats from Qwen and Gemma MLX models.
    """
    
    # Patterns for reasoning/thought blocks that should be stripped from the final output
    THOUGHT_PATTERNS = [
        r'<think>.*?</think>',            # Qwen 
        r'<think>.*',                     # Unfinished Qwen
        r'<\|?channel\|?>?thought.*?<\|?/?channel\|?>?', # Gemma
        r'<\|?channel\|?>?thought.*',     # Unfinished Gemma
    ]

    # Patterns for tool call wrappers
    TOOL_CALL_PATTERNS = [
        r'<tool_call>(.*?)</tool_call>',          # Qwen
        r'<\|?tool_call\|?>(.*?)<\|?/?tool_call\|?>?', # Gemma
    ]

    @classmethod
    def strip_thoughts(cls, text: str) -> str:
        """Removes thought blocks from the response text."""
        clean_text = text
        for pattern in cls.THOUGHT_PATTERNS:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        return clean_text.strip()

    @classmethod
    def parse(cls, response_text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Parses the raw response text, strips thoughts, and extracts tool calls.
        Returns a tuple of (clean_content, list_of_tool_calls).
        """
        clean_text = cls.strip_thoughts(response_text)
        tool_calls = []

        # Strategy 1: Look for explicit tool call tags
        for pattern in cls.TOOL_CALL_PATTERNS:
            # We iterate over matches so we can remove them from clean_text
            matches = list(re.finditer(pattern, clean_text, flags=re.DOTALL | re.IGNORECASE))
            for match in matches:
                call_content = match.group(1).strip()
                extracted = cls._parse_call_content(call_content)
                
                if extracted:
                    if isinstance(extracted, list):
                        tool_calls.extend(extracted)
                    else:
                        tool_calls.append(extracted)
                
                # Remove the matched block from clean_text
                clean_text = clean_text.replace(match.group(0), '')

        clean_text = clean_text.strip()
        return clean_text if clean_text else None, tool_calls

    @classmethod
    def _parse_call_content(cls, content: str) -> Any:
        """Attempts to parse the inner content of a tool call block."""
        # 1. Try parsing as JSON (dict or list of dicts)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                return cls._format_tool_call(data["name"], data.get("arguments", {}))
            elif isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        results.append(cls._format_tool_call(item["name"], item.get("arguments", {})))
                if results:
                    return results
        except json.JSONDecodeError:
            pass

        # 2. Try parsing as Python AST (e.g., call:func_name(kwargs) or func_name(kwargs))
        if content.lower().startswith("call:"):
            content = content[5:].strip()
            
        try:
            tree = ast.parse(content, mode='eval')
            if isinstance(tree.body, ast.Call) and hasattr(tree.body.func, 'id'):
                func_name = tree.body.func.id
                kwargs = {}
                for keyword in tree.body.keywords:
                    # literal_eval evaluates strings, numbers, dicts, lists safely
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                return cls._format_tool_call(func_name, kwargs)
        except Exception:
            pass

        return None

    @classmethod
    def _format_tool_call(cls, name: str, arguments: Any) -> Dict[str, Any]:
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
