import json
from typing import List


def write_json_file(
    data: dict | list, path: str, *, ensure_ascii: bool = False, indent: int = 2
) -> None:
    """Write `data` as JSON to `path` using UTF-8.

    - `data` must be a JSON-serializable `dict` or `list`.
    - Ensures parent directories exist.
    - Uses `ensure_ascii=False` by default so non-ASCII characters are preserved.
    - Raises `TypeError` if `data` is not JSON-serializable.
    """
    import os

    # Quick validation to ensure the provided data is JSON-serializable.
    try:
        json.dumps(data, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError) as e:
        raise TypeError("Provided data is not JSON-serializable") from e

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def list_to_text(lst: List[str]) -> str:
    return "\n".join(lst)
