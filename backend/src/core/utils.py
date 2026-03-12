"""
General-purpose utility helpers shared across the application.
"""

import json
import os
from typing import List


def write_json_file(
    data: dict | list,
    path: str,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """Persist *data* as pretty-printed JSON to *path*.

    - Parent directories are created automatically.
    - Non-ASCII characters are preserved by default (``ensure_ascii=False``).

    Raises:
        TypeError: If *data* is not JSON-serialisable.
    """
    # Quick validation before touching the filesystem.
    try:
        json.dumps(data, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError) as exc:
        raise TypeError("Provided data is not JSON-serializable") from exc

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=ensure_ascii, indent=indent)


def list_to_text(lst: List[str]) -> str:
    """Join a list of strings with newlines."""
    return "\n".join(lst)
