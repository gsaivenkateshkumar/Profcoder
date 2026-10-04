"""Offline Python syntax check for a single project file.

Reuses ``ProjectFileEditor``'s safe single-file resolution and read-only
text reading (the same reader used by preview and the agent's read_file
tool). Source is only ever passed to ``ast.parse`` -- never imported,
executed, or written back to disk.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.file_edits import FileEditError, ProjectFileEditor

MAX_SYNTAX_MESSAGE_CHARS = 2000

__all__ = ["FileEditError", "MAX_SYNTAX_MESSAGE_CHARS", "check_python_syntax"]


def check_python_syntax(root: Path, relative_path: str) -> dict[str, object]:
    if not isinstance(relative_path, str) or not relative_path.casefold().endswith(".py"):
        raise FileEditError("Only .py files can be syntax-checked")

    editor = ProjectFileEditor(root)
    _, _, canonical_path = editor._resolve_target(relative_path)
    source = editor.read_text(relative_path)

    try:
        ast.parse(source, filename=canonical_path)
    except SyntaxError as exc:
        return {
            "path": canonical_path,
            "ok": False,
            "line": exc.lineno,
            "column": exc.offset,
            "message": str(exc.msg)[:MAX_SYNTAX_MESSAGE_CHARS],
        }
    except (ValueError, RecursionError) as exc:
        return {
            "path": canonical_path,
            "ok": False,
            "line": None,
            "column": None,
            "message": str(exc)[:MAX_SYNTAX_MESSAGE_CHARS],
        }
    return {"path": canonical_path, "ok": True, "line": None, "column": None, "message": None}
