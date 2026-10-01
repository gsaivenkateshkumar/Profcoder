"""Read-only, bounded line-range preview of a single project file.

Reuses ``ProjectFileEditor``'s safe single-file resolution -- traversal,
exclusion, symlink/junction, hard-link, and size checks -- and its UTF-8/
binary text decoding. Only the read-only ``read_text`` is ever called; this
module never writes, edits, or executes anything.
"""

from __future__ import annotations

from pathlib import Path

from app.file_edits import FileEditError, ProjectFileEditor

MAX_PREVIEW_LINES = 300
MAX_PREVIEW_LINE_CHARS = 2000

__all__ = ["FileEditError", "MAX_PREVIEW_LINES", "MAX_PREVIEW_LINE_CHARS", "preview_project_file"]


def preview_project_file(
    root: Path, relative_path: str, start_line: int = 1, line_count: int = 120
) -> dict[str, object]:
    start_line = max(1, int(start_line))
    line_count = max(1, min(int(line_count), MAX_PREVIEW_LINES))

    editor = ProjectFileEditor(root)
    _, _, canonical_path = editor._resolve_target(relative_path)
    lines = editor.read_text(relative_path).splitlines()

    total_lines = len(lines)
    start_index = min(start_line - 1, total_lines)
    end_index = min(start_index + line_count, total_lines)
    selected = lines[start_index:end_index]
    bounded = [line[:MAX_PREVIEW_LINE_CHARS] for line in selected]

    return {
        "path": canonical_path,
        "start_line": start_index + 1,
        "end_line": start_index + len(bounded),
        "total_lines": total_lines,
        "lines": bounded,
        "truncated": (
            end_index < total_lines
            or any(len(line) > MAX_PREVIEW_LINE_CHARS for line in selected)
        ),
    }
