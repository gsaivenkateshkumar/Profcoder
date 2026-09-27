"""Offline, read-only lookup of Python function/class definitions by name.

Pilot only: not wired to the agent or any HTTP route. It walks files with the
same exclusions and bounds as search_text, parses Python source with ``ast``
(which never executes it), and reports exact definition lines.
"""

from __future__ import annotations

import ast
import keyword
import unicodedata
from pathlib import Path

from app.project_files import (
    MAX_EXCERPT_CHARS,
    MAX_FILE_BYTES,
    MAX_QUERY_CHARS,
    MAX_SCAN_BYTES,
    MAX_SEARCH_FILES,
    MAX_SEARCH_RESULTS,
    ProjectPathError,
    _is_reparse_point,
    _is_within,
    _walk_files,
    resolve_scope,
)

DEFINITION_KINDS = {
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "async function",
    ast.ClassDef: "class",
}


def _valid_name(name: str) -> bool:
    parts = name.split(".")
    return (
        0 < len(name) <= MAX_QUERY_CHARS
        and all(part.isidentifier() and not keyword.iskeyword(part) for part in parts)
    )


def _definitions(tree: ast.AST) -> list[tuple[str, ast.AST, str]]:
    """Return (qualified name, node, kind) for every def/class, outermost first."""
    found: list[tuple[str, ast.AST, str]] = []
    pending: list[tuple[ast.AST, tuple[str, ...], bool]] = [(tree, (), False)]
    while pending:
        node, scope, in_class = pending.pop()
        for child in reversed(list(ast.iter_child_nodes(node))):
            kind = DEFINITION_KINDS.get(type(child))
            if kind is None:
                pending.append((child, scope, in_class))
                continue
            qualified = (*scope, child.name)
            if kind != "class" and in_class:
                kind = "async method" if kind == "async function" else "method"
            found.append((".".join(qualified), child, kind))
            pending.append((child, qualified, kind == "class"))
    return sorted(found, key=lambda item: (item[1].lineno, item[1].col_offset))


def find_definitions(root: Path, name: str, relative_path: str = ".") -> dict[str, object]:
    """Find ``def``/``class`` definitions named ``name`` (or a dotted qualified name).

    A plain name matches any definition with that exact, case-sensitive name;
    ``Class.method`` matches only that qualified definition.
    """
    if not isinstance(name, str) or not _valid_name(name):
        raise ProjectPathError("Definition name must be a Python identifier or dotted name")
    name = unicodedata.normalize("NFKC", name)  # match the parser's identifier form
    last_part = name.rsplit(".", 1)[-1]
    canonical_root, scope = resolve_scope(root, relative_path)
    files, truncated = _walk_files(canonical_root, scope, file_limit=MAX_SEARCH_FILES)
    qualified = "." in name
    results: list[dict[str, object]] = []
    bytes_scanned = 0
    unparsed = 0

    for path in files:
        if path.suffix.casefold() != ".py":
            continue
        try:
            canonical_file = path.resolve(strict=True)
            if not _is_within(canonical_root, canonical_file):
                truncated = True
                continue
            metadata = canonical_file.lstat()
            if _is_reparse_point(metadata):
                truncated = True
                continue
            file_size = metadata.st_size
            remaining = MAX_SCAN_BYTES - bytes_scanned
            if file_size > MAX_FILE_BYTES:
                continue
            if file_size > remaining:
                truncated = True
                break
            with canonical_file.open("rb") as file:
                content = file.read(min(file_size, remaining))
        except OSError:
            truncated = True
            continue

        bytes_scanned += len(content)
        if b"\x00" in content:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            unparsed += 1
            continue
        # Exact shortcut: an ASCII file lacking the name cannot define it. Non-ASCII
        # files are always parsed because Python NFKC-normalizes identifiers.
        if text.isascii() and last_part not in text:
            continue
        try:
            tree = ast.parse(text, filename=path.name)  # parses only; never executes
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            unparsed += 1
            continue

        lines = text.splitlines()
        relative_file = canonical_file.relative_to(canonical_root).as_posix()
        for qualified_name, node, kind in _definitions(tree):
            if (qualified_name if qualified else node.name) != name:
                continue
            line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            results.append({
                "path": relative_file,
                "line": node.lineno,
                "kind": kind,
                "qualified_name": qualified_name,
                "excerpt": line.strip()[:MAX_EXCERPT_CHARS],
            })
            if len(results) >= MAX_SEARCH_RESULTS:
                return {"results": results, "truncated": True, "unparsed_files": unparsed}

    return {"results": results, "truncated": truncated, "unparsed_files": unparsed}
