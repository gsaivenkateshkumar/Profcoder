"""Offline feasibility tool: extract Python function-completion pairs from a
reviewed corpus manifest.

Reads only the files an existing, already-reviewed manifest selects (through
the same validated path/size/hash checks as preparation), parses each with
``ast`` -- never ``exec``/``eval`` -- and for every module-level function or
class method builds a candidate pair:

  prompt = a leading docstring or adjacent ``#`` comment block, plus the
           function signature
  target = the remaining function body, exact original source text

Candidates keep their source project, corpus-internal path, original
repository-relative path (when the manifest records one), commit, and rights
basis, and never cross the manifest's existing train/validation split when
checking for duplicates. Only aggregate counts are ever printed; prompt and
target text are written solely to local report files the caller chooses.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from model_lab.audit import scan_text
from model_lab.prepare import read_selection
from model_lab.tokenizer import encode

MAX_CANDIDATES = 5000
MAX_FUNCTIONS_PER_FILE = 200
MIN_DESCRIPTIVE_CHARS = 8
TRIVIAL_DESCRIPTIONS = {"todo", "fixme", "note", "pragma: no cover", "xxx"}
CONTEXT_LENGTHS = (128, 256, 512, 1024)


@dataclass(frozen=True)
class Candidate:
    project: str
    split: str
    corpus_path: str
    source_path: str | None
    repository: str | None
    commit: str | None
    rights_basis: str | None
    license: str | None
    qualified_name: str
    lineno: int
    prompt_source: str
    prompt: str
    target: str
    prompt_bytes: int
    target_bytes: int
    total_tokens: int
    fits: dict[int, bool]
    content_sha256: str


def _walk_functions(node: ast.AST, class_stack: list[str]):
    """Yield (def_node, qualified_name) for module functions and class methods.

    Functions nested inside other functions are skipped: their indentation
    and enclosing-scope semantics make isolated prompt/target extraction
    unreliable, so this is a stated limitation rather than a bug.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, ".".join(class_stack + [child.name])
        elif isinstance(child, ast.ClassDef):
            yield from _walk_functions(child, class_stack + [child.name])
        else:
            yield from _walk_functions(child, class_stack)


def _is_trivial_body(stmts: list[ast.stmt]) -> bool:
    if len(stmts) != 1:
        return False
    stmt = stmts[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return True
    if isinstance(stmt, ast.Raise):
        exc = stmt.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        name = target.id if isinstance(target, ast.Name) else (
            target.attr if isinstance(target, ast.Attribute) else None
        )
        return name == "NotImplementedError"
    return False


def _is_descriptive(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < MIN_DESCRIPTIVE_CHARS:
        return False
    if stripped.strip(" .:#").lower() in TRIVIAL_DESCRIPTIONS:
        return False
    return any(ch.isalpha() for ch in stripped)


def _leading_comment(source_lines: list[str], def_lineno: int, def_col: int) -> tuple[str, str] | None:
    """Return (original comment lines, descriptive content) directly above def_lineno, or None."""
    index = def_lineno - 2  # zero-indexed line directly above the def line
    start = def_lineno - 1
    content_parts: list[str] = []
    while index >= 0:
        line = source_lines[index]
        stripped = line.strip()
        if stripped == "" or stripped.startswith("@"):
            break
        if not stripped.startswith("#") or (len(line) - len(line.lstrip(" \t"))) != def_col:
            break
        content_parts.append(stripped.lstrip("#").strip())
        start = index
        index -= 1
    if start == def_lineno - 1:
        return None
    content_parts.reverse()
    return "\n".join(source_lines[start : def_lineno - 1]), "\n".join(content_parts)


def _extract_one(
    node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str], source_text: str
) -> tuple[str, str, str] | str:
    """Return (prompt, target, prompt_source) or a rejection-reason string."""
    if not node.body:
        return "trivial_target"
    first_body = node.body[0]
    signature_lines = source_lines[node.lineno - 1 : first_body.lineno - 1]
    if not signature_lines or not "\n".join(signature_lines).strip():
        return "oneliner_def_unsupported"
    signature_text = "\n".join(signature_lines)

    docstring = ast.get_docstring(node, clean=False)
    if docstring is not None and docstring.strip():
        doc_source = ast.get_source_segment(source_text, first_body)
        if doc_source is None:
            return "docstring_extract_failed"
        if not _is_descriptive(docstring):
            return "no_descriptive_prompt"
        indent = " " * first_body.col_offset
        prompt = signature_text + "\n" + indent + doc_source
        prompt_source = "docstring"
        remaining = node.body[1:]
    else:
        leading = _leading_comment(source_lines, node.lineno, node.col_offset)
        if leading is None or not _is_descriptive(leading[1]):
            return "no_descriptive_prompt"
        comment_text, _ = leading
        prompt = comment_text + "\n" + signature_text
        prompt_source = "comment"
        remaining = node.body

    if not remaining or _is_trivial_body(remaining):
        return "trivial_target"
    target_text = "\n".join(source_lines[remaining[0].lineno - 1 : remaining[-1].end_lineno])
    if not target_text.strip():
        return "trivial_target"
    return prompt, target_text, prompt_source


def extract_candidates(manifest_path: Path) -> dict[str, object]:
    """Read-only extraction; never executes any selected source."""
    selection = read_selection(manifest_path)
    manifest = json.loads(selection.manifest_bytes)
    source_paths: dict[tuple[str, str], str] = {}
    if selection.version == 3:
        for item in manifest["samples"]:
            source_paths[(item["project"], item["path"])] = item["source_path"]

    rejections: Counter = Counter()
    accepted: list[Candidate] = []
    seen_split_by_hash: dict[str, str] = {}
    total_considered = 0

    for sample in selection.samples:
        try:
            tree = ast.parse(sample.data.decode("utf-8"))
        except (SyntaxError, ValueError, RecursionError, UnicodeDecodeError):
            rejections["unparsable_file"] += 1
            continue
        source_text = sample.data.decode("utf-8")
        source_lines = source_text.split("\n")
        for count, (node, qualified_name) in enumerate(_walk_functions(tree, []), start=1):
            if count > MAX_FUNCTIONS_PER_FILE:
                break
            total_considered += 1
            if total_considered > MAX_CANDIDATES:
                raise ValueError("candidate count exceeds the safety bound")
            outcome = _extract_one(node, source_lines, source_text)
            if isinstance(outcome, str):
                rejections[outcome] += 1
                continue
            prompt, target, prompt_source = outcome
            combined = prompt + "\n" + target
            if scan_text(combined):
                rejections["unsafe_content"] += 1
                continue
            digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            previous_split = seen_split_by_hash.get(digest)
            if previous_split is not None:
                rejections["duplicate_cross_split" if previous_split != sample.split else "duplicate_same_split"] += 1
                continue
            seen_split_by_hash[digest] = sample.split
            token_count = len(encode(combined, add_bos=True, add_eos=True))
            project_meta = selection.projects.get(sample.project or "", {})
            accepted.append(Candidate(
                project=sample.project or "",
                split=sample.split,
                corpus_path=sample.path,
                source_path=source_paths.get((sample.project, sample.path)),
                repository=project_meta.get("repository"),
                commit=project_meta.get("commit"),
                rights_basis=project_meta.get("rights_basis"),
                license=project_meta.get("license"),
                qualified_name=qualified_name,
                lineno=node.lineno,
                prompt_source=prompt_source,
                prompt=prompt,
                target=target,
                prompt_bytes=len(prompt.encode("utf-8")),
                target_bytes=len(target.encode("utf-8")),
                total_tokens=token_count,
                fits={length: token_count <= length for length in CONTEXT_LENGTHS},
                content_sha256=digest,
            ))

    sorted_lengths = sorted(CONTEXT_LENGTHS)
    length_buckets = Counter()
    for candidate in accepted:
        previous = 0
        for length in sorted_lengths:
            if candidate.total_tokens <= length:
                bucket = f"<={length}" if previous == 0 else f"{previous + 1}-{length}"
                break
            previous = length
        else:
            bucket = f">{sorted_lengths[-1]}"
        length_buckets[bucket] += 1

    report = {
        "manifest_sha256": hashlib.sha256(selection.manifest_bytes).hexdigest(),
        "files_scanned": len(selection.samples),
        "candidates_considered": total_considered,
        "candidates_accepted": len(accepted),
        "rejection_reasons": dict(sorted(rejections.items())),
        "length_buckets": dict(sorted(length_buckets.items())),
        "fits": {length: sum(c.fits[length] for c in accepted) for length in sorted_lengths},
        "project_distribution": dict(sorted(Counter(c.project for c in accepted).items())),
        "split_distribution": dict(sorted(Counter(c.split for c in accepted).items())),
        "prompt_source_distribution": dict(sorted(Counter(c.prompt_source for c in accepted).items())),
        "note": "Heuristic AST-only extraction; prompts come only from docstrings or adjacent leading comments.",
    }
    return {"report": report, "candidates": accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory for candidates/report JSON; never overwrites")
    args = parser.parse_args()
    try:
        result = extract_candidates(args.manifest)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        print(f"extraction refused: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    args.output.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output / "candidates.json"
    report_path = args.output / "report.json"
    for path in (candidates_path, report_path):
        if path.exists():
            print(f"extraction refused: {path} already exists", file=sys.stderr)
            raise SystemExit(2)
    with candidates_path.open("x", encoding="utf-8") as target:
        json.dump([asdict(c) for c in result["candidates"]], target, indent=1)
    with report_path.open("x", encoding="utf-8") as target:
        json.dump(result["report"], target, indent=2)
    print(json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()
