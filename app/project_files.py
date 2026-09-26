from __future__ import annotations

import os
import stat
from pathlib import Path, PureWindowsPath


MAX_PATH_CHARS = 1024
MAX_RELATIVE_PATH_CHARS = 512
MAX_DIRECTORY_ENTRIES = 5000
MAX_LIST_RESULTS = 100
MAX_SEARCH_FILES = 200
MAX_SEARCH_RESULTS = 50
MAX_FILE_BYTES = 512_000
MAX_SCAN_BYTES = 5_000_000
MAX_FILE_PROBE_BYTES = 4096
MAX_PROBE_BYTES_PER_REQUEST = 1_000_000
MAX_QUERY_CHARS = 160
MAX_EXCERPT_CHARS = 240

EXCLUDED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "virtualenv", "env",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".cache", "cache", "site-packages", ".aws", ".ssh",
}
EXCLUDED_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".der", ".jks",
    ".keystore", ".sqlite", ".sqlite3", ".db", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".pyc", ".class", ".docx", ".bin",
    ".xlsx", ".pptx", ".mp3", ".mp4", ".wav", ".woff", ".woff2",
}
PRIVATE_KEY_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
REPARSE_POINT_ATTRIBUTE = 0x400


class ProjectPathError(ValueError):
    pass


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", REPARSE_POINT_ATTRIBUTE)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_attribute)


def resolve_scope(root: Path, relative_path: str) -> tuple[Path, Path]:
    try:
        canonical_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectPathError("Project root is unavailable") from exc

    windows_path = PureWindowsPath(relative_path)
    if (
        not relative_path
        or len(relative_path) > MAX_PATH_CHARS
        or "\x00" in relative_path
        or ":" in relative_path
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in windows_path.parts
    ):
        raise ProjectPathError("Path must stay inside the configured project root")

    candidate = canonical_root.joinpath(*windows_path.parts)
    try:
        scope = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectPathError("Path must be an existing directory inside the configured project root") from exc

    if not _is_within(canonical_root, scope) or not scope.is_dir():
        raise ProjectPathError("Path must be an existing directory inside the configured project root")
    return canonical_root, scope


def _is_excluded(name: str, *, is_directory: bool) -> bool:
    lower_name = name.casefold()
    suffix = Path(lower_name).suffix
    if lower_name in EXCLUDED_DIRECTORIES:
        return True
    if lower_name.startswith(".env") or any(
        marker in lower_name for marker in ("credential", "secret", "password")
    ):
        return True
    name_parts = lower_name.replace("-", "_").replace(".", "_").split("_")
    if lower_name in {".npmrc", ".pypirc", ".netrc", "auth.json"} or any(
        part in {"token", "tokens"} for part in name_parts
    ):
        return True
    if lower_name in PRIVATE_KEY_NAMES or suffix in EXCLUDED_SUFFIXES:
        return True
    return is_directory and lower_name in {"credentials", "secrets"}


def _looks_like_text(path: Path, size: int) -> tuple[bool, int]:
    with path.open("rb") as file:
        sample = file.read(min(size, MAX_FILE_PROBE_BYTES))
    if b"\x00" in sample:
        return False, len(sample)
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(sample):
            return False, len(sample)
        try:
            sample[: exc.start].decode("utf-8")
        except UnicodeDecodeError:
            return False, len(sample)
    return True, len(sample)


def _walk_files(root: Path, scope: Path, *, file_limit: int) -> tuple[list[Path], bool]:
    pending = [scope]
    files: list[Path] = []
    entries_seen = 0
    probe_bytes = 0
    truncated = False

    while pending:
        directory = pending.pop()
        try:
            directory_metadata = directory.lstat()
            if _is_reparse_point(directory_metadata):
                continue
            canonical_directory = directory.resolve(strict=True)
            if not _is_within(root, canonical_directory):
                continue
            with os.scandir(canonical_directory) as iterator:
                entries = []
                for entry in iterator:
                    if entries_seen + len(entries) >= MAX_DIRECTORY_ENTRIES:
                        return files, True
                    entries.append(entry)
        except (OSError, RuntimeError):
            truncated = True
            continue

        entries_seen += len(entries)
        directories: list[Path] = []
        for entry in sorted(entries, key=lambda item: item.name.casefold()):
            try:
                metadata = entry.stat(follow_symlinks=False)
                if _is_reparse_point(metadata):
                    continue
                is_directory = stat.S_ISDIR(metadata.st_mode)
                if _is_excluded(entry.name, is_directory=is_directory):
                    continue
                candidate = Path(entry.path).resolve(strict=True)
                if not _is_within(root, candidate):
                    continue
                if is_directory:
                    directories.append(candidate)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_FILE_BYTES:
                    relative_path = candidate.relative_to(root).as_posix()
                    if len(relative_path) > MAX_RELATIVE_PATH_CHARS:
                        truncated = True
                        continue
                    probe_size = min(metadata.st_size, MAX_FILE_PROBE_BYTES)
                    if probe_bytes + probe_size > MAX_PROBE_BYTES_PER_REQUEST:
                        return files, True
                    if not _looks_like_text(candidate, metadata.st_size)[0]:
                        probe_bytes += probe_size
                        continue
                    probe_bytes += probe_size
                    files.append(candidate)
                    if len(files) > file_limit:
                        return files[:file_limit], True
            except (OSError, RuntimeError, ValueError):
                continue
        pending.extend(reversed(directories))

    return files, truncated


def list_project_files(root: Path, relative_path: str = ".") -> dict[str, object]:
    canonical_root, scope = resolve_scope(root, relative_path)
    files, truncated = _walk_files(canonical_root, scope, file_limit=MAX_LIST_RESULTS)
    return {
        "files": [path.relative_to(canonical_root).as_posix() for path in files],
        "truncated": truncated,
    }


def _excerpt(line: str, query: str) -> str:
    match_index = line.casefold().find(query.casefold())
    available_start = max(0, len(line) - MAX_EXCERPT_CHARS)
    start = max(0, min(match_index - (MAX_EXCERPT_CHARS - len(query)) // 2, available_start))
    return line[start : start + MAX_EXCERPT_CHARS]


def search_project_files(root: Path, query: str, relative_path: str = ".") -> dict[str, object]:
    if not query or len(query) > MAX_QUERY_CHARS:
        raise ProjectPathError("Search query is empty or too long")

    canonical_root, scope = resolve_scope(root, relative_path)
    files, truncated = _walk_files(canonical_root, scope, file_limit=MAX_SEARCH_FILES)
    results: list[dict[str, object]] = []
    bytes_scanned = 0
    folded_query = query.casefold()

    for path in files:
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
            continue

        relative_file = canonical_file.relative_to(canonical_root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if folded_query in line.casefold():
                results.append({
                    "path": relative_file,
                    "line": line_number,
                    "excerpt": _excerpt(line, query),
                })
                if len(results) >= MAX_SEARCH_RESULTS:
                    return {"results": results, "truncated": True}

    return {"results": results, "truncated": truncated}