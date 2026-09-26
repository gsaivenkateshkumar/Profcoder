from __future__ import annotations

import difflib
import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from app.project_files import (
    MAX_FILE_BYTES,
    MAX_PATH_CHARS,
    ProjectPathError,
    _is_excluded,
    _is_reparse_point,
    _is_within,
    resolve_scope,
)


FileSignature = tuple[int, int, int, int]


class FileEditError(ValueError):
    pass


class UnsafeFileError(FileEditError):
    pass


class UnsupportedTextError(FileEditError):
    pass


class StaleFileError(FileEditError):
    pass


@dataclass(frozen=True)
class TextChangePreview:
    path: str
    original_sha256: str
    proposed_sha256: str
    diff: str
    _editor_id: object = field(repr=False, compare=False)
    _original_bytes: bytes = field(repr=False, compare=False)
    _proposed_bytes: bytes = field(repr=False, compare=False)
    _signature: FileSignature = field(repr=False, compare=False)
    _mode: int = field(repr=False, compare=False)


@dataclass(frozen=True)
class AppliedTextChange:
    path: str
    original_sha256: str
    applied_sha256: str
    _editor_id: object = field(repr=False, compare=False)
    _original_bytes: bytes = field(repr=False, compare=False)
    _signature: FileSignature = field(repr=False, compare=False)
    _mode: int = field(repr=False, compare=False)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _signature(metadata: os.stat_result) -> FileSignature:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size)


def _decode_text(content: bytes) -> tuple[str, bytes, str]:
    if b"\x00" in content:
        raise UnsupportedTextError("Binary files cannot be edited")

    bom = b"\xef\xbb\xbf" if content.startswith(b"\xef\xbb\xbf") else b""
    try:
        text = content[len(bom):].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedTextError("Only UTF-8 text files are supported") from exc
    if any(
        (ord(character) < 32 and character not in "\t\n\r\f")
        or ord(character) == 127
        for character in text
    ):
        raise UnsupportedTextError("Binary control characters cannot be edited")

    line_endings = set(re.findall(r"\r\n|\r|\n", text))
    if len(line_endings) > 1:
        raise UnsupportedTextError("Mixed line endings are not supported")
    newline = next(iter(line_endings), "\n")
    return text, bom, newline


def _format_text(text: str, bom: bytes, newline: str) -> bytes:
    if "\x00" in text:
        raise UnsupportedTextError("Binary text is not supported")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    formatted = normalized.replace("\n", newline)
    try:
        content = bom + formatted.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedTextError("Proposed text is not valid UTF-8") from exc
    if len(content) > MAX_FILE_BYTES:
        raise UnsupportedTextError("Edited file exceeds the size limit")
    return content


def _logical_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.splitlines(keepends=True)


class ProjectFileEditor:
    """Internal preview/apply/rollback interface scoped to one configured root."""

    def __init__(self, project_root: Path):
        try:
            root = Path(project_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise UnsafeFileError("Configured project root is unavailable") from exc
        if not root.is_dir():
            raise UnsafeFileError("Configured project root must be a directory")
        self._root = root
        self._editor_id = object()

    def _resolve_target(self, relative_path: str) -> tuple[Path, os.stat_result, str]:
        if not isinstance(relative_path, str) or not relative_path:
            raise UnsafeFileError("A relative project file path is required")
        windows_path = PureWindowsPath(relative_path)
        if (
            len(relative_path) > MAX_PATH_CHARS
            or "\x00" in relative_path
            or ":" in relative_path
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in windows_path.parts
            or not windows_path.parts
        ):
            raise UnsafeFileError("Path must stay inside the configured project root")

        parts = windows_path.parts
        parent_path = "/".join(parts[:-1]) or "."
        try:
            canonical_root, _ = resolve_scope(self._root, parent_path)
        except ProjectPathError as exc:
            raise UnsafeFileError("Parent path is outside the configured project root") from exc
        if canonical_root != self._root:
            raise UnsafeFileError("Path must stay inside the configured project root")

        current = canonical_root
        metadata: os.stat_result | None = None
        for index, part in enumerate(parts):
            is_directory = index < len(parts) - 1
            if _is_excluded(part, is_directory=is_directory):
                raise UnsafeFileError("Excluded project paths cannot be edited")
            current = current / part
            try:
                metadata = current.lstat()
                if _is_reparse_point(metadata):
                    raise UnsafeFileError("Symlinks and junctions cannot be edited")
                resolved = current.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise UnsafeFileError("Project file does not exist") from exc
            if _is_excluded(resolved.name, is_directory=is_directory):
                raise UnsafeFileError("Excluded project paths cannot be edited")
            if not _is_within(canonical_root, resolved):
                raise UnsafeFileError("Path must stay inside the configured project root")
            if is_directory and not stat.S_ISDIR(metadata.st_mode):
                raise UnsafeFileError("Parent path is not a directory")

        assert metadata is not None
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeFileError("Only regular files can be edited")
        if getattr(metadata, "st_nlink", 1) > 1:
            raise UnsafeFileError("Hard-linked files cannot be edited")
        if metadata.st_size > MAX_FILE_BYTES:
            raise UnsafeFileError("File exceeds the size limit")
        return current, metadata, "/".join(parts)

    def _read_target(
        self, relative_path: str
    ) -> tuple[Path, bytes, os.stat_result]:
        target, before, _ = self._resolve_target(relative_path)
        before_signature = _signature(before)
        try:
            with target.open("rb") as source:
                opened = os.fstat(source.fileno())
                if _signature(opened) != before_signature or not stat.S_ISREG(opened.st_mode):
                    raise StaleFileError("File changed while it was being read")
                content = source.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise UnsafeFileError("File exceeds the size limit")
            checked_target, checked, _ = self._resolve_target(relative_path)
        except FileEditError:
            raise
        except OSError as exc:
            raise UnsafeFileError("Project file could not be read") from exc
        if _signature(checked) != before_signature or checked_target != target:
            raise StaleFileError("File changed while it was being read")
        return checked_target, content, checked

    def preview(
        self,
        relative_path: str,
        proposed_text: str,
        *,
        expected_sha256: str,
    ) -> TextChangePreview:
        if not isinstance(proposed_text, str):
            raise UnsupportedTextError("Proposed content must be text")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise StaleFileError("A full expected SHA-256 hash is required")

        target, original_bytes, metadata = self._read_target(relative_path)
        original_sha256 = _sha256(original_bytes)
        if original_sha256.casefold() != expected_sha256.casefold():
            raise StaleFileError("File hash does not match the expected original")

        original_text, bom, newline = _decode_text(original_bytes)
        proposed_bytes = _format_text(proposed_text, bom, newline)
        proposed_text_on_disk, _, _ = _decode_text(proposed_bytes)
        diff = "".join(difflib.unified_diff(
            _logical_lines(original_text),
            _logical_lines(proposed_text_on_disk),
            fromfile=f"a/{target.relative_to(self._root).as_posix()}",
            tofile=f"b/{target.relative_to(self._root).as_posix()}",
            lineterm="\n",
        ))
        return TextChangePreview(
            path=target.relative_to(self._root).as_posix(),
            original_sha256=original_sha256,
            proposed_sha256=_sha256(proposed_bytes),
            diff=diff,
            _editor_id=self._editor_id,
            _original_bytes=original_bytes,
            _proposed_bytes=proposed_bytes,
            _signature=_signature(metadata),
            _mode=stat.S_IMODE(metadata.st_mode),
        )

    def _atomic_replace(
        self,
        relative_path: str,
        content: bytes,
        *,
        expected_sha256: str,
        expected_signature: FileSignature,
        mode: int,
    ) -> os.stat_result:
        target, current, metadata = self._read_target(relative_path)
        if _sha256(current) != expected_sha256 or _signature(metadata) != expected_signature:
            raise StaleFileError("File changed before replacement")

        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            os.chmod(temporary_path, mode)

            latest_target, latest, latest_metadata = self._read_target(relative_path)
            if (
                _sha256(latest) != expected_sha256
                or _signature(latest_metadata) != expected_signature
            ):
                raise StaleFileError("File changed before replacement")
            os.replace(temporary_path, latest_target)
            temporary_path = None
        except FileEditError:
            raise
        except OSError as exc:
            raise FileEditError("Atomic file replacement failed") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        _, written, written_metadata = self._read_target(relative_path)
        if written != content:
            raise StaleFileError("File changed during replacement")
        return written_metadata

    def apply(self, preview: TextChangePreview) -> AppliedTextChange:
        if preview._editor_id is not self._editor_id:
            raise UnsafeFileError("Preview belongs to a different editor")
        if _sha256(preview._original_bytes) != preview.original_sha256:
            raise UnsafeFileError("Preview original content is invalid")
        if _sha256(preview._proposed_bytes) != preview.proposed_sha256:
            raise UnsafeFileError("Preview proposed content is invalid")

        self._atomic_replace(
            preview.path,
            preview._proposed_bytes,
            expected_sha256=preview.original_sha256,
            expected_signature=preview._signature,
            mode=preview._mode,
        )
        _, applied, applied_metadata = self._read_target(preview.path)
        return AppliedTextChange(
            path=preview.path,
            original_sha256=preview.original_sha256,
            applied_sha256=_sha256(applied),
            _editor_id=self._editor_id,
            _original_bytes=preview._original_bytes,
            _signature=_signature(applied_metadata),
            _mode=preview._mode,
        )

    def rollback(self, change: AppliedTextChange) -> None:
        if change._editor_id is not self._editor_id:
            raise UnsafeFileError("Change belongs to a different editor")
        _, current, metadata = self._read_target(change.path)
        if (
            _sha256(current) != change.applied_sha256
            or _signature(metadata) != change._signature
        ):
            raise StaleFileError("File changed after apply; rollback refused")
        self._atomic_replace(
            change.path,
            change._original_bytes,
            expected_sha256=change.applied_sha256,
            expected_signature=change._signature,
            mode=change._mode,
        )