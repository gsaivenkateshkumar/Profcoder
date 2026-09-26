from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from app.file_edits import (
    MAX_FILE_BYTES,
    ProjectFileEditor,
    StaleFileError,
    UnsafeFileError,
    UnsupportedTextError,
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root, ProjectFileEditor(root)


def test_preview_apply_and_rollback_preserve_utf8_bom_and_crlf(project):
    root, editor = project
    target = root / "src" / "sample.py"
    target.parent.mkdir()
    original = b"\xef\xbb\xbffirst\r\nsecond\r\n"
    target.write_bytes(original)

    preview = editor.preview(
        "src/sample.py", "first\nupdated\n", expected_sha256=sha256(original)
    )
    assert "--- a/src/sample.py" in preview.diff
    assert "+++ b/src/sample.py" in preview.diff
    assert target.read_bytes() == original

    receipt = editor.apply(preview)
    applied = b"\xef\xbb\xbffirst\r\nupdated\r\n"
    assert target.read_bytes() == applied
    assert receipt.applied_sha256 == sha256(applied)

    editor.rollback(receipt)
    assert target.read_bytes() == original


def test_preview_rejects_wrong_original_hash(project):
    root, editor = project
    target = root / "sample.txt"
    target.write_text("original\n", encoding="utf-8")

    with pytest.raises(StaleFileError, match="hash"):
        editor.preview("sample.txt", "proposed\n", expected_sha256="0" * 64)
    assert target.read_text(encoding="utf-8") == "original\n"


def test_apply_rejects_file_changed_after_preview(project):
    root, editor = project
    target = root / "sample.txt"
    original = b"original\n"
    target.write_bytes(original)
    preview = editor.preview("sample.txt", "proposed\n", expected_sha256=sha256(original))
    later_edit = b"later user edit\n"
    target.write_bytes(later_edit)

    with pytest.raises(StaleFileError):
        editor.apply(preview)
    assert target.read_bytes() == later_edit


def test_rollback_refuses_to_overwrite_later_user_edit(project):
    root, editor = project
    target = root / "sample.txt"
    original = b"original\n"
    target.write_bytes(original)
    receipt = editor.apply(
        editor.preview("sample.txt", "applied\n", expected_sha256=sha256(original))
    )
    later_edit = b"later user edit\n"
    target.write_bytes(later_edit)

    with pytest.raises(StaleFileError, match="rollback refused"):
        editor.rollback(receipt)
    assert target.read_bytes() == later_edit


@pytest.mark.parametrize(
    "relative_path",
    [".env", ".env.local", ".git/config", ".venv/module.py", "credentials.json", "secrets/token.txt"],
)
def test_preview_rejects_excluded_paths(project, relative_path):
    root, editor = project
    target = root / Path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not displayed", encoding="utf-8")

    with pytest.raises(UnsafeFileError):
        editor.preview(relative_path, "changed", expected_sha256=sha256(b"not displayed"))


@pytest.mark.parametrize("relative_path", ["../outside.txt", "C:\\outside.txt"])
def test_preview_rejects_paths_outside_root(project, tmp_path, relative_path):
    root, editor = project
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(UnsafeFileError):
        editor.preview(relative_path, "changed", expected_sha256=sha256(b"outside"))
    assert outside.read_text(encoding="utf-8") == "outside"


def test_preview_rejects_binary_oversized_and_mixed_line_endings(project):
    root, editor = project
    binary = root / "binary.txt"
    binary.write_bytes(b"text\x00binary")
    control = root / "control.txt"
    control.write_bytes(b"text\x07binary")
    oversized = root / "large.txt"
    oversized.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    mixed = root / "mixed.txt"
    mixed.write_bytes(b"one\r\ntwo\n")

    for target in (binary, control, mixed):
        with pytest.raises(UnsupportedTextError):
            editor.preview(
                target.name,
                "changed",
                expected_sha256=sha256(target.read_bytes()),
            )

    with pytest.raises(UnsafeFileError, match="size limit"):
        editor.preview(
            oversized.name,
            "changed",
            expected_sha256=sha256(oversized.read_bytes()),
        )


def test_preview_rejects_windows_alias_of_excluded_directory(project):
    if os.name != "nt":
        pytest.skip("Windows path aliases are unavailable")
    root, editor = project
    metadata = root / ".git"
    metadata.mkdir()
    (metadata / "config").write_text("not displayed", encoding="utf-8")

    with pytest.raises(UnsafeFileError):
        editor.preview(
            ".git./config",
            "changed",
            expected_sha256=sha256(b"not displayed"),
        )


def test_preview_rejects_file_hard_linked_outside_root(project, tmp_path):
    root, editor = project
    target = root / "shared.txt"
    target.write_text("shared content", encoding="utf-8")
    outside = tmp_path / "outside-link.txt"
    try:
        os.link(target, outside)
    except OSError as exc:
        pytest.skip(f"file hard links unavailable: {exc}")

    with pytest.raises(UnsafeFileError, match="Hard-linked"):
        editor.preview(
            "shared.txt",
            "changed",
            expected_sha256=sha256(target.read_bytes()),
        )
    assert outside.read_text(encoding="utf-8") == "shared content"


def test_preview_rejects_outside_windows_junction(project, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows junctions are unavailable")
    root, editor = project
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "sample.txt"
    target.write_text("outside", encoding="utf-8")
    junction = root / "outside-link"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("Windows directory junction creation is unavailable")

    try:
        with pytest.raises(UnsafeFileError):
            editor.preview(
                "outside-link/sample.txt",
                "changed",
                expected_sha256=sha256(target.read_bytes()),
            )
    finally:
        subprocess.run(
            ["cmd.exe", "/d", "/c", "rmdir", str(junction)],
            capture_output=True,
            check=False,
        )