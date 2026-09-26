from __future__ import annotations

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.project_files import (
    MAX_EXCERPT_CHARS,
    MAX_FILE_BYTES,
    MAX_LIST_RESULTS,
    MAX_SCAN_BYTES,
    MAX_SEARCH_RESULTS,
    _is_reparse_point,
)


@pytest.fixture
def project_client(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("REPO_ROOT", str(project_root))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(app) as client:
        yield client, project_root


def test_project_search_returns_relative_path_line_and_excerpt(project_client):
    client, project_root = project_client
    source = project_root / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("first line\nvalue = 'needle in context'\n", encoding="utf-8")

    listing = client.get("/project/files").json()
    search = client.get("/project/search", params={"q": "needle"}).json()

    assert listing == {"files": ["src/module.py"], "truncated": False}
    assert search == {
        "results": [{"path": "src/module.py", "line": 2, "excerpt": "value = 'needle in context'"}],
        "truncated": False,
    }


def test_project_tools_exclude_secrets_caches_binary_and_large_files(project_client):
    client, root = project_client
    (root / "visible.txt").write_text("needle", encoding="utf-8")
    (root / ".env.local").write_text("needle", encoding="utf-8")
    (root / "credentials.json").write_text("needle", encoding="utf-8")
    (root / "access-token.json").write_text("needle", encoding="utf-8")
    (root / ".npmrc").write_text("needle", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "tracked.txt").write_text("needle", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "package.py").write_text("needle", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "module.pyc").write_bytes(b"needle")
    (root / "opaque.txt").write_bytes(b"needle\x00binary")
    (root / "large.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))

    listing = client.get("/project/files").json()
    search = client.get("/project/search", params={"q": "needle"}).json()

    assert listing == {"files": ["visible.txt"], "truncated": False}
    assert search["results"] == [{"path": "visible.txt", "line": 1, "excerpt": "needle"}]


def test_project_paths_reject_traversal(project_client, tmp_path):
    client, _ = project_client

    assert client.get("/project/files", params={"path": "../outside"}).status_code == 400
    assert client.get(
        "/project/search", params={"q": "needle", "path": "../outside"}
    ).status_code == 400


def test_outside_symlink_is_rejected(project_client, tmp_path):
    client, root = project_client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("needle", encoding="utf-8")
    junction = root / "outside-link"
    try:
        junction.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    assert client.get("/project/files", params={"path": "outside-link"}).status_code == 400
    assert client.get("/project/search", params={"q": "needle"}).json()["results"] == []


def test_windows_reparse_points_are_detected():
    metadata = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    assert _is_reparse_point(metadata)


def test_project_endpoints_enforce_result_and_query_limits(project_client):
    client, root = project_client
    (root / "000-matches.txt").write_text("hit\n" * (MAX_SEARCH_RESULTS + 5), encoding="utf-8")
    for index in range(MAX_LIST_RESULTS + 1):
        (root / f"file-{index:03}.txt").write_text("ordinary", encoding="utf-8")

    listing = client.get("/project/files").json()
    search = client.get("/project/search", params={"q": "hit"}).json()
    too_long = client.get("/project/search", params={"q": "q" * 161})

    assert len(listing["files"]) == MAX_LIST_RESULTS
    assert listing["truncated"] is True
    assert len(search["results"]) == MAX_SEARCH_RESULTS
    assert search["truncated"] is True
    assert all(len(result["excerpt"]) <= MAX_EXCERPT_CHARS for result in search["results"])
    assert too_long.status_code == 422


def test_project_endpoints_require_explicit_root(monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(app) as client:
        assert client.get("/project/files").status_code == 503
        assert client.get("/project/search", params={"q": "text"}).status_code == 503


def test_search_obeys_aggregate_scan_byte_limit(project_client):
    client, root = project_client
    file_count = MAX_SCAN_BYTES // MAX_FILE_BYTES + 2
    for index in range(file_count):
        (root / f"part-{index:02}.txt").write_bytes(b"x" * MAX_FILE_BYTES)

    response = client.get("/project/search", params={"q": "not-present"}).json()

    assert response["results"] == []
    assert response["truncated"] is True