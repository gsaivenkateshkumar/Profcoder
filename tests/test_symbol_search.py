from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.project_files import MAX_FILE_BYTES, MAX_SCAN_BYTES, MAX_SEARCH_RESULTS, ProjectPathError
from app.symbol_search import find_definitions


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")  # exact byte sizes on Windows
    return path


def locations(result: dict[str, object]) -> list[tuple[str, int, str, str]]:
    return [(r["path"], r["line"], r["kind"], r["qualified_name"]) for r in result["results"]]


SOURCE = '''\
"""target appears in this docstring but is not defined here."""
import functools


def target():
    return "target()"  # a call-like string


@functools.lru_cache
def cached(value):
    return target()


async def fetch_target():
    pass


class Shape:
    def target(self):
        def target():
            return 1
        return target()

    async def area(self):
        return 0

    class Inner:
        def target(self):
            pass


if True:
    class Guarded:
        pass

try:
    def in_try():
        pass
except Exception:
    pass

Target = 3
'''


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "pkg/module.py", SOURCE)
    return root


def test_finds_every_definition_with_exact_lines_and_kinds(project):
    assert locations(find_definitions(project, "target")) == [
        ("pkg/module.py", 5, "function", "target"),
        ("pkg/module.py", 19, "method", "Shape.target"),
        ("pkg/module.py", 20, "function", "Shape.target.target"),
        ("pkg/module.py", 28, "method", "Shape.Inner.target"),
    ]
    # Decorated definitions report the def line, not the decorator line.
    assert locations(find_definitions(project, "cached")) == [("pkg/module.py", 10, "function", "cached")]
    assert locations(find_definitions(project, "fetch_target"))[0][2] == "async function"
    assert locations(find_definitions(project, "area"))[0][2] == "async method"
    assert locations(find_definitions(project, "Guarded")) == [("pkg/module.py", 33, "class", "Guarded")]
    assert locations(find_definitions(project, "in_try")) == [("pkg/module.py", 37, "function", "in_try")]


def test_qualified_names_case_and_non_definitions(project):
    assert locations(find_definitions(project, "Shape.Inner.target")) == [
        ("pkg/module.py", 28, "method", "Shape.Inner.target")
    ]
    assert find_definitions(project, "Shape.missing")["results"] == []
    # Case-sensitive, and assignments, calls, strings, and docstrings are not definitions.
    assert find_definitions(project, "Target")["results"] == []
    assert find_definitions(project, "functools")["results"] == []
    excerpt = find_definitions(project, "target")["results"][0]["excerpt"]
    assert excerpt == "def target():"


def test_source_is_parsed_not_executed(tmp_path):
    root = tmp_path / "project"
    marker = tmp_path / "executed.txt"
    write(root, "danger.py", f"open({str(marker)!r}, 'w').write('ran')\nraise SystemExit(3)\ndef safe():\n    pass\n")
    assert locations(find_definitions(root, "safe")) == [("danger.py", 3, "function", "safe")]
    assert not marker.exists()


def test_unparseable_and_non_python_files_are_skipped(tmp_path):
    root = tmp_path / "project"
    write(root, "broken.py", "def target(:\n")
    write(root, "notes.txt", "def target():\n")
    write(root, "ok.py", "def target():\n    pass\n")
    result = find_definitions(root, "target")
    assert locations(result) == [("ok.py", 1, "function", "target")]
    assert result["unparsed_files"] == 1


def test_non_ascii_identifiers_match_their_normalized_name(tmp_path):
    root = tmp_path / "project"
    write(root, "lig.py", "def \ufb01nd():\n    pass\n")  # U+FB01 ligature normalizes to "fi"
    assert locations(find_definitions(root, "find")) == [("lig.py", 1, "function", "find")]


def test_exclusions_and_scope_match_search_text(tmp_path):
    root = tmp_path / "project"
    write(root, "visible.py", "def target():\n    pass\n")
    for hidden in (".venv/lib.py", ".git/hook.py", "__pycache__/cached.py", "node_modules/pkg/x.py",
                   "site-packages/mod.py", "my_secret_helpers.py", "api_token.py", ".env.py"):
        write(root, hidden, "def target():\n    pass\n")
    write(root, "sub/inner.py", "def target():\n    pass\n")
    assert {r["path"] for r in find_definitions(root, "target")["results"]} == {"sub/inner.py", "visible.py"}
    assert [r["path"] for r in find_definitions(root, "target", "sub")["results"]] == ["sub/inner.py"]
    for bad_scope in ("../outside", str(tmp_path), "C:/Windows", "missing"):
        with pytest.raises(ProjectPathError):
            find_definitions(root, "target", bad_scope)


def test_outside_symlink_is_not_followed(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    write(outside, "private.py", "def target():\n    pass\n")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        if sys.platform != "win32":
            pytest.skip(f"directory symlink creation unavailable: {exc}")
        # Directory junctions need no privilege and are reparse points too.
        import subprocess

        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                              capture_output=True, text=True)
        if made.returncode != 0 or not link.exists():
            pytest.skip("neither a symlink nor a junction could be created")
    assert (link / "private.py").exists()  # the link really points outside
    assert find_definitions(root, "target")["results"] == []
    with pytest.raises(ProjectPathError):
        find_definitions(root, "target", "link")


def test_rejects_invalid_names(project):
    for bad in ("", "1abc", "class", "a..b", "a.", "a b", "x" * 161, "a-b", "def target"):
        with pytest.raises(ProjectPathError):
            find_definitions(project, bad)
    with pytest.raises(ProjectPathError):
        find_definitions(project, None)  # type: ignore[arg-type]


def test_size_scan_and_result_bounds(tmp_path):
    root = tmp_path / "project"
    big = "def target():\n    pass\n" + "#" * MAX_FILE_BYTES
    write(root, "big.py", big)
    assert find_definitions(root, "target")["results"] == []  # over the per-file limit

    many = tmp_path / "many"
    write(many, "defs.py", "".join(f"class C{i}:\n    def target(self):\n        pass\n" for i in range(MAX_SEARCH_RESULTS + 5)))
    result = find_definitions(many, "target")
    assert len(result["results"]) == MAX_SEARCH_RESULTS
    assert result["truncated"] is True

    scan = tmp_path / "scan"
    filler = "x = 1\n" * (MAX_FILE_BYTES // 6 - 10)  # each file stays under the per-file limit
    for index in range(MAX_SCAN_BYTES // MAX_FILE_BYTES + 2):
        write(scan, f"part_{index:02}.py", filler + "def target():\n    pass\n")
    bounded = find_definitions(scan, "target")
    assert bounded["truncated"] is True
    assert len(bounded["results"]) < MAX_SCAN_BYTES // MAX_FILE_BYTES + 2
