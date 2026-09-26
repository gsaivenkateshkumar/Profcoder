from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import pytest

from app.test_runner import (
    GitChangeReader,
    MAX_OUTPUT_BYTES_PER_STREAM,
    ProjectTestRunner,
    TestPreset,
    _ProcessCapture,
)


def make_runner(root, script, *, timeout=5, output_limit=4096):
    preset = TestPreset(Path(sys.executable), ("-c", script))
    return ProjectTestRunner(
        {"unit": preset}, timeout_seconds=timeout, output_limit_bytes=output_limit
    )


def test_runner_success_uses_project_cwd_and_strips_credentials(tmp_path, monkeypatch):
    root = tmp_path / "trusted-project"
    root.mkdir()
    monkeypatch.setenv("RUNNER_TEST_PROVIDER_API_KEY", "not-for-child")
    script = "import os; print(os.getcwd()); print('credential=' + str('RUNNER_TEST_PROVIDER_API_KEY' in os.environ))"

    result = make_runner(root, script).run(root, "unit")

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [str(root.resolve()), "credential=False"]
    assert result.stderr == ""
    assert result.elapsed_seconds < 5
    assert not result.timed_out
    assert result.launch_error is None


def test_runner_reports_failing_exit_code_and_stderr(tmp_path):
    script = "import sys; print('failure detail', file=sys.stderr); raise SystemExit(7)"
    result = make_runner(tmp_path, script).run(tmp_path, "unit")

    assert result.exit_code == 7
    assert result.stderr.strip() == "failure detail"
    assert not result.timed_out


def test_runner_bounds_noisy_stdout_and_stderr(tmp_path):
    script = "import sys; sys.stdout.write('o' * 20000); sys.stderr.write('e' * 20000)"
    result = make_runner(tmp_path, script, output_limit=1024).run(tmp_path, "unit")

    assert result.exit_code == 0
    assert len(result.stdout.encode("utf-8")) <= 1024
    assert len(result.stderr.encode("utf-8")) <= 1024
    assert result.stdout_truncated
    assert result.stderr_truncated


def _process_is_running(pid: int) -> bool:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        try:
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_timeout_terminates_child_process_tree(tmp_path):
    pid_file = tmp_path / "child-pid.txt"
    child_code = "import time; time.sleep(60)"
    parent_script = (
        "import pathlib,subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    result = make_runner(tmp_path, parent_script, timeout=1).run(tmp_path, "unit")

    assert result.timed_out
    assert result.launch_error is None
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_is_running(child_pid)


def test_runner_reports_launch_failure_and_rejects_dynamic_commands(tmp_path):
    missing = tmp_path / "missing-executable.exe"
    runner = ProjectTestRunner({"broken": TestPreset(missing, ())})
    result = runner.run(tmp_path, "broken")

    assert result.exit_code is None
    assert result.launch_error == "process_start_failed"
    with pytest.raises(TypeError):
        TestPreset(Path(sys.executable), "-c print('not a preset tuple')")
    with pytest.raises(ValueError, match="Unknown"):
        runner.run(tmp_path, "model-supplied")
    with pytest.raises(ValueError, match="absolute"):
        runner.run(Path("."), "broken")


def test_git_reader_uses_fixed_commands_and_excludes_sensitive_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "outside-git"))

    def fake_capture(arguments, **kwargs):
        args = tuple(arguments)
        calls.append(args)
        assert "GIT_DIR" not in kwargs["environment"]
        if "status" in args:
            output = b"## main\n M src/example.py\n"
        elif "--name-only" in args and "--cached" not in args:
            output = b"src/example.py\x00.env\x00"
        elif "--name-only" in args:
            output = b"docs/readme.md\x00credentials.json\x00"
        elif "--cached" in args:
            output = b"staged diff"
        else:
            output = b"working diff"
        return _ProcessCapture(0, output, b"", False, False, False, None, 0.01)

    monkeypatch.setattr("app.test_runner._capture_process", fake_capture)
    reader = GitChangeReader(Path(sys.executable))
    result = reader.read(tmp_path)

    assert result.status == "## main\n M src/example.py\n"
    assert result.working_diff == "working diff"
    assert result.staged_diff == "staged diff"
    assert not result.truncated
    working_diff_call = next(call for call in calls if "diff" in call and "--name-only" not in call and "--cached" not in call)
    staged_diff_call = next(call for call in calls if "diff" in call and "--name-only" not in call and "--cached" in call)
    assert "src/example.py" in working_diff_call
    assert working_diff_call.index("--literal-pathspecs") < working_diff_call.index("diff")
    assert "--no-renames" in working_diff_call
    assert ".env" not in working_diff_call
    assert "docs/readme.md" in staged_diff_call
    assert "credentials.json" not in staged_diff_call


def test_git_reader_stops_when_name_list_is_truncated(tmp_path, monkeypatch):
    def fake_capture(arguments, **kwargs):
        if "status" in arguments:
            return _ProcessCapture(0, b"## main\n", b"", False, False, False, None, 0.01)
        return _ProcessCapture(0, b"", b"", False, True, False, None, 0.01)

    monkeypatch.setattr("app.test_runner._capture_process", fake_capture)
    result = GitChangeReader(Path(sys.executable), output_limit_bytes=256).read(tmp_path)

    assert result.truncated
    assert result.working_diff == ""
    assert result.staged_diff == ""


def test_preset_and_runner_limits_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        TestPreset(Path("python"), ())
    with pytest.raises(ValueError, match="Timeout"):
        ProjectTestRunner({"unit": TestPreset(Path(sys.executable), ())}, timeout_seconds=301)
    with pytest.raises(ValueError, match="Output limit"):
        ProjectTestRunner({"unit": TestPreset(Path(sys.executable), ())}, output_limit_bytes=MAX_OUTPUT_BYTES_PER_STREAM + 1)