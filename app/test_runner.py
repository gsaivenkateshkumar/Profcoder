from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Mapping, Sequence

from app.project_files import _is_excluded


MAX_TIMEOUT_SECONDS = 300.0
MAX_OUTPUT_BYTES_PER_STREAM = 64 * 1024
MAX_GIT_TIMEOUT_SECONDS = 30.0
MAX_GIT_OUTPUT_BYTES = 64 * 1024
MAX_TEST_PRESETS = 32
MAX_PRESET_ARGUMENTS = 128
MAX_PRESET_ARGUMENT_CHARS = 8192

_CREDENTIAL_ENV_MARKERS = (
    "API_KEY", "APIKEY", "API_TOKEN", "ACCESS_KEY", "ACCESS_TOKEN",
    "SECRET", "CREDENTIAL", "PASSWORD", "AUTHORIZATION", "BEARER",
    "_TOKEN", "TOKEN_", "_PAT",
)
_CREDENTIAL_ENV_NAMES = {"SSH_AUTH_SOCK", "SSH_AGENT_PID", "GIT_ASKPASS", "GIT_SSH_COMMAND"}


@dataclass(frozen=True)
class TestPreset:
    __test__ = False

    executable_path: Path
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        executable = Path(self.executable_path).expanduser()
        if not executable.is_absolute():
            raise ValueError("Preset executable path must be absolute")
        if not isinstance(self.arguments, tuple):
            raise TypeError("Preset arguments must be a fixed tuple of strings")
        if len(self.arguments) > MAX_PRESET_ARGUMENTS:
            raise ValueError("Preset has too many arguments")
        if any(not isinstance(argument, str) or "\x00" in argument for argument in self.arguments):
            raise ValueError("Preset arguments must be NUL-free strings")
        if sum(len(argument) for argument in self.arguments) > MAX_PRESET_ARGUMENT_CHARS:
            raise ValueError("Preset arguments exceed the size limit")
        if os.name == "nt" and executable.suffix.casefold() in {".bat", ".cmd"}:
            raise ValueError("Batch files are not supported as preset executables")
        object.__setattr__(self, "executable_path", executable.resolve(strict=False))


@dataclass(frozen=True)
class TestRunResult:
    __test__ = False

    exit_code: int | None
    elapsed_seconds: float
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    launch_error: str | None


@dataclass(frozen=True)
class GitReviewResult:
    status: str
    working_diff: str
    staged_diff: str
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class _ProcessCapture:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    launch_error: str | None
    elapsed_seconds: float


def _credential_variable(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in _CREDENTIAL_ENV_NAMES or any(
        marker in upper_name for marker in _CREDENTIAL_ENV_MARKERS
    )


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if _credential_variable(name):
            environment.pop(name, None)
    return environment


def _drain_pipe(stream, output: bytearray, truncated: list[bool], limit: int) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            remaining = limit - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True
    except (OSError, ValueError):
        pass


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        taskkill = Path(system_root) / "System32" / "taskkill.exe"
        if taskkill.is_file():
            try:
                subprocess.run(
                    [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_child_environment(),
                    shell=False,
                    timeout=3.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass

    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _capture_process(
    arguments: Sequence[str],
    *,
    working_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    output_limit_bytes: int,
) -> _ProcessCapture:
    started = time.monotonic()
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        process = subprocess.Popen(
            list(arguments),
            cwd=working_directory,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **options,
        )
    except (OSError, ValueError):
        return _ProcessCapture(None, b"", b"", False, False, False, "process_start_failed", time.monotonic() - started)

    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = [False]
    stderr_truncated = [False]
    readers = [
        threading.Thread(target=_drain_pipe, args=(process.stdout, stdout, stdout_truncated, output_limit_bytes), daemon=True),
        threading.Thread(target=_drain_pipe, args=(process.stderr, stderr, stderr_truncated, output_limit_bytes), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)

    for reader in readers:
        reader.join(timeout=1.0)
    if any(reader.is_alive() for reader in readers):
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
        for reader in readers:
            reader.join(timeout=0.2)

    return _ProcessCapture(
        process.poll(),
        bytes(stdout),
        bytes(stderr),
        timed_out,
        stdout_truncated[0],
        stderr_truncated[0],
        None,
        time.monotonic() - started,
    )


def _resolve_project_root(project_root: Path) -> Path:
    selected_root = Path(project_root).expanduser()
    if not selected_root.is_absolute():
        raise ValueError("Selected project root must be absolute")
    try:
        root = selected_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Selected project root is unavailable") from exc
    if not root.is_dir():
        raise ValueError("Selected project root must be a directory")
    return root


class ProjectTestRunner:
    """Run only named, owner-configured presets in a selected trusted project."""

    def __init__(
        self,
        presets: Mapping[str, TestPreset],
        *,
        timeout_seconds: float = 60.0,
        output_limit_bytes: int = MAX_OUTPUT_BYTES_PER_STREAM,
    ):
        if not presets or len(presets) > MAX_TEST_PRESETS:
            raise ValueError("Configure between one and 32 test presets")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("Timeout must be between 0 and 300 seconds")
        if (
            isinstance(output_limit_bytes, bool)
            or not isinstance(output_limit_bytes, int)
            or not 1 <= output_limit_bytes <= MAX_OUTPUT_BYTES_PER_STREAM
        ):
            raise ValueError("Output limit must be between 1 and 65536 bytes per stream")
        for name, preset in presets.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
                raise ValueError("Preset names must be simple identifiers")
            if not isinstance(preset, TestPreset):
                raise TypeError("Each configured preset must be a TestPreset")
        self._presets = MappingProxyType(dict(presets))
        self._timeout_seconds = float(timeout_seconds)
        self._output_limit_bytes = output_limit_bytes

    def run(self, project_root: Path, preset_name: str) -> TestRunResult:
        if not isinstance(preset_name, str) or preset_name not in self._presets:
            raise ValueError("Unknown configured test preset")
        root = _resolve_project_root(project_root)
        preset = self._presets[preset_name]
        capture = _capture_process(
            (str(preset.executable_path), *preset.arguments),
            working_directory=root,
            environment=_child_environment(),
            timeout_seconds=self._timeout_seconds,
            output_limit_bytes=self._output_limit_bytes,
        )
        return TestRunResult(
            exit_code=capture.exit_code,
            elapsed_seconds=round(capture.elapsed_seconds, 3),
            stdout=capture.stdout.decode("utf-8", errors="replace"),
            stderr=capture.stderr.decode("utf-8", errors="replace"),
            timed_out=capture.timed_out,
            stdout_truncated=capture.stdout_truncated,
            stderr_truncated=capture.stderr_truncated,
            launch_error=capture.launch_error,
        )


def _reviewable_git_paths(name_output: bytes) -> list[str]:
    paths: list[str] = []
    for raw_path in name_output.split(b"\x00"):
        if not raw_path:
            continue
        relative_path = os.fsdecode(raw_path).replace("\\", "/")
        windows_path = PureWindowsPath(relative_path)
        parts = relative_path.split("/")
        if (
            windows_path.is_absolute()
            or windows_path.drive
            or ":" in relative_path
            or ".." in parts
            or any(_is_excluded(part, is_directory=index < len(parts) - 1) for index, part in enumerate(parts))
        ):
            continue
        paths.append(relative_path)
    return paths


class GitChangeReader:
    """Read bounded status and safe tracked diffs without invoking Git hooks/tools."""

    def __init__(
        self,
        git_executable: Path,
        *,
        timeout_seconds: float = 10.0,
        output_limit_bytes: int = MAX_GIT_OUTPUT_BYTES,
    ):
        executable = Path(git_executable).expanduser()
        if not executable.is_absolute():
            raise ValueError("Git executable path must be absolute")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= MAX_GIT_TIMEOUT_SECONDS
        ):
            raise ValueError("Git timeout must be between 0 and 30 seconds")
        if (
            isinstance(output_limit_bytes, bool)
            or not isinstance(output_limit_bytes, int)
            or not 1 <= output_limit_bytes <= MAX_GIT_OUTPUT_BYTES
        ):
            raise ValueError("Git output limit must be between 1 and 65536 bytes")
        self._executable = executable.resolve(strict=False)
        self._timeout_seconds = float(timeout_seconds)
        self._output_limit_bytes = output_limit_bytes

    def _git(self, root: Path, arguments: Sequence[str]) -> _ProcessCapture:
        environment = _child_environment()
        for name in tuple(environment):
            if name.upper().startswith("GIT_"):
                environment.pop(name, None)
        environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"})
        return _capture_process(
            (str(self._executable), "-c", "core.fsmonitor=false", "--no-pager", *arguments),
            working_directory=root,
            environment=environment,
            timeout_seconds=self._timeout_seconds,
            output_limit_bytes=self._output_limit_bytes,
        )

    def read(self, project_root: Path) -> GitReviewResult:
        root = _resolve_project_root(project_root)
        status = self._git(root, ("status", "--short", "--branch", "--untracked-files=normal"))
        status_text = status.stdout.decode("utf-8", errors="replace")
        truncated = status.stdout_truncated or status.stderr_truncated
        if status.launch_error or status.timed_out or status.exit_code != 0:
            return GitReviewResult(status_text, "", "", truncated, "git_status_failed")

        working_names = self._git(root, ("diff", "--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv"))
        staged_names = self._git(root, ("diff", "--cached", "--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv"))
        name_results = (working_names, staged_names)
        truncated = truncated or any(result.stdout_truncated or result.stderr_truncated for result in name_results)
        if any(result.launch_error or result.timed_out or result.exit_code != 0 for result in name_results):
            return GitReviewResult(status_text, "", "", False, "git_diff_failed")
        if truncated:
            return GitReviewResult(status_text, "", "", True, None)

        working_paths = _reviewable_git_paths(working_names.stdout)
        staged_paths = _reviewable_git_paths(staged_names.stdout)
        working_diff = self._read_diff(root, working_paths, staged=False)
        staged_diff = self._read_diff(root, staged_paths, staged=True)
        truncated = truncated or working_diff.stdout_truncated or working_diff.stderr_truncated
        truncated = truncated or staged_diff.stdout_truncated or staged_diff.stderr_truncated
        if any(result.launch_error or result.timed_out or result.exit_code != 0 for result in (working_diff, staged_diff)):
            return GitReviewResult(status_text, "", "", truncated, "git_diff_failed")
        return GitReviewResult(
            status_text,
            working_diff.stdout.decode("utf-8", errors="replace"),
            staged_diff.stdout.decode("utf-8", errors="replace"),
            truncated,
            None,
        )

    def _read_diff(self, root: Path, paths: list[str], *, staged: bool) -> _ProcessCapture:
        if not paths:
            return _ProcessCapture(0, b"", b"", False, False, False, None, 0.0)
        arguments = ["--literal-pathspecs", "diff"]
        if staged:
            arguments.append("--cached")
        arguments.extend(("--no-renames", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=2", "--", *paths))
        return self._git(root, arguments)