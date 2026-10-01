# Profcoder

Profcoder is a personal coding agent served from this computer to other devices. It combines an online Groq backend with a small local model for limited offline operation, while keeping repository work, research, and memory under explicit control.

## Goals
- Online coding via Groq-backed model provider
- Authenticated device client access from other devices
- Code search and targeted code navigation
- Controlled edit and validation tools
- Researched and verified memory with auditability
- Offline fallback for limited local assistance
- Replaceable model providers and later evaluation loops

## Original-model experiments
The isolated [model lab](model_lab/README.md) contains a reviewed toy corpus,
reversible byte tokenizer, and a small, randomly initialized CPU Transformer
with bounded training. The demo corpus is not enough to build a useful coding
assistant; the existing online agent remains separate from this experiment.

## Architecture snapshot
- `server/`: remote service entrypoint and orchestration
- `client/`: authenticated device client and transport layer
- `tools/`: code search, repo operations, edit/test helpers
- `memory/`: verified memory store and retrieval
- `providers/`: model provider abstractions for Groq and local models
- `evals/`: later evaluation harness and benchmarking

## Phase 0 scope
- Detect host environment and installed tooling
- Initialize the project skeleton
- Document roadmap and security boundaries
- Prepare Git hygiene and environment template

## Security
- Never commit secrets or local credentials
- Use environment variables only
- Keep `.env` out of source control
- Prefer explicit permissions for device authentication and repo access

## Local read-only project tools
Set `REPO_ROOT` in the project `.env` to the single directory the server may browse:

```dotenv
REPO_ROOT=F:\code\my-project
```

Start the server locally with:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

The server stays bound to `127.0.0.1`. List files or search text using paths
relative to that root:

```text
GET http://127.0.0.1:8001/project/files?path=src
GET http://127.0.0.1:8001/project/search?q=TODO&path=src
GET http://127.0.0.1:8001/project/definitions?name=Class.method&path=src
```

Search returns relative paths, one-based line numbers, and bounded excerpts.
`/project/definitions` looks up a Python `def`/`class` by exact name (or a
dotted qualified name such as `Class.method`) using `ast` parsing only; it
never executes any file, works with no `GROQ_API_KEY` set, and makes no
provider call — this is offline code navigation, not generation by the
from-scratch Profcoder model in `model_lab/`. The read-only endpoints exclude
credentials, environment files, VCS data, virtual environments, caches, binary
and oversized files. They do not send indexed content automatically. Requests
are bounded by path, scan, file, result, and excerpt limits. If `REPO_ROOT` is
unset or invalid, these endpoints return `503`.

## Local project inspection
`POST http://127.0.0.1:8001/agent/inspect` accepts `{"question":"..."}` and
uses Groq function calling with read-only project listing, text search, Python
definition lookup, safe text-file reading, and (when configured) bounded Git
review. Set
`PROFCODER_GIT_EXECUTABLE` to an absolute Git executable path to enable Git
review; leave it unset to omit that tool. The endpoint has no editing, rollback,
test-running, shell, or web-research tools.

Retrieved source snippets and selected Git diffs are included in the online Groq
conversation to answer the question. Use only project roots whose readable code
you are willing to send to Groq. Requests are capped at 12 KiB, five model turns,
six tool calls, 64 KiB serialized context, and a 12 KiB JSON response. File reads
are limited to 100 lines and 6 KiB of text. The API remains bound to `127.0.0.1`.

The `find_definitions` tool (`app/symbol_search.py`) finds where a Python
function, method, or class is defined, by exact name or dotted name such as
`Config.load`. It returns path, line, kind, and qualified name. The lookup itself
runs offline: it parses `.py` files with Python's `ast` module without executing
them, and it uses the same path exclusions, symlink/junction refusal, and file,
scan, and result limits as text search. It is still one tool inside
`/agent/inspect`, so using it **requires a configured Groq key**, and its results
are sent to Groq like other tool results. It does not connect the from-scratch
`model_lab` model to the agent. On a fixed 16-task, 31-location benchmark, it
found 31/31 definitions with no extra hits. Text search for the bare name found
22/31 and returned 354 non-definition lines.

```powershell
$body = @{ question = "Where is the chat route defined?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8001/agent/inspect `
	-Method Post -ContentType application/json -Body $body
```

## Internal file-editing interface
`app.file_edits.ProjectFileEditor` is an internal Python interface only. It adds
no HTTP route and is not callable by Groq. Construct it with the project root
resolved from `REPO_ROOT`, then provide the original file's SHA-256 over raw
bytes:

```python
import hashlib
import os
from pathlib import Path

from app.file_edits import ProjectFileEditor

project_root = Path(os.environ["REPO_ROOT"]).resolve(strict=True)
target = project_root / "src/example.py"
proposed_text = target.read_text(encoding="utf-8").replace("old", "new")
original_hash = hashlib.sha256(target.read_bytes()).hexdigest()
editor = ProjectFileEditor(project_root)
preview = editor.preview("src/example.py", proposed_text, expected_sha256=original_hash)
preview.diff
receipt = editor.apply(preview)
editor.rollback(receipt)
```

The editor reuses project path exclusions and rejects symlinks, Windows
junctions, hard links, binary files, unsupported encodings, excluded paths, and
paths outside the root. Supported files are UTF-8 (with or without BOM), use
uniform LF, CRLF, or CR endings, and are at most 500 KiB. Apply checks the
expected content hash and file signature before an atomic same-directory
replacement. Rollback receipts are in-memory and tied to one editor instance;
rollback refuses if the applied file has changed. There is no persistent undo
journal. A hostile process that swaps parent directories in the narrow interval
between validation and replacement remains a filesystem race; explicit NTFS
file ACLs are not preserved (mode bits are).

## Internal test runner and Git review
`app.test_runner.ProjectTestRunner` accepts only owner-defined named presets.
Each `TestPreset` stores an absolute executable path and an immutable tuple of
arguments; `run(project_root, preset_name)` accepts neither command strings nor
per-call arguments. Example owner configuration:

```python
from pathlib import Path

from app.test_runner import ProjectTestRunner, TestPreset

presets = {
	"unit": TestPreset(
		Path(r"F:\profcoder\.venv\Scripts\python.exe"),
		("-m", "pytest", "-q"),
	),
}
runner = ProjectTestRunner(presets, timeout_seconds=120)
result = runner.run(project_root, "unit")
```

Presets are limited to 32; timeouts are capped at 300 seconds and stdout/stderr
at 64 KiB each (lower limits can be configured). Provider credential variables
are removed from the child environment. A timeout kills the process tree on
Windows when `taskkill.exe` is available. Running a preset executes code from
the selected project with the current user's permissions; this runner is not an
OS sandbox. No HTTP execution route or Groq tool integration is provided.

`GitChangeReader` is also internal and read-only. It runs fixed status and diff
operations with external diff/text conversion disabled, excludes sensitive
project paths from diff content, and bounds each captured stream and command
time. It does not execute project commands.

## Status
The FastAPI app has a Groq provider and bounded local project tools. The
original-model experiment has a corpus/tokenizer preparation stage only;
training, checkpoints, evaluation, quantization, and offline inference are
future milestones.
