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
```

Search returns relative paths, one-based line numbers, and bounded excerpts. The
read-only endpoints exclude credentials, environment files, VCS data, virtual
environments, caches, binary and oversized files; indexed content is not sent to
Groq. Requests are bounded by path, scan, file, result, and excerpt limits. If
`REPO_ROOT` is unset or invalid, these endpoints return `503`.

## Status
This repository is in the initial project setup stage. Dependencies, credentials, and live deployments have not been installed or configured.
