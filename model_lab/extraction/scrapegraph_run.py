"""Method B: ScrapeGraphAI on the SAME saved snapshots with a local Ollama model.

Run only in the separate ScrapeGraphAI environment, from the repository root:

    F:\\profcoder-scrapegraph-venv\\Scripts\\python.exe -m model_lab.extraction.scrapegraph_run --page select

Safeguards: telemetry is disabled (environment flag and API call), and a socket
guard refuses every non-loopback connection, so neither page content nor
prompts can leave the machine and no page is re-fetched. Only a local Ollama
server on 127.0.0.1 is used; there is no cloud LLM and no provider bill.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import threading
import time
from pathlib import Path

PINNED_VERSION = "2.3.0"
DEFAULT_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://127.0.0.1:11434"
MAX_OUTPUT_TOKENS = 1024
PAGE_TIMEOUT_SECONDS = 900
PROMPT = (
    "From this SQLite documentation page, extract: title (the page title); headings "
    "(every section heading, exactly as written, in order); summary (the first sentence "
    "that explains what the statement does, copied verbatim); sql (every SQL statement, "
    "syntax form, or example shown on the page, copied exactly). Ignore navigation, "
    "menus, and syntax-diagram show/hide toggles. Do not invent content."
)

_blocked: list[str] = []


def _is_loopback(host: object) -> bool:
    if host in ("localhost", "::1"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def install_network_guard() -> list[str]:
    """Refuse non-loopback socket connections for the rest of this process."""
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
            _blocked.append(sock.family.name)
            raise PermissionError("network guard: non-loopback connection refused")

    def connect(sock, address):
        check(sock, address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        check(sock, address)
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    return _blocked


def normalize_answer(answer: object) -> dict[str, object]:
    """Coerce a ScrapeGraphAI answer into the shared fields; drop non-string values."""
    if isinstance(answer, dict) and set(answer) == {"content"} and isinstance(answer["content"], dict):
        answer = answer["content"]
    answer = answer if isinstance(answer, dict) else {}

    def text(value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    def texts(value):
        values = value if isinstance(value, list) else [value]
        return [t for t in map(text, values) if t is not None]

    return {
        "title": text(answer.get("title")),
        "headings": texts(answer.get("headings")),
        "summary": text(answer.get("summary")),
        "sql": texts(answer.get("sql")),
    }


class ResourceSampler(threading.Thread):
    """Poll this process and local Ollama processes for peak RSS and CPU time."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        import psutil

        self.psutil, self.interval = psutil, interval
        self.stop_event = threading.Event()
        self.peak_self = self.peak_ollama = 0
        self.cpu_start = self._ollama_cpu()

    def _ollama(self):
        # The Ollama server delegates inference to a llama-server child process.
        for process in self.psutil.process_iter(["name"]):
            if (process.info["name"] or "").lower().startswith(("ollama", "llama-server")):
                yield process

    def _ollama_cpu(self) -> float:
        total = 0.0
        for process in self._ollama():
            try:
                times = process.cpu_times()
                total += times.user + times.system
            except self.psutil.Error:
                pass
        return total

    def run(self):
        me = self.psutil.Process()
        while not self.stop_event.is_set():
            self.peak_self = max(self.peak_self, me.memory_info().rss)
            ollama = 0
            for process in self._ollama():
                try:
                    ollama += process.memory_info().rss
                except self.psutil.Error:
                    pass
            self.peak_ollama = max(self.peak_ollama, ollama)
            self.stop_event.wait(self.interval)

    def finish(self) -> dict[str, object]:
        self.stop_event.set()
        self.join()
        return {
            "peak_process_rss_bytes": self.peak_self,
            "peak_ollama_rss_bytes": self.peak_ollama,
            "ollama_cpu_seconds": round(self._ollama_cpu() - self.cpu_start, 2),
        }


def run_page(
    pilot_dir: Path, page: str, model: str, output_dir: Path,
    timeout: float = PAGE_TIMEOUT_SECONDS,
) -> dict[str, object]:
    os.environ["SCRAPEGRAPHAI_TELEMETRY_ENABLED"] = "false"
    blocked = install_network_guard()

    from importlib.metadata import version

    from pydantic import BaseModel
    from scrapegraphai.graphs import SmartScraperGraph
    from scrapegraphai.telemetry import disable_telemetry
    from scrapegraphai.telemetry.telemetry import is_telemetry_enabled

    from model_lab.extraction.parse import load_snapshot

    if version("scrapegraphai") != PINNED_VERSION:
        raise RuntimeError(f"expected scrapegraphai {PINNED_VERSION}")
    disable_telemetry()

    class PilotFields(BaseModel):
        title: str
        headings: list[str]
        summary: str
        sql: list[str]

    html, record = load_snapshot(pilot_dir, page)
    if html.lstrip().lower().startswith("http"):
        raise ValueError("source must be local HTML content, not a URL")
    config = {
        "llm": {
            "model": f"ollama/{model}", "model_tokens": 8192, "num_ctx": 8192,
            "temperature": 0, "seed": 7, "base_url": OLLAMA_URL,
            "num_gpu": 0,  # CPU-only budget: never offload layers to a GPU
            "num_predict": MAX_OUTPUT_TOKENS,  # an uncapped call once generated 12,700+ tokens
        },
        "verbose": False,
        "headless": True,
    }
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    outcome: dict[str, object] = {"answer": None, "error": None, "exec_info": None}

    def work():
        try:
            graph = SmartScraperGraph(prompt=PROMPT, source=html, config=config, schema=PilotFields)
            if graph.input_key != "local_dir":
                raise RuntimeError("ScrapeGraphAI did not treat the snapshot as local content")
            outcome["answer"] = graph.run()
            outcome["exec_info"] = graph.get_execution_info()
        except Exception as exc:  # recorded, not hidden: the comparison reports failures
            outcome["error"] = f"{type(exc).__name__}: {exc}"[:500]

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout)
    timed_out = worker.is_alive()
    if timed_out:
        outcome["error"] = f"timed out after {timeout} s"
    answer, error, exec_info = outcome["answer"], outcome["error"], outcome["exec_info"]
    elapsed = time.perf_counter() - started
    resources = sampler.finish()

    totals = next((row for row in exec_info or [] if row.get("node_name") == "TOTAL RESULT"), {})
    result = {
        "method": f"scrapegraphai-{PINNED_VERSION}+ollama/{model}",
        "page": page,
        "source": {k: record[k] for k in ("url", "sha256", "html_file", "text_file", "text_sha256")},
        "fields": normalize_answer(answer),
        "raw_answer": answer,
        "elapsed_seconds": round(elapsed, 2),
        **resources,
        "llm": {
            "provider": "ollama (local, 127.0.0.1)",
            "model": model,
            "prompt_tokens": totals.get("prompt_tokens"),
            "completion_tokens": totals.get("completion_tokens"),
            "total_tokens": totals.get("total_tokens"),
            "successful_requests": totals.get("successful_requests"),
            "reported_cost_usd": totals.get("total_cost_USD"),
            "provider_charge": "none: local model, no API account",
        },
        "execution_info": exec_info,
        "telemetry_enabled": is_telemetry_enabled(),
        "network_connections_blocked": len(blocked),
        "error": error,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{page}.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("model_lab/runs/extraction-pilot-v1"))
    parser.add_argument("--page", choices=("select", "insert"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = run_page(args.pilot, args.page, args.model, args.pilot / "outputs" / "scrapegraphai")
    print(json.dumps({k: result[k] for k in (
        "page", "elapsed_seconds", "peak_process_rss_bytes", "peak_ollama_rss_bytes",
        "ollama_cpu_seconds", "telemetry_enabled", "network_connections_blocked", "error",
    )} | {"llm": result["llm"]}, indent=2))
    if str(result["error"]).startswith("timed out"):
        # Exiting closes the HTTP connection, which makes the local server abandon the call.
        os._exit(3)


if __name__ == "__main__":
    main()
