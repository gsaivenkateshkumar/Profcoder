"""Recheck rights and robots.txt, then save each allowlisted pilot page exactly once.

No crawling: only the fixed URLs below are requested, redirects are refused,
responses are size-capped, and requests are spaced out. Output goes to a new
(ignored) directory and is never overwritten. Nothing here is training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from model_lab.extraction.text import TEXT_VERSION, readable

HOST = "sqlite.org"
RIGHTS_URL = "https://sqlite.org/copyright.html"
ROBOTS_URL = "https://sqlite.org/robots.txt"
PAGES = {
    "select": "https://sqlite.org/lang_select.html",
    "insert": "https://sqlite.org/lang_insert.html",
}
USER_AGENT = (
    "ProfcoderExtractionPilot/1.0 (+https://github.com/gsaivenkateshkumar/Profcoder; "
    "one-time fetch of two documentation pages)"
)
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 20
MIN_DELAY_SECONDS = 5.0


@dataclass(frozen=True)
class Response:
    url: str
    status: int
    content_type: str
    charset: str | None
    body: bytes
    retrieved_utc: str


Fetcher = Callable[[str], Response]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def http_fetch(url: str) -> Response:
    """GET one allowlisted https URL; refuse redirects and bodies over MAX_BYTES."""
    _check_url(url)
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BYTES + 1)
            status, headers = response.status, response.headers
    except urllib.error.HTTPError as error:
        status, headers, body = error.code, error.headers or Message(), b""
    if len(body) > MAX_BYTES:
        raise ValueError(f"{url} exceeds {MAX_BYTES} bytes")
    return Response(
        url=url, status=status,
        content_type=(headers.get_content_type() if headers else ""),
        charset=(headers.get_content_charset() if headers else None),
        body=body,
        retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    allowed = {RIGHTS_URL, ROBOTS_URL, *PAGES.values()}
    if url not in allowed or parts.scheme != "https" or parts.hostname != HOST:
        raise ValueError("URL is not on the pilot allowlist")


def _decode(response: Response) -> str:
    return response.body.decode(response.charset or "utf-8", errors="strict")


def _record(response: Response) -> dict[str, object]:
    return {
        "url": response.url, "status": response.status,
        "content_type": response.content_type, "retrieved_utc": response.retrieved_utc,
        "bytes": len(response.body), "sha256": hashlib.sha256(response.body).hexdigest(),
    }


RIGHTS_STATEMENT = re.compile(r"(?is)\bdocumentation\b.{0,200}\bpublic domain\b|\bpublic domain\b.{0,200}\bdocumentation\b")


def check_access(fetch: Fetcher, wait: float) -> tuple[dict[str, object], dict[str, bytes]]:
    """Return (checks, evidence files). Raises if rights or robots rules disallow the pilot."""
    rights = fetch(RIGHTS_URL)
    if rights.status != 200 or rights.content_type != "text/html":
        raise PermissionError("rights page unavailable; source stopped")
    rights_text = readable(_decode(rights))
    statement = next(
        (b for b in rights_text.blocks
         if b.kind == "paragraph" and RIGHTS_STATEMENT.search(rights_text.block_text(b))),
        None,
    )
    if statement is None:
        raise PermissionError("rights page no longer states the documentation is public domain")
    time.sleep(wait)

    robots = fetch(ROBOTS_URL)
    if robots.status != 200 or robots.content_type != "text/plain":
        raise PermissionError("robots.txt unavailable; source stopped")
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(_decode(robots).splitlines())
    allowed = {name: parser.can_fetch(USER_AGENT, url) for name, url in PAGES.items()}
    delay = parser.crawl_delay(USER_AGENT)
    checks = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rights": {
            **_record(rights), "evidence_file": "access/copyright.html",
            "statement_offsets": [statement.start, statement.end],
            "statement_sha256": hashlib.sha256(
                rights_text.block_text(statement).encode("utf-8")
            ).hexdigest(),
            "basis": "public-domain statement (no license identifier)",
        },
        "robots": {**_record(robots), "evidence_file": "access/robots.txt",
                   "user_agent": USER_AGENT, "can_fetch": allowed, "crawl_delay": delay},
    }
    return checks, {"access/copyright.html": rights.body, "access/robots.txt": robots.body}


def run(output_dir: Path, fetch: Fetcher = http_fetch, *, delay: float | None = None) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("pilot output exists; each page is saved once, never overwritten")
    wait = MIN_DELAY_SECONDS if delay is None else delay
    files: dict[str, bytes] = {}
    checks, evidence = check_access(fetch, wait)
    files.update(evidence)
    wait = max(wait, float(checks["robots"]["crawl_delay"] or 0))
    output_dir.mkdir(parents=True, exist_ok=False)
    pages: dict[str, object] = {}
    for name, url in PAGES.items():
        if not checks["robots"]["can_fetch"][name]:
            pages[name] = {"url": url, "skipped": "disallowed by robots.txt"}
            continue
        time.sleep(wait)
        response = fetch(url)
        if response.status != 200 or response.content_type != "text/html":
            pages[name] = {**_record(response), "skipped": "unexpected status or content type"}
            continue
        doc = readable(_decode(response))
        files[f"snapshots/{name}.html"] = response.body
        files[f"snapshots/{name}.txt"] = doc.text.encode("utf-8")
        pages[name] = {
            **_record(response),
            "html_file": f"snapshots/{name}.html",
            "text_file": f"snapshots/{name}.txt",
            "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
            "text_version": TEXT_VERSION,
            "rights_evidence": "access/checks.json#rights",
            "access_evidence": "access/checks.json#robots",
        }
    for relative, data in files.items():
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
    summary = {"checks": checks, "pages": pages}
    with (output_dir / "access" / "checks.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model_lab/runs/extraction-pilot-v1"))
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps({
        "rights_ok": True,
        "robots_can_fetch": summary["checks"]["robots"]["can_fetch"],
        "pages": {k: {f: v.get(f) for f in ("status", "bytes", "sha256", "skipped")}
                  for k, v in summary["pages"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
