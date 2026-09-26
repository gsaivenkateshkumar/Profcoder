"""Method A: deterministic extraction from saved pilot snapshots, with text offsets.

Fields shared with every method: title, headings, summary, sql. Each value is
an exact span of the saved readable-text snapshot, so it is traceable to the
original HTML, whose SHA-256 is rechecked against the fetch record first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from model_lab.extraction.text import Readable, readable
from model_lab.resources import peak_rss_bytes

METHOD = "deterministic-html-v1"
# Diagram show/hide toggles and "last updated" footers are paragraph-level chrome;
# menus, taglines, and tables of contents sit in navigation containers or divs.
CHROME = re.compile(r"^(?:[\w-]+: (?:hide|show)|This page was last updated on .*)$")
SQL_STATEMENT = re.compile(
    r"^(?:SELECT|INSERT|REPLACE|UPDATE|DELETE|WITH|VALUES|CREATE)\b[^\n]{0,200};$"
)


def load_snapshot(pilot_dir: Path, page: str) -> tuple[str, dict[str, object]]:
    """Return saved HTML after checking it still matches the recorded SHA-256."""
    pilot_dir = Path(pilot_dir)
    record = json.loads((pilot_dir / "access/checks.json").read_text(encoding="utf-8"))["pages"][page]
    if "html_file" not in record:
        raise ValueError(f"{page} was not saved: {record.get('skipped')}")
    raw = (pilot_dir / record["html_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"{page} snapshot no longer matches its recorded SHA-256")
    return raw.decode("utf-8"), record


def _item(doc: Readable, block) -> dict[str, object]:
    return {"value": doc.block_text(block), "evidence": [block.start, block.end]}


def extract(doc: Readable) -> dict[str, object]:
    content = [b for b in doc.blocks if not b.in_nav and not CHROME.match(doc.block_text(b))]
    title = next((b for b in content if b.kind == "title"), None)
    headings = [b for b in content if b.kind == "heading"]
    first_heading = headings[0].start if headings else 0
    paragraphs = [b for b in content if b.kind == "paragraph" and b.start >= first_heading]
    sql = [b for b in content if b.kind == "code"] + [
        b for b in paragraphs if SQL_STATEMENT.match(doc.block_text(b))
    ]
    summary = next((b for b in paragraphs if b not in sql), None)
    return {
        "title": _item(doc, title) if title else None,
        "headings": [_item(doc, b) for b in headings],
        "summary": _item(doc, summary) if summary else None,
        "sql": [_item(doc, b) for b in sorted(sql, key=lambda b: b.start)],
    }


def run(pilot_dir: Path, output_dir: Path) -> dict[str, object]:
    pilot_dir, output_dir = Path(pilot_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for page in ("select", "insert"):
        started = time.perf_counter()
        html, record = load_snapshot(pilot_dir, page)
        doc = readable(html)
        saved_text = (pilot_dir / record["text_file"]).read_text(encoding="utf-8")
        if doc.text != saved_text:
            raise ValueError(f"{page} text snapshot does not match its HTML")
        fields = extract(doc)
        result = {
            "method": METHOD, "page": page,
            "source": {k: record[k] for k in ("url", "sha256", "html_file", "text_file", "text_sha256")},
            "fields": fields,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "peak_process_rss_bytes": peak_rss_bytes(),
            "llm": None,
        }
        with (output_dir / f"{page}.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        results[page] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("model_lab/runs/extraction-pilot-v1"))
    args = parser.parse_args()
    results = run(args.pilot, args.pilot / "outputs" / "deterministic")
    print(json.dumps({page: {
        "headings": len(r["fields"]["headings"]), "sql": len(r["fields"]["sql"]),
        "elapsed_seconds": r["elapsed_seconds"],
    } for page, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
