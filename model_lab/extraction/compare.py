"""Score extraction outputs against a provisional answer key and the saved snapshots.

Every key item must occur verbatim in its page's readable-text snapshot; its
offsets are recorded as evidence. Every output item is checked the same way,
so an item that is not a verbatim span is reported as unsupported
(unattributed), whichever method produced it. Nothing here is human-verified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from model_lab.extraction.parse import CHROME, load_snapshot
from model_lab.extraction.text import readable

FIELD_KIND = {"title": "title", "heading": "heading", "summary": "paragraph", "sql": "code"}
SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def heading_key(text: str) -> str:
    return SECTION_NUMBER.sub("", norm(text)).rstrip(".").casefold()


def output_items(fields: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten a method's fields into (field, text) pairs; tolerate missing fields."""
    def text(value):
        if isinstance(value, dict):
            value = value.get("value")
        return value if isinstance(value, str) and value.strip() else None

    items = []
    for field, plural in (("title", False), ("summary", False), ("heading", True), ("sql", True)):
        raw = fields.get("headings" if field == "heading" else field)
        values = raw if plural and isinstance(raw, list) else [raw]
        items += [(field, t) for t in map(text, values) if t is not None]
    return items


def locate_key(doc, items: list[dict[str, str]]) -> list[dict[str, object]]:
    located = []
    for item in items:
        value = item["value"]
        spans = [(b.start, b.end) for b in doc.blocks
                 if b.kind == FIELD_KIND[item["field"]] and value in doc.block_text(b) and not b.in_nav]
        start = doc.text.find(value, spans[0][0]) if spans else doc.text.find(value)
        if start < 0:
            raise ValueError(f"answer-key item is not in the snapshot: {item['field']}")
        located.append({**item, "evidence": [start, start + len(value)]})
    return located


def score_page(
    doc, key: list[dict[str, object]], fields: dict[str, object], svg_text: str = "",
) -> dict[str, object]:
    items = output_items(fields)
    text = norm(doc.text)
    supported = [(f, t) for f, t in items if norm(t) in text]
    unsupported = [(f, t) for f, t in items if norm(t) not in text]
    # Fairness check: some extractors also read labels drawn inside SVG diagrams.
    with_svg = norm(svg_text) if svg_text else text
    unsupported_with_svg = [(f, t) for f, t in unsupported if norm(t) not in with_svg]
    distinct_sql = {norm(t) for f, t in items if f == "sql"}
    chrome = [(f, t) for f, t in items if CHROME.match(norm(t))]
    outputs = {"title": [], "summary": [], "heading": [], "sql": []}
    for field, value in items:
        outputs[field].append(value)
    matched = []
    for item in key:
        field, value = item["field"], item["value"]
        if field == "title":
            hit = any(norm(v).casefold() == value.casefold() for v in outputs["title"])
        elif field == "summary":
            hit = any(norm(value) in norm(v) and norm(v) in text for v in outputs["summary"])
        elif field == "heading":
            hit = any(heading_key(v) == heading_key(value) for v in outputs["heading"])
        else:
            hit = any(norm(v) == norm(value) for v in outputs["sql"])
        matched.append(hit)
    key_sql = [i["value"] for i in key if i["field"] == "sql"]
    return {
        "key_items": len(key),
        "key_items_matched": sum(matched),
        "key_items_missed": [f"{i['field']}#{n}" for n, (i, hit) in enumerate(zip(key, matched)) if not hit],
        "output_items": len(items),
        "supported_verbatim": len(supported),
        "unsupported_or_unattributed": len(unsupported),
        "unsupported_even_with_svg_text": len(unsupported_with_svg),
        "unsupported_items": [{"field": f, "value": t} for f, t in unsupported],
        "site_chrome_items": len(chrome),
        "sql_key_exact": sum(any(v == s for v in outputs["sql"]) for s in key_sql),
        "sql_key_whitespace_normalized": sum(any(norm(v) == norm(s) for v in outputs["sql"]) for s in key_sql),
        "sql_key_total": len(key_sql),
        "sql_output_items": len(outputs["sql"]),
        "sql_output_distinct": len(distinct_sql),
        "sql_output_verbatim": sum(norm(v) in text for v in outputs["sql"]),
    }


def compare(pilot_dir: Path, key_path: Path, methods: dict[str, Path]) -> dict[str, object]:
    pilot_dir = Path(pilot_dir)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "answer_key_status": key["status"],
        "human_verified": False,
        "answer_key": {},
        "methods": {},
    }
    docs, svg_texts = {}, {}
    for page, items in key["pages"].items():
        html, _ = load_snapshot(pilot_dir, page)
        docs[page] = readable(html)
        svg_texts[page] = readable(html, include_svg_text=True).text
        report["answer_key"][page] = locate_key(docs[page], items)
    for method, directory in methods.items():
        pages = {}
        for page, doc in docs.items():
            path = Path(directory) / f"{page}.json"
            if not path.is_file():
                pages[page] = {"status": "not run"}
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            pages[page] = {
                **score_page(doc, report["answer_key"][page], result.get("fields") or {},
                             svg_texts[page]),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "peak_process_rss_bytes": result.get("peak_process_rss_bytes"),
                "llm": result.get("llm"),
                "error": result.get("error"),
            }
        report["methods"][method] = pages
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("model_lab/runs/extraction-pilot-v1"))
    parser.add_argument("--key", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--method", action="append", default=[],
                        help="name=directory of <page>.json outputs; repeatable")
    args = parser.parse_args()
    methods = dict(entry.split("=", 1) for entry in args.method) or {
        "deterministic": args.pilot / "outputs/deterministic",
        "scrapegraphai": args.pilot / "outputs/scrapegraphai",
    }
    report = compare(args.pilot, args.key or args.pilot / "answer_key.json", methods)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    target = args.report or args.pilot / "comparison.json"
    with Path(target).open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(json.dumps({m: {p: {k: v for k, v in r.items() if k not in ("unsupported_items", "llm")}
                          for p, r in pages.items()} for m, pages in report["methods"].items()}, indent=1))


if __name__ == "__main__":
    main()
