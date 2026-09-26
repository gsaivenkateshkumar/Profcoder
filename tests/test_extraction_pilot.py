"""Extraction pilot tooling: access checks, text offsets, parsing, and comparison.

Uses only synthetic HTML and an in-memory fetcher; no network and no copied pages.
"""

import json
import tempfile
import unittest
from pathlib import Path

from model_lab.extraction import fetch as pilot_fetch
from model_lab.extraction.fetch import PAGES, RIGHTS_URL, ROBOTS_URL, Response, run
from model_lab.extraction.text import readable

RIGHTS_HTML = (
    "<html><body><div class='menu'><p>Home Docs</p></div>"
    "<h1>Rights</h1><p>All synthetic documentation here is in the public domain.</p>"
    "</body></html>"
)
ROBOTS_OK = "User-agent: *\nDisallow: /private\n"
PAGE_HTML = (
    "<html><head><title>Widget statement</title><style>p{}</style></head><body>"
    "<div class='menu'><a href='/'>Home</a> <a href='/x'>Search</a></div>"
    "<h1>1. The WIDGET statement</h1>"
    "<p>The WIDGET statement makes   a widget &amp; stores it.</p>"
    "<h2>2. Examples</h2>"
    "<pre>WIDGET x\n  WITH (a, b);</pre>"
    "<p>Use it <b>carefully</b>.</p><script>ignored()</script>"
    "</body></html>"
)


class FakeWeb:
    def __init__(self, robots=ROBOTS_OK, rights=RIGHTS_HTML, overrides=None):
        self.requests = []
        self.responses = {
            RIGHTS_URL: (200, "text/html", rights),
            ROBOTS_URL: (200, "text/plain", robots),
            **{url: (200, "text/html", PAGE_HTML.replace("WIDGET", name.upper()))
               for name, url in PAGES.items()},
            **(overrides or {}),
        }

    def __call__(self, url):
        pilot_fetch._check_url(url)
        self.requests.append(url)
        status, content_type, body = self.responses[url]
        return Response(url, status, content_type, "utf-8", body.encode("utf-8"),
                        "2026-09-27T00:00:00+00:00")


class ReadableTextTests(unittest.TestCase):
    def test_blocks_trace_exact_visible_text(self):
        doc = readable(PAGE_HTML)
        self.assertNotIn("ignored()", doc.text)
        self.assertNotIn("p{}", doc.text)
        by_kind = {}
        for block in doc.blocks:
            by_kind.setdefault(block.kind, []).append(doc.block_text(block))
        self.assertEqual(by_kind["heading"], ["1. The WIDGET statement", "2. Examples"])
        self.assertIn("The WIDGET statement makes a widget & stores it.", by_kind["paragraph"])
        self.assertEqual(by_kind["code"], ["WIDGET x\n  WITH (a, b);"])  # whitespace kept
        self.assertEqual(by_kind["title"], ["Widget statement"])
        svg_page = PAGE_HTML.replace("<h2>", "<svg><text>diagram-label</text></svg><h2>")
        self.assertNotIn("diagram-label", readable(svg_page).text)
        self.assertIn("diagram-label", readable(svg_page, include_svg_text=True).text)
        nav = [doc.block_text(b) for b in doc.blocks if b.in_nav]
        self.assertEqual(nav, [])  # menu links are not block kinds we record
        self.assertTrue(all(not b.in_nav for b in doc.blocks if b.kind == "heading"))


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "pilot"

    def test_saves_allowlisted_pages_once_with_evidence(self):
        web = FakeWeb()
        summary = run(self.out, web, delay=0)
        self.assertEqual(web.requests, [RIGHTS_URL, ROBOTS_URL, *PAGES.values()])
        for name in PAGES:
            record = summary["pages"][name]
            html = (self.out / record["html_file"]).read_bytes()
            self.assertEqual(record["bytes"], len(html))
            text = (self.out / record["text_file"]).read_text(encoding="utf-8")
            self.assertEqual(text, readable(html.decode()).text)
            self.assertEqual(record["rights_evidence"], "access/checks.json#rights")
        checks = json.loads((self.out / "access/checks.json").read_text(encoding="utf-8"))
        self.assertEqual(checks["checks"]["robots"]["can_fetch"], {"select": True, "insert": True})
        self.assertTrue((self.out / "access/robots.txt").is_file())
        with self.assertRaises(FileExistsError):
            run(self.out, FakeWeb(), delay=0)

    def test_rights_or_robots_failure_stops_the_source(self):
        for name, web in (
            ("rights", FakeWeb(rights="<p>All rights reserved.</p>")),
            ("robots-missing", FakeWeb(overrides={ROBOTS_URL: (404, "text/html", "")})),
            ("rights-redirect", FakeWeb(overrides={RIGHTS_URL: (301, "text/html", "")})),
        ):
            with self.subTest(name), self.assertRaises(PermissionError):
                run(self.out, web, delay=0)
            self.assertFalse(self.out.exists())
            self.assertFalse(set(web.requests) & set(PAGES.values()))

    def test_disallowed_or_bad_pages_are_skipped(self):
        robots = "User-agent: *\nDisallow: /lang_insert.html\n"
        web = FakeWeb(robots=robots, overrides={PAGES["select"]: (200, "application/pdf", "x")})
        summary = run(self.out, web, delay=0)
        self.assertEqual(summary["pages"]["insert"]["skipped"], "disallowed by robots.txt")
        self.assertNotIn(PAGES["insert"], web.requests)
        self.assertIn("skipped", summary["pages"]["select"])
        self.assertFalse((self.out / "snapshots").exists())

    def test_only_allowlisted_urls_can_be_requested(self):
        for url in ("https://sqlite.org/lang_update.html", "http://sqlite.org/lang_select.html",
                    "https://example.invalid/lang_select.html"):
            with self.subTest(url), self.assertRaises(ValueError):
                pilot_fetch.http_fetch(url)


SQL_PAGE = (
    "<html><head><title>WIDGET</title></head><body>"
    "<div class=nosearch><div class='tagline'>Tiny. Quick. Sturdy.<br>Pick all three.</div>"
    "<ul class=menu><li>Home</li><li>Docs</li></ul></div>"
    "<h1>1. Overview</h1>"
    "<p><b>widget-stmt:</b> <button>hide</button></p>"
    "<svg><text>WIDGET</text><text>widget-name</text></svg>"
    "<p>The WIDGET statement makes widgets.</p>"
    "<p>WIDGET INTO t VALUES(...);</p>"
    "<h2>1.1. Details</h2>"
    "<pre>WIDGET a,  b FROM t;</pre>"
    "<p>This page was last updated on 2026-01-01 00:00:00Z</p>"
    "</body></html>"
)


class DeterministicExtractionTests(unittest.TestCase):
    def setUp(self):
        from model_lab.extraction import parse

        self.parse = parse
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pilot = Path(self.temp.name) / "pilot"
        self.page_html = SQL_PAGE.replace("WIDGET", "SELECT")
        web = FakeWeb(overrides={
            url: (200, "text/html", SQL_PAGE.replace("WIDGET", name.upper()))
            for name, url in PAGES.items()
        })
        run(self.pilot, web, delay=0)

    def test_fields_are_exact_traceable_spans_without_chrome(self):
        results = self.parse.run(self.pilot, self.pilot / "outputs/deterministic")
        fields = results["select"]["fields"]
        text = (self.pilot / "snapshots/select.txt").read_text(encoding="utf-8")
        self.assertEqual(fields["title"]["value"], "SELECT")
        self.assertEqual([h["value"] for h in fields["headings"]], ["1. Overview", "1.1. Details"])
        self.assertEqual(fields["summary"]["value"], "The SELECT statement makes widgets.")
        self.assertEqual([s["value"] for s in fields["sql"]],
                         ["SELECT INTO t VALUES(...);", "SELECT a,  b FROM t;"])
        for item in [fields["title"], fields["summary"], *fields["headings"], *fields["sql"]]:
            start, end = item["evidence"]
            self.assertEqual(text[start:end], item["value"])
        self.assertIsNone(results["select"]["llm"])
        with self.assertRaises(FileExistsError):
            self.parse.run(self.pilot, self.pilot / "outputs/deterministic")

    def test_modified_snapshot_is_refused(self):
        html = self.pilot / "snapshots/insert.html"
        html.write_bytes(html.read_bytes() + b"<!-- edited -->")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.parse.load_snapshot(self.pilot, "insert")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        from model_lab.extraction.compare import compare

        self.compare = compare
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pilot = Path(self.temp.name) / "pilot"
        run(self.pilot, FakeWeb(overrides={
            url: (200, "text/html", SQL_PAGE.replace("WIDGET", name.upper()))
            for name, url in PAGES.items()
        }), delay=0)
        self.key = self.pilot / "key.json"
        page_key = lambda name: [
            {"field": "title", "value": name},
            {"field": "summary", "value": f"The {name} statement makes widgets."},
            {"field": "heading", "value": "1.1. Details"},
            {"field": "sql", "value": "a,  b FROM t;".join([f"{name} ", ""])},
        ]
        self.key.write_text(json.dumps({"status": "provisional-unreviewed", "pages": {
            "select": page_key("SELECT"), "insert": page_key("INSERT"),
        }}), encoding="utf-8")

    def write_method(self, name, page, fields, **extra):
        directory = self.pilot / "outputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{page}.json").write_text(json.dumps({"fields": fields, **extra}), encoding="utf-8")
        return directory

    def test_scores_support_fidelity_chrome_and_missing_runs(self):
        good = self.write_method("good", "select", {
            "title": "SELECT", "summary": "The SELECT statement makes widgets.",
            "headings": ["Details"], "sql": ["SELECT a,  b FROM t;"],
        })
        loose = self.write_method("loose", "select", {
            "title": "SELECT",
            "summary": "SELECT builds widgets quickly.",        # paraphrase: unattributed
            "headings": ["1.1. Details", "widget-stmt: hide"],  # chrome leak
            "sql": ["SELECT a, b FROM t;", "SELECT * FROM widgets;",  # spacing changed, invented
                    "SELECT widget-name", "SELECT widget-name"],      # diagram labels, repeated
        }, llm={"model": "fake"})
        report = self.compare(self.pilot, self.key, {"good": good, "loose": loose})
        self.assertFalse(report["human_verified"])
        g = report["methods"]["good"]["select"]
        self.assertEqual((g["key_items_matched"], g["unsupported_or_unattributed"]), (4, 0))
        self.assertEqual((g["sql_key_exact"], g["sql_key_whitespace_normalized"]), (1, 1))
        l = report["methods"]["loose"]["select"]
        self.assertEqual(l["key_items_matched"], 3)            # summary missed
        self.assertEqual(l["site_chrome_items"], 1)
        self.assertEqual(l["sql_key_exact"], 0)                # double space lost
        self.assertEqual(l["sql_key_whitespace_normalized"], 1)
        self.assertEqual(
            sorted(i["value"] for i in l["unsupported_items"]),
            ["SELECT * FROM widgets;", "SELECT builds widgets quickly.",
             "SELECT widget-name", "SELECT widget-name"],
        )
        # SVG labels exist in the original page, so they are not counted as invented.
        self.assertEqual(l["unsupported_even_with_svg_text"], 2)
        self.assertEqual((l["sql_output_items"], l["sql_output_distinct"]), (4, 3))
        self.assertEqual(l["llm"], {"model": "fake"})
        self.assertEqual(report["methods"]["good"]["insert"], {"status": "not run"})
        for page in ("select", "insert"):
            for item in report["answer_key"][page]:
                text = (self.pilot / f"snapshots/{page}.txt").read_text(encoding="utf-8")
                start, end = item["evidence"]
                self.assertEqual(text[start:end], item["value"])

    def test_key_items_must_exist_in_the_snapshot(self):
        data = json.loads(self.key.read_text(encoding="utf-8"))
        data["pages"]["select"][0]["value"] = "Invented title"
        self.key.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not in the snapshot"):
            self.compare(self.pilot, self.key, {})


class ScrapeGraphRunnerHelperTests(unittest.TestCase):
    def test_normalize_answer_keeps_only_shared_string_fields(self):
        from model_lab.extraction.scrapegraph_run import normalize_answer

        self.assertEqual(
            normalize_answer({"content": {"title": " T ", "headings": ["a", 3, ""],
                                          "summary": None, "sql": "SELECT 1;", "extra": "x"}}),
            {"title": "T", "headings": ["a"], "summary": None, "sql": ["SELECT 1;"]},
        )
        self.assertEqual(normalize_answer("not json"),
                         {"title": None, "headings": [], "summary": None, "sql": []})

    def test_network_guard_refuses_non_loopback_connections(self):
        import socket
        import subprocess
        import sys

        code = (
            "import socket\n"
            "from model_lab.extraction.scrapegraph_run import install_network_guard\n"
            "blocked = install_network_guard()\n"
            "s = socket.socket()\n"
            "try:\n"
            "    s.connect(('192.0.2.1', 443))\n"
            "except PermissionError:\n"
            "    print('refused', len(blocked))\n"
            "server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen(1)\n"
            "client = socket.socket(); client.connect(server.getsockname()); print('loopback ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.stdout.split("\n")[:2], ["refused 1", "loopback ok"], result.stderr)


if __name__ == "__main__":
    unittest.main()
