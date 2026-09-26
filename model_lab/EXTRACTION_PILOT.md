# Optional extraction pilot (run 2026-09-27; results below)

ScrapeGraphAI is an optional way to **extract** information from approved
documents. It uses an existing LLM; it does not train Profcoder's model. Keep
it out of the API environment and the existing training/evaluation pipeline.
No package, web page, extraction model, or training sample is added by this plan.

## Smallest useful source set

Use exactly these two SQLite documentation pages as a first, read-only pilot:

| Page | URL | Content group |
| --- | --- | --- |
| SELECT | https://sqlite.org/lang_select.html | SQLite documentation |
| INSERT | https://sqlite.org/lang_insert.html | SQLite documentation |

SQLite [states that its documentation is public domain](https://sqlite.org/copyright.html).
Its [robots.txt](https://sqlite.org/robots.txt), checked on 2026-09-27,
disallows `/cvstrac`, `/src`, `/docsrc`, `/cgi` and `/contrib`; neither selected
`/lang_*.html` path is on that list. **Recheck** both rights and access rules
immediately before retrieving any page. If terms, robots rules, response
status, or content type disallow this use, skip the page. Make no broad crawl,
no authenticated request, and no fetch of linked pages or assets. Fetch each
approved page once, with a descriptive user agent and a conservative rate.
The library's MIT license covers ScrapeGraphAI itself, not the page content.
Public-domain SQLite documentation is not CC0 and must not be labeled CC0.

Treat the two pages as **one source project**. Neither page is part of the
existing ASTRA training or company-management-system validation corpus. Keep
both outside those splits during the pilot. If later admitted to training,
assign the whole SQLite source project to one split only; do not train on one
page and validate on the other. Version-3 manifests do not currently represent
public-domain-without-license, so extend the rights schema and tests before
admitting these pages. Never invent a license identifier.

## Controlled comparison

1. Retrieve the two allowed pages to an ignored local pilot directory with a
   fixed size limit, timeout, and no additional crawling. Store the raw HTML,
   SHA-256, retrieval UTC time, source URL, HTTP status and content type,
   rights/access-rule URLs and check date, and a readable text snapshot.
   Preserve these originals alongside all derived records. Do not commit them.
2. Write a small, manually reviewed answer key for the same task on each page:
   extract section headings, a short explanation, and any SQL syntax/example
   grounded by an exact quote or source offset. Around five reviewed items per
   page are enough for this feasibility check. Exclude navigation and site
   chrome; do not invent example SQL that the page does not contain.
3. Method A: use a deterministic HTML/text parser on **the saved snapshots**.
   Method B: run ScrapeGraphAI's single-page extractor on **those same local
   snapshots**, with the same output fields. Confirm its local-file API in the
   pinned release before coding; prevent an extra network fetch. Isolate its
   dependencies in a separate environment. A local Ollama LLM is an optional
   benchmark if it fits the CPU/RAM budget; a cloud LLM requires an explicit
   opt-in before page content is sent or billed. Do not assume the original
   1.9M-parameter Profcoder model can act as an extraction LLM.
4. Audit both outputs for secrets/personal information, manually verify each
   item against saved text, and remove duplicates. Measure per-page elapsed
   time, peak RAM if available, precision/recall against the answer key,
   unsupported items, SQL code preserved exactly, and attributable evidence.
   Record model ID, prompt/completion tokens and actual provider charges when
   available. If usage or billing is unknown, mark cost **unknown**, not zero;
   for a local LLM report CPU time and memory instead of an API bill. Direct
   parsing has no LLM usage.

Only consider adding ScrapeGraphAI to the regular ingestion path if its
verified extractions justify its measured time, resource use, and provider
cost. Reject any unsupported or unattributed training example. These pages
remain pilot material until explicitly reviewed, deduplicated, and assigned a
legal rights basis and project-level split. The finished assistant's offline
operation does not depend on fetching these pages or calling an extraction LLM.

## Tooling

All pilot files live in the ignored `model_lab/runs/extraction-pilot-v1/`
directory: snapshots, outputs, the answer key, and reports. None of them are committed.

```powershell
# 1. Recheck rights + robots.txt, then save each allowlisted page once.
& .\.venv\Scripts\python.exe -m model_lab.extraction.fetch
# 2. Method A: deterministic extraction from the saved snapshots.
& .\.venv\Scripts\python.exe -m model_lab.extraction.parse
# 3. Method B (separate environment, local Ollama on 127.0.0.1, CPU only).
$env:TIKTOKEN_CACHE_DIR = "F:\profcoder-scrapegraph-venv\tiktoken-cache"
& F:\profcoder-scrapegraph-venv\Scripts\python.exe -m model_lab.extraction.scrapegraph_run --page insert
# 4. Score both against the provisional answer key.
& .\.venv\Scripts\python.exe -m model_lab.extraction.compare
```

- `extraction.fetch` can request only the rights page, robots.txt, and the two
  pages, over HTTPS to `sqlite.org`. It refuses redirects, caps each body at
  2 MiB, uses a 20 s timeout, waits at least 5 s between requests, and sends a
  descriptive user agent. It fetches no links or assets. If the rights page
  no longer states the documentation is public domain, or robots.txt is
  unavailable, it stops before any page is fetched and writes nothing. A page
  that robots.txt disallows or that returns a non-HTML/non-200 response is
  skipped. It records each page's URL, UTC retrieval time, HTTP status, content
  type, size, and SHA-256, plus the rights and robots evidence. It saves the raw
  HTML and a readable-text snapshot, and never overwrites.
- `extraction.parse` rechecks each snapshot's SHA-256 and extracts `title`,
  `headings`, `summary`, and `sql`. Every value is an exact span of the text
  snapshot and carries its character offsets. Navigation and diagram toggles
  are excluded.
- `extraction.scrapegraph_run` runs only in its own venv, pinned to
  `scrapegraphai==2.3.0`. That release's `SmartScraperGraph` treats a
  `source` not starting with `http` as local HTML (`input_key="local_dir"`,
  no fetch), and the runner asserts this. **ScrapeGraphAI telemetry is on by
  default and its payload can include the prompt, page content, and answer.**
  The runner disables it and installs a socket guard that refuses every
  non-loopback connection. It forces CPU-only inference (`num_gpu: 0`), caps
  output at 1,024 tokens per call, and stops a page after 900 s. tiktoken's
  vocabulary must be cached beforehand (`TIKTOKEN_CACHE_DIR`), because the
  guard correctly blocked its first-run download.
- `extraction.compare` requires every answer-key item to occur verbatim in
  the snapshot and records its offsets. It checks every output item from every
  method the same way, and reports key matches, unsupported (unattributed)
  items, SQL exact and whitespace-normalized fidelity, distinct vs repeated
  SQL items, and site-chrome leaks. A fairness check also counts text drawn
  inside inline SVG syntax diagrams as legitimate source text.

## Pilot results (2026-09-27, i5-10400F, 16 GB RAM)

The rights page still stated that the documentation is public domain, and
robots.txt allowed both pages (no crawl delay). SELECT was saved at
2,033,807 bytes (SHA-256 `a6d38ca1…c140`, mostly inline SVG diagrams) and
INSERT at 361,906 bytes (`03d0cdd8…509a`).

The answer key has 15 items (9 for SELECT, 6 for INSERT), each pinned to
exact snapshot offsets. It is **provisional**: the assistant wrote it before
running Method B, and the same assistant wrote Method A. Semantic accuracy is
not human-verified until the owner reviews it.

| Measure (SELECT / INSERT) | A: deterministic parser | B: ScrapeGraphAI 2.3.0 + qwen2.5:1.5b (Q4_K_M, CPU) |
| --- | --- | --- |
| Key items matched | 9/9 · 6/6 | 3/9 · 4/6 |
| Output items | 24 · 6 | 23 · 11 |
| Unsupported / unattributed | 0 · 0 | 15 · 7 (13 · 7 even counting SVG diagram text) |
| Key SQL exact | 2/2 · 3/3 | 0/2 · 1/3 |
| SQL items: total, distinct, verbatim | 3, 3, 3 · 3, 3, 3 | 15, 7, 0 · 3, 3, 1 |
| Elapsed per page | 0.18 s · 0.04 s | 333 s · 7.8 s (model already loaded) |
| Peak RAM | 22 MB process | 209 MB Python + 2.13 GB Ollama · 205 MB + 1.71 GB |
| LLM usage | none | 4 calls, 17,848 prompt + 2,365 completion tokens · 1 call, 2,987 + 115 |
| CPU time | < 1 s | 1,984 s Ollama CPU · 41 s |
| Provider cost | none | none (local model, no API account); electricity not measured |

Method B's SELECT "SQL" consisted of flattened syntax-diagram labels rather
than the page's SQL examples. They repeated with small drift (hyphens became
underscores), and the last one was cut off at the output cap. It returned 6 of
19 headings and a title the page does not contain. On INSERT it changed
whitespace and punctuation in two of the three syntax forms and invented
section headings.

Earlier attempts are kept locally as failures, not hidden. In the first, the
guard blocked tiktoken's download and nothing ran. In the second, Ollama
silently offloaded 12 of 29 layers to the GT 710, which broke the CPU-only
budget: 245 s for INSERT. The third was a valid but superseded uncapped
INSERT run. In the fourth, one uncapped SELECT call generated more than
12,700 tokens over about 19 minutes without finishing and was stopped.

**Recommendation: do not add ScrapeGraphAI to the ingestion path.** On these
pages it was slower by three to four orders of magnitude, used about 100×
more memory, needed telemetry and network safeguards, and produced mostly
unattributable or altered text. Direct parsing matched every provisional key
item with exact, traceable spans. A larger local or cloud model might do
better, but a cloud LLM needs explicit approval and the owner's review of
cost. Nothing measured here suggests an LLM extractor is needed for
well-structured documentation. SQLite pages remain pilot material only; they
are not in any training or validation split.
