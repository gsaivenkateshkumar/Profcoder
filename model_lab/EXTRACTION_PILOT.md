# Optional extraction pilot (proposal; not yet run)

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
