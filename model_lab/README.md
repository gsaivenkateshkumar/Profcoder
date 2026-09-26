# Profcoder original model lab

This package is a separate, CPU-first experiment. It does not replace the
working Groq-backed application, download models, or send training samples
online. The two newly written demonstration samples in `data/samples/` are
dedicated to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
They are far too small to teach a useful coding assistant.

## First milestone: deterministic, reviewed data and a tokenizer

From the repository root, using its Python environment:

```powershell
& .\.venv\Scripts\python.exe -m model_lab.prepare
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_lab.py -v
```

The first command reads only files explicitly listed in
`model_lab/data/manifest.json`. Every sample needs a manually reviewed source,
license identifier, explicit rights review, and matching SHA-256. The program
computes each SHA-256 over UTF-8 content with CRLF normalized to LF, so a Git
checkout on Windows produces the same token data and metadata as on Linux.
It rejects duplicate samples, symlinks/junctions, paths outside `samples/`, files
above 1 MiB, and corpora above 16 MiB. It writes `train.u16le`,
`validation.u16le`, and `metadata.json` to the ignored `model_lab/runs/demo-v1`
directory. Choose another `--output` directory for a repeat run; existing
output is never overwritten. No credentials or existing project files are
automatically ingested. A manifest flag records a human review; it cannot prove
that a license grant is valid. Review external code and preserve required
notices before adding any larger corpus.

The initial tokenizer maps each UTF-8 byte to IDs 0–255 and reserves 256 for
BOS, 257 for EOS, and 258 for padding. Its 259 IDs round-trip text without
learned vocabulary or internet access. A learned tokenizer can be compared
later, after the data pipeline and CPU training measurements work.

## Project-split manifests (version 2)

A `"version": 2` manifest adds a `project` field to every sample so that
held-out evaluation measures unseen projects rather than unseen files from
projects the model trained on. The demo `manifest.json` stays at version 1 and
produces the same token files and metadata as before.

```json
{
  "version": 2,
  "samples": [
    {
      "path": "samples/calc-tools/add.txt",
      "project": "calc-tools",
      "split": "train",
      "sha256": "<SHA-256 of the LF-normalized UTF-8 content>",
      "license": "CC0-1.0",
      "source": "Where this exact file came from, as reviewed",
      "rights_reviewed": true
    }
  ]
}
```

Project IDs are 1–64 lowercase letters, digits, or hyphens. Each file must be
at `samples/<project>/<name>.txt` for its own project; nested directories,
traversal, other projects' directories, and non-`.txt` files are rejected. Every
project must belong entirely to `train` or entirely to `validation`, and any
exact duplicate content (after CRLF→LF normalization) is rejected anywhere in
the corpus, including across splits. All version-1 checks still apply: explicit
file list, reviewed source/license/rights fields, SHA-256 match,
symlink/junction rejection, 1 MiB per file and 16 MiB total, and no overwriting
of output. Version-2 `metadata.json` also records which projects are in each
split. Duplicate detection is exact-match only; near-duplicates, forks, and
vendored copies across projects need human review.

```powershell
& .\.venv\Scripts\python.exe -m model_lab.prepare --manifest path\to\manifest.json --output model_lab/runs/projects-v1
```

The version-2 tests build small, original synthetic projects in temporary
directories. That synthetic data checks the pipeline only; it cannot
establish coding quality.

## Source-provenance manifests (version 3) and local owned corpora

Version 3 records provenance and rights once per project and the original
path once per file, so a manifest can describe code whose owner allowed local
use without granting any public license:

```json
{
  "version": 3,
  "projects": {
    "example-app": {
      "split": "train",
      "repository": "https://github.com/owner/example-app",
      "commit": "<full 40-character commit hash>",
      "rights_basis": "owner-authorized-local-use",
      "license": null,
      "authorized_by": "Who authorized which use, and when",
      "rights_reviewed": true
    }
  },
  "samples": [
    {
      "path": "samples/example-app/src__app.py.txt",
      "project": "example-app",
      "source_path": "src/app.py",
      "sha256": "<SHA-256 of the LF-normalized copy>",
      "transform": "verbatim"
    }
  ]
}
```

`rights_basis` is either `public-license`, which requires an SPDX-style
`license` and `authorized_by: null`, or `owner-authorized-local-use`, which
requires `license: null` and a non-empty `authorized_by` note. An owner's
permission is therefore never recorded as a license. Samples take their split from their
project. A repository may belong to only one project, so it cannot be split
across train and validation under two names. Each source path must be relative
and unique, and each copy is marked `verbatim` or `redacted`. The version-1/2
path, hash, symlink, duplicate-content, and size checks still apply. Metadata
from a version-3 manifest lists each project's repository, commit, and rights,
and sets `local_only: true` when any project is owner-authorized only.

`model_lab.build_corpus` creates such a corpus from local clones and a spec
that lists every file explicitly. It reads each file from Git's object store
at the recorded commit, so uncommitted edits cannot slip in, and rejects Git
LFS pointers, non-UTF-8 or binary files, and oversized files. A reviewed redacted copy
may replace a file; it is recorded as `redacted` and must differ from the
original. It never overwrites output. Keep the spec, copies, manifest, and
prepared tokens under the ignored `model_lab/data/local/` and
`model_lab/runs/` directories. Owner-authorized code must not be committed or
published.

```powershell
& .\.venv\Scripts\python.exe -m model_lab.build_corpus --spec model_lab/data/local/owned-v1-spec.json --output model_lab/data/local/owned-v1 --git "C:\path\to\git.exe"
& .\.venv\Scripts\python.exe -m model_lab.prepare --manifest model_lab/data/local/owned-v1/manifest.json --output model_lab/runs/owned-v1-data
```

### Pre-training audit

Run `model_lab.audit` after building a corpus and before preparing or training
on it:

```powershell
& .\.venv\Scripts\python.exe -m model_lab.audit --manifest model_lab/data/local/owned-v1/manifest.json --report model_lab/runs/owned-v1-audit.json
```

It reads only the files the manifest lists. There is no directory scanning,
and unlisted files next to the samples are ignored. Every file goes through the same checks as
preparation: path containment, symlink/junction and hard-link refusal, the
1 MiB per-file and 16 MiB total limits, strict UTF-8, and the SHA-256 match.
Exact duplicates stay preparation errors. The report
lists only file paths, finding categories, and counts. It never includes
matched values, line numbers, or surrounding text, and refusal messages come
from fixed validation strings. Exit status is 0 when nothing is flagged, 1 when
review is required, and 2 when the manifest or a file is refused. `--report`
never overwrites. Keep reports under ignored `model_lab/runs/`.

- `credential:*`: private-key blocks, known token formats, hard-coded values
  assigned to credential-named identifiers (`password`, `api_key`,
  `*_token`, …), URLs with embedded passwords, and literal comparisons against
  values read from a password/secret/code prompt or `getpass`. Empty,
  `<placeholder>`, `your_…`, template, and prompt-like strings are skipped.
- `personal:*`: email addresses outside reserved example domains, phone
  numbers, public IPv4 addresses, home-directory paths with a user name, and
  Luhn-valid card numbers or national-ID-shaped numbers in string literals.
- `review:*`: high-entropy literals and lines over 10,000 characters, which
  are counted but not scanned.
- Near-duplicates: files from *different* projects that share many sampled
  8-token shingles (estimated Jaccard ≥ 0.5 or containment ≥ 0.7) are listed
  as `needs-manual-review`, marked if they cross train/validation. The audit
  never declares a pair safe or unsafe. Shingles common to more than 25 files
  are ignored as boilerplate, and renamed or reformatted copies can evade this
  check.

**A clean audit is not proof of safety.** Patterns miss credentials with
unusual names or formats, obfuscated or split values, and most personal data;
person names in particular are not detected. They also flag some harmless
code. Manual review is still required before training on or sharing any
corpus. On the current local owned corpus the audit reports no findings and no
cross-project near-duplicate pairs, after credentials and names had already
been redacted by hand.

The spec has the same project fields plus `clone` (local clone path), `files`
(explicit source paths), and `redacted` (source path → reviewed replacement
file). Builder tests need Git; set `PROFCODER_GIT` if `git` is not on `PATH`,
otherwise they are skipped.

First owned-corpus run (`owned-v1`, local only, 2026-09-27): 133 Python files
(1,379,780 tokens) from one project for training, and 1 Python file (32,573
tokens) from a different project for validation. Under the default 100-step
budget (51,200 training-token positions, about 3.7% of one pass over the
training split), the run took 6.7 s at about 11,200 training tokens/second,
peaked at 367 MB RAM, and ended with train loss 2.77. Validation loss over all
32,572 validation targets (128 windows of 256 tokens) was 2.74. A uniform guess
over 259 byte IDs scores ln 259 ≈ 5.56. The two projects differ in domain, age,
and style, and validation is a single file, so this loss mostly reflects
byte-level Python and English statistics. It does not measure code
correctness, generalization, or coding ability. The validation set is a
single file from one small, older project. One file's style dominates the
score, it cannot show variance across projects, and it is too small to compare
closely matched runs. Add several independent held-out projects before
treating validation loss as a model-selection signal.

## CPU training milestone

Install the CPU-only PyTorch build into a separate training environment; use
the official [PyTorch CPU installation selector](https://pytorch.org/get-started/locally/)
if the command changes. The application environment does not need PyTorch.
From the repository root on Windows, after data preparation:

```powershell
& .\.venv\Scripts\python.exe -m venv F:\profcoder-model-venv
& F:\profcoder-model-venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& F:\profcoder-model-venv\Scripts\python.exe -m unittest discover -s tests -p test_model_training.py -v
& F:\profcoder-model-venv\Scripts\python.exe -m model_lab.train --run model_lab/runs/smoke-v1 --max-steps 2 --batch-size 2 --seq-length 32 --max-seconds 120
```

If the smoke run passes, start the bounded full experiment:

```powershell
& F:\profcoder-model-venv\Scripts\python.exe -m model_lab.train --run model_lab/runs/train-v1
```

`--resume --run model_lab/runs/train-v1` continues a saved run up to the
100-step total. Reuse the original batch size and sequence length on resume;
the program checks those settings, model configuration, and a fingerprint of
the prepared corpus. Choose a new `--run` directory to begin again. Runs are
ignored by Git, and the training environment lives outside the repository.
`checkpoint.pt` holds model,
optimizer, and CPU random-number states; it is saved atomically at validation
intervals and at the end. Only load your own run files. The loader uses
`weights_only=True`.

`summary.json` reports completed steps, training tokens/second measured during
steps, invocation wall time, process CPU time and average busy CPU cores, peak
process resident memory, checkpoint size, and train/validation loss. Training
throughput excludes evaluation and checkpoint writing. Validation uses up to
four fixed windows from the distinct demo validation file; it does not measure
general coding ability. The training data are too short for 128-token validation
windows, so the default training sequence is **64** tokens. Sampled training
windows may cross document boundaries in this toy milestone.

First measured `train-v1` run (i5-10400F, 6 threads, 2026-09-27): 100 steps,
51,200 training tokens, 6.2 s invocation wall time, about 11,500 training
tokens/second, 347 MB peak process RAM, 22.6 MB checkpoint, final train loss
0.52 and validation loss 3.49. The gap shows memorization of the tiny demo
corpus; these are toy-corpus measurements, not evidence of coding ability.

## Offline generation milestone

`model_lab.generate` loads a checkpoint written by `model_lab.train` (with
`weights_only=True`), rebuilds the model from its saved configuration, and
continues a prompt on the CPU. It is not connected to the Groq application or
the API.

```powershell
& F:\profcoder-model-venv\Scripts\python.exe -m model_lab.generate --checkpoint model_lab/runs/train-v1/checkpoint.pt --prompt "def add(a, b):" --max-new-tokens 64
& F:\profcoder-model-venv\Scripts\python.exe -m unittest discover -s tests -p test_model_generation.py -v
```

Prompts are limited to 4,096 UTF-8 bytes and output to 1–512 new tokens; only
the most recent context-length tokens are fed to the model. Decoding is greedy
unless `--temperature` (up to 2.0) is set, optionally with `--top-k` and
`--seed`. Generation stops at EOS, which is not included in the output; BOS and
padding are never produced. Because training documents are raw bytes followed by
EOS, the prompt is not prefixed with BOS. Invalid UTF-8 byte sequences are
shown as U+FFFD. Expect repetitive, fragmentary text from the demo model.

## Read-only held-out evaluation

`model_lab.evaluate` scores a local checkpoint on the `validation` split of a
prepared corpus. It loads with `weights_only=True`, runs under `torch.no_grad()`
in eval mode, has no optimizer, writes no checkpoint, and does not import or
call the Groq application.

```powershell
& F:\profcoder-model-venv\Scripts\python.exe -m model_lab.evaluate --checkpoint model_lab/runs/train-v1/checkpoint.pt --data model_lab/runs/demo-v1
& F:\profcoder-model-venv\Scripts\python.exe -m unittest discover -s tests -p test_model_evaluation.py -v
```

Windows are consecutive and non-overlapping from token 0. Window *i* feeds
tokens `[start, end)` and scores next-token targets `[start+1, end+1)`; the
last window may be shorter, so each target is scored at most once. The window
length defaults to the checkpoint's context length (`--seq-length` overrides
it), and at most `--max-windows` (default 64, maximum 1,024) are scored. The JSON
report lists every window's token ranges and loss, the token-weighted mean loss,
how many target tokens were scored, and whether the whole split was covered.
It also gives the checkpoint's SHA-256 and step, the corpus fingerprint, and
whether that fingerprint matches the corpus the checkpoint was trained on.
`--report PATH` also writes the JSON but refuses to overwrite an existing file.
Windows may cross document (EOS) boundaries. On the demo corpus, `train-v1`
scores all 98 validation targets in one 256-token window with a loss of 3.488.
This number only compares runs on this toy data; it says nothing about coding
quality.

## Training target and budget

- First **randomly initialized** decoder-only Transformer: 4 layers, width
  192, 4 attention heads, feed-forward width 768, context 256 tokens; tie
  input/output embeddings. This is roughly **1.9 million parameters**, using
  the 259-token byte vocabulary. Keep all computation on the CPU.
- The weights, gradients and two FP32 Adam moments alone would occupy roughly
  **30 MB** (about 16 bytes per parameter); activations, temporary arrays,
  Python, the optimizer, and Windows require more. The measured `train-v1`
  peak was 347 MB of process RAM.
- First training run: cap it at 100 steps with batch 8 and sequence length 64
  (at most 51,200 training-token positions) and a 30-minute soft cutoff. Record
  tokens per second, peak process memory, CPU usage, checkpoint size, train and
  held-out validation loss, and exact wall time before increasing the budget.
  Measured on the i5-10400F with 6 threads: about 11,500 training tokens/second
  and 6.2 s for the whole 100-step invocation (see `train-v1` above). This
  throughput is specific to this 1.9M-parameter model and 64-token sequences;
  re-measure after changing model size, sequence length, or thread count. A step
  or checkpoint may slightly exceed the time limit because they finish before
  checking the clock again.
- Demo success: validate all samples; reproduce byte-for-byte token files; run
  a finite CPU training step, save/reload a checkpoint with optimizer and RNG
  state, and compute finite validation loss from the toy held-out file.
  Fluent code generation is **not** an expected result from this demo corpus.

## Optional document extraction (pilot run)

The [two-page SQLite pilot](EXTRACTION_PILOT.md) compared deterministic HTML
extraction (`model_lab/extraction/parse.py`) with ScrapeGraphAI 2.3.0 and a
local CPU-only `qwen2.5:1.5b`, both on the same saved, permitted snapshots.
Against a provisional, not yet human-verified answer key, direct parsing
matched 15/15 items with exact, traceable spans in under 0.2 s per page.
ScrapeGraphAI matched 7/15, produced 22 unattributed or altered items, and took
up to 333 s and 2.1 GB of Ollama memory per page, with no provider charge. It
is **not** recommended for the ingestion path. ScrapeGraphAI extracts with an
existing LLM and does not train the Profcoder model. The pilot pages are in no
training or validation split, and the owned ASTRA/company corpus, split, and
checkpoints are unchanged.

## Next milestones

1. Review the local owner-authorized corpus and add independently reviewed
   held-out projects. One company source file is a limited validation set.
2. Review the provisional extraction answer key. Keep deterministic parsing
   for structured documentation. Revisit LLM extraction only for sources that
   parsing cannot handle, and only with explicit approval for any cloud model.
3. Compare offline CPU generation latency/quality with a separate, existing
   open-weight model in Profcoder. Consider quantization only after the
   measured baseline has been reviewed. Consider sparse experts and disk
   streaming only if actual measurements show a benefit.

Proposed repository layout: `model_lab/` holds this experimental pipeline and
ignored run artifacts; `app/` retains the existing agent/API; `tests/` verifies
both independently. Checkpoints and licensed external datasets stay outside
Git until their provenance, size and publication terms have been reviewed.
