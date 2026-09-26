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
rejects duplicate samples, symlinks/junctions, paths outside `samples/`, files
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

## Training target and budget (estimates, not measured performance)

- First **randomly initialized** decoder-only Transformer: 4 layers, width
  192, 4 attention heads, feed-forward width 768, context 256 tokens; tie
  input/output embeddings. This is roughly **1.9 million parameters**, using
  the 259-token byte vocabulary. Keep all computation on the CPU.
- The weights, gradients and two FP32 Adam moments alone would occupy roughly
  **30 MB** (about 16 bytes per parameter); activations, temporary arrays,
  Python, the optimizer, and Windows require more. This is a sizing estimate,
  not a measured peak-RAM figure.
- First training run: cap it at 100 steps with batch 8 and context 128 (at most
  102,400 training-token positions) and a 30-minute wall-clock cutoff. Record
  tokens per second, peak process memory, CPU usage, checkpoint size, train and
  held-out validation loss, and exact wall time before increasing the budget.
  Actual throughput and training time on the i5-10400F are **unknown** until
  measured on that computer.
- Demo success: validate all samples; reproduce byte-for-byte token files; then
  (in the next milestone) run a finite CPU training step, save/reload a
  checkpoint with optimizer and RNG state, and compute finite held-out loss.
  Fluent code generation is **not** an expected result from this demo corpus.

## Next milestones

1. Install a CPU build of PyTorch in a **separate training environment** after
   selecting the current Windows/CPU installer at
   [PyTorch Get Started](https://pytorch.org/get-started/locally/). Do not add
   it to the API's small `requirements.txt` or use the GT 710 for this baseline.
2. Implement the model, finite-step CPU trainer, resumable checkpoints and
   held-out loss; benchmark the server before choosing a longer run. Expand to
   carefully reviewed, deduplicated original/permissively licensed code,
   splitting by source project so evaluation is genuinely held out.
3. Add offline CPU generation and compare latency/quality with a separate,
   existing open-weight model in Profcoder. Consider quantization after the
   baseline works. Consider sparse experts and disk streaming only if actual
   measurements show a benefit.

Proposed repository layout: `model_lab/` holds this experimental pipeline and
ignored run artifacts; `app/` retains the existing agent/API; `tests/` verifies
both independently. Checkpoints and licensed external datasets stay outside
Git until their provenance, size and publication terms have been reviewed.
