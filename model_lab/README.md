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

## Training target and budget (estimates until measured on the server)

- First **randomly initialized** decoder-only Transformer: 4 layers, width
  192, 4 attention heads, feed-forward width 768, context 256 tokens; tie
  input/output embeddings. This is roughly **1.9 million parameters**, using
  the 259-token byte vocabulary. Keep all computation on the CPU.
- The weights, gradients and two FP32 Adam moments alone would occupy roughly
  **30 MB** (about 16 bytes per parameter); activations, temporary arrays,
  Python, the optimizer, and Windows require more. This is a sizing estimate,
  not a measured peak-RAM figure.
- First training run: cap it at 100 steps with batch 8 and sequence length 64
  (at most 51,200 training-token positions) and a 30-minute soft cutoff. Record
  tokens per second, peak process memory, CPU usage, checkpoint size, train and
  held-out validation loss, and exact wall time before increasing the budget.
  Actual throughput and training time on the i5-10400F are **unknown** until
  measured on that computer. A step or checkpoint may slightly exceed the time
  limit because they finish before checking the clock again.
- Demo success: validate all samples; reproduce byte-for-byte token files; run
  a finite CPU training step, save/reload a checkpoint with optimizer and RNG
  state, and compute finite validation loss from the toy held-out file.
  Fluent code generation is **not** an expected result from this demo corpus.

## Next milestones

1. Review the measured `train-v1` baseline (throughput, wall time, RAM, loss,
   checkpoint size) before increasing the budget. Do not use the GT 710 for
   this baseline.
2. Expand to
   carefully reviewed, deduplicated original/permissively licensed code,
   splitting by source project so evaluation is genuinely held out.
3. Compare offline CPU generation latency/quality with a separate, existing
   open-weight model in Profcoder. Consider quantization only after the
   measured baseline has been reviewed. Consider sparse experts and disk streaming only if actual
   measurements show a benefit.

Proposed repository layout: `model_lab/` holds this experimental pipeline and
ignored run artifacts; `app/` retains the existing agent/API; `tests/` verifies
both independently. Checkpoints and licensed external datasets stay outside
Git until their provenance, size and publication terms have been reviewed.
