"""Read-only held-out loss for a local checkpoint on a prepared validation split."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model_lab.generate import load_checkpoint
from model_lab.model import ByteTransformer
from model_lab.train import load_corpus

MAX_WINDOWS = 1024


@torch.no_grad()
def evaluate_tokens(
    model: ByteTransformer, tokens: torch.Tensor, seq_length: int, max_windows: int,
) -> dict[str, object]:
    """Score consecutive, non-overlapping windows from token 0; no weights are changed.

    Window i feeds tokens[start:end] and scores next-token targets tokens[start+1:end+1].
    The last window may be shorter so every target is scored at most once.
    """
    if type(seq_length) is not int or not 1 <= seq_length <= model.config.context_length:
        raise ValueError(f"seq_length must be 1..{model.config.context_length}")
    if type(max_windows) is not int or not 1 <= max_windows <= MAX_WINDOWS:
        raise ValueError(f"max_windows must be 1..{MAX_WINDOWS}")
    total_targets = tokens.numel() - 1
    if total_targets < 1:
        raise ValueError("split needs at least two tokens")
    was_training = model.training
    model.eval()
    windows = []
    loss_sum = 0.0
    try:
        start = 0
        while start < total_targets and len(windows) < max_windows:
            end = min(start + seq_length, total_targets)
            logits, _ = model(tokens[start:end].unsqueeze(0))
            window_sum = F.cross_entropy(
                logits[0], tokens[start + 1 : end + 1], reduction="sum"
            ).item()
            if not math.isfinite(window_sum):
                raise ValueError("evaluation loss is not finite")
            loss_sum += window_sum
            windows.append({
                "input_tokens": [start, end],
                "target_tokens": [start + 1, end + 1],
                "loss": window_sum / (end - start),
            })
            start = end
    finally:
        model.train(was_training)
    evaluated = start
    return {
        "split_tokens": tokens.numel(),
        "evaluated_target_tokens": evaluated,
        "covered_entire_split": evaluated == total_targets,
        "seq_length": seq_length,
        "windows": windows,
        "loss": loss_sum / evaluated,
    }


def evaluate(
    checkpoint_path: Path, data_dir: Path, *, seq_length: int | None = None,
    max_windows: int = 64,
) -> dict[str, object]:
    """Load the checkpoint and validation split read-only and report token-weighted loss."""
    start = time.perf_counter()
    model, info = load_checkpoint(checkpoint_path)
    corpus, fingerprint = load_corpus(Path(data_dir), 1)
    length = model.config.context_length if seq_length is None else seq_length
    report = evaluate_tokens(model, corpus["validation"], length, max_windows)
    return {
        "checkpoint_sha256": info["sha256"],
        "checkpoint_step": info["step"],
        "data_fingerprint": fingerprint,
        "corpus_matches_checkpoint": info["data_fingerprint"] == fingerprint,
        "split": "validation",
        **report,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model_lab/runs/train-v1/checkpoint.pt"))
    parser.add_argument("--data", type=Path, default=Path("model_lab/runs/demo-v1"))
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--report", type=Path, help="write JSON here; never overwrites")
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 12:
        parser.error("--cpu-threads must be 1..12")
    torch.set_num_threads(args.cpu_threads)
    result = evaluate(
        args.checkpoint, args.data, seq_length=args.seq_length, max_windows=args.max_windows
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as target:
            target.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
