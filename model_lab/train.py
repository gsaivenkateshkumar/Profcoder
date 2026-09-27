"""Bounded, offline CPU training with resumable checkpoints and held-out loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import torch

from model_lab.model import ByteTransformer, ModelConfig
from model_lab.resources import peak_rss_bytes
from model_lab.tokenizer import TOKENIZER_VERSION, VOCAB_SIZE

MAX_STEPS = 1000
MAX_SECONDS = 1800
MAX_TOKEN_FILE_BYTES = 2 * (16 * 1024 * 1024 + 1000)


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    max_steps: int = 100
    max_seconds: float = 1800
    batch_size: int = 8
    seq_length: int = 64
    learning_rate: float = 3e-4
    eval_every: int = 10
    eval_windows: int = 4
    seed: int = 1337
    cpu_threads: int = 6

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_steps", MAX_STEPS), ("batch_size", 32),
            ("eval_every", MAX_STEPS), ("eval_windows", 16),
            ("cpu_threads", 12),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be 1..{maximum}")
        if type(self.seq_length) is not int or not 2 <= self.seq_length <= self.model.context_length:
            raise ValueError("seq_length must fit model context")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= MAX_SECONDS:
            raise ValueError("max_seconds must be within the 30-minute budget")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.01:
            raise ValueError("learning_rate must be within (0, 0.01]")

    def resume_settings(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


def load_corpus(data_dir: Path, seq_length: int) -> tuple[dict[str, torch.Tensor], str]:
    """Check the prepared corpus's version, bounds, and token IDs before training."""
    root = Path(data_dir)
    metadata_path = root / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("prepared corpus needs metadata.json")
    if metadata_path.stat().st_size > 64 * 1024:
        raise ValueError("corpus metadata is too large")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "uint16-le"
        or metadata.get("tokenizer") != TOKENIZER_VERSION
        or metadata.get("vocab_size") != VOCAB_SIZE
        or not isinstance(metadata.get("tokens"), dict)
    ):
        raise ValueError("unsupported prepared corpus")
    digest = hashlib.sha256(metadata_bytes)
    splits: dict[str, torch.Tensor] = {}
    for name in ("train", "validation"):
        path = root / f"{name}.u16le"
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("prepared split must be a regular file")
        byte_count = path.stat().st_size
        if not 0 < byte_count <= MAX_TOKEN_FILE_BYTES or byte_count % 2:
            raise ValueError("invalid prepared split size")
        raw = path.read_bytes()
        if len(raw) != byte_count:
            raise ValueError("prepared split changed while reading")
        values = array("H")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        expected = metadata["tokens"].get(name)
        if type(expected) is not int or expected != len(values) or len(values) <= seq_length:
            raise ValueError("prepared split has too few tokens or inconsistent metadata")
        if max(values) >= VOCAB_SIZE:
            raise ValueError("prepared split has invalid token IDs")
        digest.update(name.encode("ascii") + b"\x00" + raw)
        splits[name] = torch.tensor(list(values), dtype=torch.long, device="cpu")
    return splits, digest.hexdigest()


def sample_batch(
    tokens: torch.Tensor, seq_length: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        tokens.numel() - seq_length, (batch_size,), generator=generator, device="cpu"
    )
    offsets = torch.arange(seq_length)
    indices = starts[:, None] + offsets[None, :]
    return tokens[indices], tokens[indices + 1]


@torch.no_grad()
def validation_loss(
    model: ByteTransformer, tokens: torch.Tensor, seq_length: int, windows: int,
    deadline: float,
) -> float | None:
    """Measure fixed, evenly spaced validation windows without training on them."""
    max_start = tokens.numel() - seq_length - 1
    if windows == 1:
        starts = [0]
    else:
        starts = sorted({round(i * max_start / (windows - 1)) for i in range(windows)})
    was_training = model.training
    model.eval()
    losses = []
    try:
        for start in starts:
            if time.perf_counter() >= deadline:
                return None
            _, loss = model(
                tokens[start : start + seq_length].unsqueeze(0),
                tokens[start + 1 : start + seq_length + 1].unsqueeze(0),
            )
            losses.append(loss.item())
    finally:
        model.train(was_training)
    average = sum(losses) / len(losses)
    if not math.isfinite(average):
        raise ValueError("validation loss is not finite")
    return average


def write_checkpoint(
    path: Path, model: ByteTransformer, optimizer: torch.optim.Optimizer,
    sampler: torch.Generator, step: int, fingerprint: str, config: TrainConfig,
    last_validation_loss: float | None,
) -> None:
    payload = {
        "version": 1,
        "step": step,
        "data_fingerprint": fingerprint,
        "model_config": config.model.as_dict(),
        "train_settings": config.resume_settings(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "sampling_rng_state": sampler.get_state(),
        "validation_loss": last_validation_loss,
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train(
    data_dir: Path, run_dir: Path, config: TrainConfig = TrainConfig(), *,
    resume: bool = False,
) -> dict[str, object]:
    """Train at most MAX_STEPS (1000) total steps; each invocation lasts at most ~30 minutes."""
    start = time.perf_counter()
    cpu_start = time.process_time()
    deadline = start + config.max_seconds
    torch.set_num_threads(config.cpu_threads)
    corpus, fingerprint = load_corpus(data_dir, config.seq_length)
    run_dir = Path(run_dir)
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError("run directory is missing")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(config.seed)
    model = ByteTransformer(config.model).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    sampler = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    checkpoint_path = run_dir / "checkpoint.pt"
    step = 0
    val_loss = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("version") != 1
            or checkpoint.get("model_config") != config.model.as_dict()
            or checkpoint.get("train_settings") != config.resume_settings()
            or checkpoint.get("data_fingerprint") != fingerprint
            or type(checkpoint.get("step")) is not int
            or not 0 <= checkpoint["step"] <= config.max_steps
        ):
            raise ValueError("checkpoint does not match model, corpus, or training settings")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        sampler.set_state(checkpoint["sampling_rng_state"])
        step = checkpoint["step"]
        val_loss = checkpoint["validation_loss"]

    starting_step = step
    latest_train_loss = None
    training_seconds = 0.0
    while step < config.max_steps and time.perf_counter() < deadline:
        step_start = time.perf_counter()
        x, y = sample_batch(corpus["train"], config.seq_length, config.batch_size, sampler)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss).item():
            raise ValueError("train loss is not finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm).item():
            raise ValueError("train gradients are not finite")
        optimizer.step()
        training_seconds += time.perf_counter() - step_start
        latest_train_loss = loss.item()
        step += 1
        if step % config.eval_every == 0 or step == config.max_steps:
            measured = validation_loss(
                model, corpus["validation"], config.seq_length, config.eval_windows, deadline
            )
            if measured is not None:
                val_loss = measured
            write_checkpoint(
                checkpoint_path, model, optimizer, sampler, step, fingerprint, config, val_loss
            )

    needs_final_validation = val_loss is None or (
        starting_step != step and step % config.eval_every != 0 and step != config.max_steps
    )
    if needs_final_validation:
        measured = validation_loss(
            model, corpus["validation"], config.seq_length, config.eval_windows, deadline
        )
        if measured is not None:
            val_loss = measured
    write_checkpoint(checkpoint_path, model, optimizer, sampler, step, fingerprint, config, val_loss)
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    trained_tokens = (step - starting_step) * config.batch_size * config.seq_length
    summary: dict[str, object] = {
        "step": step,
        "steps_this_invocation": step - starting_step,
        "training_tokens_this_invocation": trained_tokens,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": round(elapsed, 3),
        "process_cpu_seconds": round(cpu_seconds, 3),
        "average_cpu_cores": round(cpu_seconds / elapsed, 2) if elapsed else None,
        "training_tokens_per_second": (
            round(trained_tokens / training_seconds, 2) if training_seconds else None
        ),
        "peak_process_rss_bytes": peak_rss_bytes(),
        "last_train_loss": latest_train_loss,
        "validation_loss": val_loss,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "resumed": resume,
        "stopped_at_time_limit": step < config.max_steps and time.perf_counter() >= deadline,
    }
    summary_path = run_dir / "summary.json"
    temp_summary = run_dir / "summary.tmp"
    temp_summary.write_bytes(
        (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    os.replace(temp_summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("model_lab/runs/demo-v1"))
    parser.add_argument("--run", type=Path, default=Path("model_lab/runs/train-v1"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-windows", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = TrainConfig(
        max_steps=args.max_steps, max_seconds=args.max_seconds,
        batch_size=args.batch_size, seq_length=args.seq_length,
        eval_every=args.eval_every, eval_windows=args.eval_windows,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(train(args.data, args.run, settings, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
