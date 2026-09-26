"""Offline CPU text generation from this project's own training checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import stat
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from model_lab.model import ByteTransformer, ModelConfig
from model_lab.tokenizer import BOS_ID, EOS_ID, PAD_ID, decode, encode

MAX_PROMPT_BYTES = 4096
MAX_NEW_TOKENS = 512
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_TEMPERATURE = 2.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    stopped_at_eos: bool


def load_model(checkpoint_path: Path) -> ByteTransformer:
    """Rebuild the model from a version-1 checkpoint written by model_lab.train."""
    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("checkpoint must be a regular file")
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint is too large")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("unsupported checkpoint")
    model_config = checkpoint.get("model_config")
    if (
        checkpoint.get("version") != 1
        or not isinstance(model_config, dict)
        or set(model_config) != set(ModelConfig().as_dict())
        or not all(type(value) is int for value in model_config.values())
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ValueError("unsupported checkpoint")
    model = ByteTransformer(ModelConfig(**model_config))
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint weights do not match its model configuration") from error
    return model.eval()


@torch.no_grad()
def generate(
    model: ByteTransformer, prompt: str, max_new_tokens: int, *,
    temperature: float = 0.0, top_k: int | None = None, seed: int = 0,
) -> GenerationResult:
    """Greedy by default; stops at EOS or after max_new_tokens. Never emits BOS/PAD."""
    prompt_ids = encode(prompt)
    if len(prompt_ids) > MAX_PROMPT_BYTES:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_BYTES} UTF-8 bytes")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
    if not math.isfinite(temperature) or not 0 <= temperature <= MAX_TEMPERATURE:
        raise ValueError(f"temperature must be within [0, {MAX_TEMPERATURE}]")
    vocab_size = model.config.vocab_size
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= vocab_size):
        raise ValueError(f"top_k must be 1..{vocab_size}")

    context_length = model.config.context_length
    # Training documents are raw bytes followed by EOS, so an empty prompt starts after EOS.
    tokens = prompt_ids or [EOS_ID]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated: list[int] = []
    stopped_at_eos = False
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            window = torch.tensor([tokens[-context_length:]], dtype=torch.long, device="cpu")
            logits, _ = model(window)
            logits = logits[0, -1].float()
            logits[[BOS_ID, PAD_ID]] = -math.inf
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, top_k).values[-1]
                    logits = logits.masked_fill(logits < threshold, -math.inf)
                probabilities = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, 1, generator=generator))
            if next_id == EOS_ID:
                stopped_at_eos = True
                break
            generated.append(next_id)
            tokens.append(next_id)
    finally:
        model.train(was_training)
    # A small model can emit invalid UTF-8; show it rather than failing.
    text = decode(generated, errors="replace")
    return GenerationResult(text=text, token_ids=generated, stopped_at_eos=stopped_at_eos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model_lab/runs/train-v1/checkpoint.pt"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 12:
        parser.error("--cpu-threads must be 1..12")
    torch.set_num_threads(args.cpu_threads)
    start = time.perf_counter()
    model = load_model(args.checkpoint)
    result = generate(
        model, args.prompt, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed,
    )
    print(json.dumps({
        "text": result.text,
        "generated_tokens": len(result.token_ids),
        "stopped_at_eos": result.stopped_at_eos,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
