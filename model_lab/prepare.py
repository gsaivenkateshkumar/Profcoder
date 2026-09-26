"""Verify an explicitly reviewed corpus and produce deterministic CPU token files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from array import array
from pathlib import Path, PurePosixPath

from model_lab.tokenizer import TOKENIZER_VERSION, VOCAB_SIZE, encode

MAX_MANIFEST_BYTES = 64 * 1024
MAX_SAMPLES = 1000
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
SPLITS = ("train", "validation")


def _sample_bytes(root: Path, relative_name: str) -> bytes:
    if not isinstance(relative_name, str):
        raise ValueError("sample path must be a string")
    pure = PurePosixPath(relative_name)
    if (
        pure.is_absolute()
        or relative_name != pure.as_posix()
        or "\\" in relative_name
        or ":" in relative_name
        or "\x00" in relative_name
        or len(pure.parts) != 2
        or pure.parts[0] != "samples"
        or pure.suffix != ".txt"
    ):
        raise ValueError("sample must be samples/<name>.txt under the manifest directory")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        metadata = candidate.lstat()
        reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        )
        if stat.S_ISLNK(metadata.st_mode) or reparse:
            raise ValueError("linked sample paths are unsupported")
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("sample must be a regular, single-link file")
    if not 0 < metadata.st_size <= MAX_FILE_BYTES:
        raise ValueError("sample is empty or exceeds the file size limit")
    data = candidate.read_bytes()
    if len(data) > MAX_FILE_BYTES or b"\x00" in data:
        raise ValueError("sample is oversized or contains NUL bytes")
    data.decode("utf-8", errors="strict")
    return data


def prepare(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    """Prepare a tiny reviewed corpus; refuse existing output to prevent overwrite."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.resolve(strict=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be a regular file")
    if not 0 < manifest_path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("manifest is empty or too large")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("manifest needs a bounded list of samples")

    grouped: dict[str, list[tuple[str, bytes]]] = {split: [] for split in SPLITS}
    paths: set[str] = set()
    digests: set[str] = set()
    total_bytes = 0
    for item in samples:
        if not isinstance(item, dict) or set(item) != {
            "path", "split", "sha256", "license", "source", "rights_reviewed"
        }:
            raise ValueError("invalid sample manifest fields")
        relative_name, split, expected = item["path"], item["split"], item["sha256"]
        license_id, source = item["license"], item["source"]
        if type(split) is not str or split not in SPLITS:
            raise ValueError("unsupported split")
        if (
            type(expected) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or type(license_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{1,63}", license_id)
            or type(source) is not str
            or not source.strip()
            or len(source) > 256
            or item["rights_reviewed"] is not True
        ):
            raise ValueError("sample requires a reviewed source, license and SHA-256")
        if relative_name in paths or expected in digests:
            raise ValueError("duplicate file or content across corpus splits")
        data = _sample_bytes(root, relative_name)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("corpus exceeds total size limit")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("sample SHA-256 mismatch")
        paths.add(relative_name)
        digests.add(expected)
        grouped[split].append((relative_name, data))
    if any(not grouped[split] for split in SPLITS):
        raise ValueError("train and validation splits must both contain samples")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    token_counts: dict[str, int] = {}
    for split in SPLITS:
        count = 0
        with (output_dir / f"{split}.u16le").open("xb") as target:
            for _, data in sorted(grouped[split]):
                ids = encode(data.decode("utf-8"), add_eos=True)
                tokens = array("H", ids)
                if sys.byteorder != "little":
                    tokens.byteswap()
                target.write(tokens.tobytes())
                count += len(ids)
        token_counts[split] = count
    metadata: dict[str, object] = {
        "format": "uint16-le",
        "tokenizer": TOKENIZER_VERSION,
        "vocab_size": VOCAB_SIZE,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokens": token_counts,
        "source_bytes": total_bytes,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("model_lab/data/manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("model_lab/runs/demo-v1"))
    args = parser.parse_args()
    metadata = prepare(args.manifest, args.output)
    print("Prepared CPU corpus:", metadata["tokens"], "tokens")


if __name__ == "__main__":
    main()
