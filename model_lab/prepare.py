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
SAMPLE_FIELDS = {"path", "split", "sha256", "license", "source", "rights_reviewed"}
PROJECT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
SHA256 = re.compile(r"[0-9a-f]{64}")
LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{1,63}")
COMMIT = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._-]+)+")
PUBLIC_LICENSE = "public-license"
OWNER_AUTHORIZED = "owner-authorized-local-use"
V3_PROJECT_FIELDS = {
    "split", "repository", "commit", "rights_basis", "license", "authorized_by",
    "rights_reviewed",
}
V3_SAMPLE_FIELDS = {"path", "project", "source_path", "sha256", "transform"}
TRANSFORMS = ("verbatim", "redacted")


def _version_3_samples(
    manifest: dict[str, object],
) -> tuple[list[tuple[str, str, str, str]], dict[str, dict[str, object]]]:
    """Validate per-project provenance and rights; return (path, split, sha256, project).

    Rights are either a public license (an SPDX-style ID) or the owner's
    authorization for local use only, which records no license at all.
    """
    projects, samples = manifest.get("projects"), manifest.get("samples")
    if set(manifest) != {"version", "projects", "samples"} or not isinstance(projects, dict):
        raise ValueError("version 3 manifest needs projects and samples")
    if not 2 <= len(projects) <= MAX_SAMPLES:
        raise ValueError("manifest needs a bounded set of projects")
    repositories: set[str] = set()
    for name, project in projects.items():
        if not PROJECT_ID.fullmatch(name):
            raise ValueError("project must be a lowercase identifier")
        if not isinstance(project, dict) or set(project) != V3_PROJECT_FIELDS:
            raise ValueError("invalid project manifest fields")
        if project["split"] not in SPLITS or type(project["split"]) is not str:
            raise ValueError("unsupported split")
        repository, commit = project["repository"], project["commit"]
        if (
            type(repository) is not str or len(repository) > 256
            or not REPOSITORY.fullmatch(repository)
            or type(commit) is not str or not COMMIT.fullmatch(commit)
        ):
            raise ValueError("project requires an https source repository and full commit hash")
        # One repository under two project names would let it leak across splits.
        key = repository.lower().removesuffix(".git")
        if key in repositories:
            raise ValueError("source repository is assigned to more than one project")
        repositories.add(key)
        basis, license_id, authorized_by = (
            project["rights_basis"], project["license"], project["authorized_by"]
        )
        if project["rights_reviewed"] is not True:
            raise ValueError("project rights must be reviewed")
        if basis == PUBLIC_LICENSE:
            valid = (
                type(license_id) is str and bool(LICENSE_ID.fullmatch(license_id))
                and authorized_by is None
            )
        elif basis == OWNER_AUTHORIZED:
            valid = (
                license_id is None and type(authorized_by) is str
                and bool(authorized_by.strip()) and len(authorized_by) <= 256
            )
        else:
            raise ValueError("unsupported rights basis")
        if not valid:
            raise ValueError(
                "public-license rights need a license ID; owner-authorized local use "
                "needs authorized_by and no license"
            )

    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("manifest needs a bounded list of samples")
    result = []
    sources: set[tuple[str, str]] = set()
    for item in samples:
        if not isinstance(item, dict) or set(item) != V3_SAMPLE_FIELDS:
            raise ValueError("invalid sample manifest fields")
        project, source_path = item["project"], item["source_path"]
        if type(project) is not str or project not in projects:
            raise ValueError("sample names an undeclared project")
        pure = PurePosixPath(source_path) if type(source_path) is str else None
        if (
            pure is None
            or not 0 < len(source_path) <= 256
            or pure.is_absolute()
            or source_path != pure.as_posix()
            or any(part in (".", "..") for part in pure.parts)
            or any(ch in source_path for ch in "\\:\x00")
        ):
            raise ValueError("source_path must be a relative path in the source repository")
        if (project, source_path) in sources:
            raise ValueError("duplicate source file")
        sources.add((project, source_path))
        if item["transform"] not in TRANSFORMS or type(item["transform"]) is not str:
            raise ValueError("transform must be verbatim or redacted")
        if type(item["sha256"]) is not str or not SHA256.fullmatch(item["sha256"]):
            raise ValueError("sample requires a SHA-256")
        if type(item["path"]) is not str:
            raise ValueError("sample path must be a string")
        result.append((item["path"], str(projects[project]["split"]), item["sha256"], project))
    if {project for _, _, _, project in result} != set(projects):
        raise ValueError("every declared project needs at least one sample")
    return result, projects


def _sample_bytes(root: Path, relative_name: str, project: str | None = None) -> bytes:
    """Version 1 reads samples/<name>.txt; version 2 reads samples/<project>/<name>.txt."""
    if not isinstance(relative_name, str):
        raise ValueError("sample path must be a string")
    pure = PurePosixPath(relative_name)
    expected_parts = ("samples",) if project is None else ("samples", project)
    if (
        pure.is_absolute()
        or relative_name != pure.as_posix()
        or "\\" in relative_name
        or ":" in relative_name
        or "\x00" in relative_name
        or len(pure.parts) != len(expected_parts) + 1
        or pure.parts[:-1] != expected_parts
        or pure.suffix != ".txt"
    ):
        if project is None:
            raise ValueError("sample must be samples/<name>.txt under the manifest directory")
        raise ValueError("sample must be samples/<project>/<name>.txt for its own project")
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
    text = data.decode("utf-8", errors="strict")
    # Git may check out tracked text with CRLF on Windows. Hash and tokenize
    # canonical UTF-8 LF content so the same manifest works on both platforms.
    return text.replace("\r\n", "\n").encode("utf-8")


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
    version = manifest.get("version") if isinstance(manifest, dict) else None
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("unsupported manifest version")
    projects: dict[str, dict[str, object]] = {}
    if version == 3:
        samples, projects = _version_3_samples(manifest)
    else:
        samples = manifest.get("samples")
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("manifest needs a bounded list of samples")

    fields = SAMPLE_FIELDS if version == 1 else SAMPLE_FIELDS | {"project"}
    grouped: dict[str, list[tuple[str, bytes]]] = {split: [] for split in SPLITS}
    project_splits: dict[str, str] = {name: str(p["split"]) for name, p in projects.items()}
    paths: set[str] = set()
    digests: set[str] = set()
    total_bytes = 0
    for item in samples:
        if version == 3:
            # Already validated, with provenance and rights checked per project.
            relative_name, split, expected, project = item
        else:
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError("invalid sample manifest fields")
            relative_name, split, expected = item["path"], item["split"], item["sha256"]
            license_id, source = item["license"], item["source"]
            if type(split) is not str or split not in SPLITS:
                raise ValueError("unsupported split")
            project = None
            if version == 2:
                project = item["project"]
                if type(project) is not str or not PROJECT_ID.fullmatch(project):
                    raise ValueError("project must be a lowercase identifier")
                # A project held out for validation must never contribute training text.
                if project_splits.setdefault(project, split) != split:
                    raise ValueError("project spans train and validation splits")
            if (
                type(expected) is not str
                or not SHA256.fullmatch(expected)
                or type(license_id) is not str
                or not LICENSE_ID.fullmatch(license_id)
                or type(source) is not str
                or not source.strip()
                or len(source) > 256
                or item["rights_reviewed"] is not True
            ):
                raise ValueError("sample requires a reviewed source, license and SHA-256")
        if relative_name in paths or expected in digests:
            raise ValueError("duplicate file or content across corpus splits")
        data = _sample_bytes(root, relative_name, project)
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
        "manifest_sha256": hashlib.sha256(
            manifest_bytes.replace(b"\r\n", b"\n")
        ).hexdigest(),
        "tokens": token_counts,
        "source_bytes": total_bytes,
    }
    if version >= 2:
        # Version 1 metadata stays byte-identical so existing run fingerprints still match.
        metadata["manifest_version"] = version
        metadata["projects"] = {
            split: sorted(p for p, s in project_splits.items() if s == split) for split in SPLITS
        }
    if version == 3:
        metadata["sources"] = {
            name: {key: project[key] for key in ("repository", "commit", "rights_basis", "license")}
            for name, project in sorted(projects.items())
        }
        metadata["local_only"] = any(
            project["rights_basis"] == OWNER_AUTHORIZED for project in projects.values()
        )
    (output_dir / "metadata.json").write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
