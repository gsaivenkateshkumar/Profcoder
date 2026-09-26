"""Copy explicitly listed, reviewed files from local Git clones into a version-3 corpus.

Each file is read from the Git object store at the project's recorded commit,
not from the working tree, so the manifest's commit is exactly what was copied.
Nothing is downloaded, and existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from model_lab.prepare import (
    MAX_FILE_BYTES, MAX_MANIFEST_BYTES, V3_PROJECT_FIELDS, _version_3_samples,
)

SPEC_ONLY_FIELDS = {"clone", "files", "redacted"}
LFS_POINTER = b"version https://git-lfs"


def _git(git: str, clone: Path, *args: str) -> bytes:
    completed = subprocess.run(
        [git, "-C", str(clone), *args], capture_output=True, check=False, timeout=60
    )
    if completed.returncode != 0:
        raise ValueError(f"git {args[0]} failed in {clone.name}")
    return completed.stdout


def _canonical(raw: bytes, label: str) -> bytes:
    if raw.startswith(LFS_POINTER):
        raise ValueError(f"{label} is a Git LFS pointer, not source text")
    if b"\x00" in raw or not 0 < len(raw) <= MAX_FILE_BYTES:
        raise ValueError(f"{label} is empty, oversized, or binary")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    return text.replace("\r\n", "\n").encode("utf-8")


def build(spec_path: Path, output_dir: Path, *, git: str = "git") -> dict[str, int]:
    spec = json.loads(Path(spec_path).read_bytes())
    projects = spec.get("projects") if isinstance(spec, dict) else None
    if not isinstance(projects, dict) or set(spec) != {"projects"}:
        raise ValueError("spec needs a projects object")
    manifest_projects: dict[str, dict[str, object]] = {}
    planned: list[tuple[str, str, bytes, str]] = []
    for name, project in projects.items():
        if not isinstance(project, dict) or set(project) != V3_PROJECT_FIELDS | SPEC_ONLY_FIELDS:
            raise ValueError(f"invalid spec fields for {name}")
        files, redacted = project["files"], project["redacted"]
        if not isinstance(files, list) or not files or not isinstance(redacted, dict):
            raise ValueError(f"{name} needs an explicit file list and a redacted map")
        if not set(redacted) <= set(files):
            raise ValueError(f"{name} redacts a file that is not selected")
        clone, commit = Path(project["clone"]), project["commit"]
        if _git(git, clone, "rev-parse", "--verify", f"{commit}^{{commit}}").strip().decode() != commit:
            raise ValueError(f"{name} commit is not a full hash present in the clone")
        for source_path in files:
            label = f"{name}:{source_path}"
            if type(source_path) is not str:
                raise ValueError(f"{name} file paths must be strings")
            original = _canonical(_git(git, clone, "cat-file", "blob", f"{commit}:{source_path}"), label)
            data, transform = original, "verbatim"
            if source_path in redacted:
                data = _canonical(Path(redacted[source_path]).read_bytes(), f"redacted {label}")
                if data == original:
                    raise ValueError(f"redacted {label} is identical to the source")
                transform = "redacted"
            planned.append((name, source_path, data, transform))
        manifest_projects[name] = {key: project[key] for key in V3_PROJECT_FIELDS}

    samples = []
    for name, source_path, data, transform in planned:
        samples.append({
            "path": f"samples/{name}/{source_path.replace('/', '__')}.txt",
            "project": name,
            "source_path": source_path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "transform": transform,
        })
    manifest = {"version": 3, "projects": manifest_projects, "samples": samples}
    _version_3_samples(manifest)  # reject bad provenance or rights before writing anything
    encoded = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest would exceed the size limit; select fewer files")
    if len({sample["path"] for sample in samples}) != len(samples):
        raise ValueError("two source files map to the same sample name")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for sample, (_, _, data, _) in zip(samples, planned):
        target = output_dir / sample["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
    with (output_dir / "manifest.json").open("xb") as handle:
        handle.write(encoded)
    counts: dict[str, int] = {}
    for name, *_ in planned:
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git", default="git", help="git executable")
    args = parser.parse_args()
    print(json.dumps(build(args.spec, args.output, git=args.git)))


if __name__ == "__main__":
    main()
