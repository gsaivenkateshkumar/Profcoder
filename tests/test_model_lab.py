"""Corpus integrity and offline tokenizer tests; no model or network needed."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from model_lab.prepare import prepare
from model_lab.tokenizer import EOS_ID, VOCAB_SIZE, decode, encode


class ByteTokenizerTests(unittest.TestCase):
    def test_unicode_round_trip_and_document_boundary(self):
        sample = 'def greet():\n    return "வணக்கம்"\n'
        ids = encode(sample, add_eos=True)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(decode(ids, skip_special=True), sample)
        with self.assertRaises(ValueError):
            decode(ids)
        with self.assertRaises(ValueError):
            decode([VOCAB_SIZE])


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "samples").mkdir()
        self.samples = {
            "train": b"def add(a, b):\n    return a + b\n",
            "validation": b"def square(n):\n    return n * n\n",
        }
        self.items = []
        for split, data in self.samples.items():
            name = f"samples/{split}.txt"
            (self.root / name).write_bytes(data)
            self.items.append({
                "path": name,
                "split": split,
                "sha256": hashlib.sha256(data).hexdigest(),
                "license": "CC0-1.0",
                "source": "Synthetic test fixture",
                "rights_reviewed": True,
            })
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_bytes(
            (json.dumps({"version": 1, "samples": self.items}, indent=2) + "\n").encode("utf-8")
        )

    def test_verified_corpus_is_deterministic_and_not_overwritten(self):
        first = self.root / "first"
        second = self.root / "second"
        metadata = prepare(self.manifest, first)
        self.assertEqual(metadata, prepare(self.manifest, second))
        for split, original in self.samples.items():
            binary = (first / f"{split}.u16le").read_bytes()
            self.assertEqual(binary, (second / f"{split}.u16le").read_bytes())
            ids = struct.unpack("<" + "H" * (len(binary) // 2), binary)
            self.assertEqual(list(ids), encode(original.decode("utf-8"), add_eos=True))
        with self.assertRaises(FileExistsError):
            prepare(self.manifest, first)

    def test_windows_crlf_checkout_matches_lf_corpus_and_metadata(self):
        first = self.root / "lf"
        second = self.root / "crlf"
        metadata = prepare(self.manifest, first)
        for split, data in self.samples.items():
            (self.root / f"samples/{split}.txt").write_bytes(
                data.replace(b"\n", b"\r\n")
            )
        self.manifest.write_bytes(self.manifest.read_bytes().replace(b"\n", b"\r\n"))
        self.assertIn(b"\r\n", self.manifest.read_bytes())
        self.assertEqual(prepare(self.manifest, second), metadata)
        for filename in ("train.u16le", "validation.u16le", "metadata.json"):
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_tampering_and_unreviewed_rights_are_rejected(self):
        (self.root / "samples/train.txt").write_text("altered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare(self.manifest, self.root / "out1")
        (self.root / "samples/train.txt").write_bytes(self.samples["train"])
        self.items[0]["rights_reviewed"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "reviewed"):
            prepare(self.manifest, self.root / "out2")

    def test_traversal_and_symlink_are_rejected(self):
        self.items[0]["path"] = "samples/../outside.txt"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "sample must be"):
            prepare(self.manifest, self.root / "out1")

        self.items[0]["path"] = "samples/train.txt"
        target = self.root / "samples/train.txt"
        target.unlink()
        try:
            target.symlink_to(self.root / "samples/validation.txt")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this Windows configuration")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "linked"):
            prepare(self.manifest, self.root / "out2")

    def test_version_1_keeps_its_fields_and_metadata(self):
        self.items[0]["project"] = "demo"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "fields"):
            prepare(self.manifest, self.root / "out1")
        del self.items[0]["project"]
        self.write_manifest()
        metadata = prepare(self.manifest, self.root / "out2")
        self.assertNotIn("projects", metadata)
        self.assertNotIn("manifest_version", metadata)

    def test_committed_demo_manifest_output_is_unchanged(self):
        manifest = Path(__file__).resolve().parents[1] / "model_lab/data/manifest.json"
        metadata = prepare(manifest, self.root / "demo")
        self.assertEqual(metadata["tokens"], {"train": 175, "validation": 99})
        self.assertEqual(
            metadata["manifest_sha256"],
            "def164bc4c0f92eb7bc32bbefbac5285fe549c473d606dc18cd4dab5fc354f6a",
        )


class ProjectManifestTests(unittest.TestCase):
    """Version 2: every sample names a source project held entirely in one split."""

    PROJECTS = {
        "calc-tools": ("train", {
            "add.txt": b"def add(a, b):\n    return a + b\n",
            "sub.txt": b"def sub(a, b):\n    return a - b\n",
        }),
        "greeter": ("train", {"greet.txt": b"def greet(name):\n    return 'hi ' + name\n"}),
        "shapes": ("validation", {"area.txt": b"def area(w, h):\n    return w * h\n"}),
    }

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = []
        for project, (split, files) in self.PROJECTS.items():
            for name, data in files.items():
                self.add_file(project, split, name, data)
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def add_file(self, project, split, name, data):
        path = self.root / "samples" / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        item = {
            "path": f"samples/{project}/{name}", "project": project, "split": split,
            "sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
            "license": "CC0-1.0", "source": "Original synthetic test project",
            "rights_reviewed": True,
        }
        self.items.append(item)
        return item

    def write_manifest(self):
        self.manifest.write_bytes(
            json.dumps({"version": 2, "samples": self.items}, indent=2).encode("utf-8")
        )

    def assert_rejected(self, pattern, name="out"):
        output = self.root / name
        with self.assertRaisesRegex(ValueError, pattern):
            prepare(self.manifest, output)
        self.assertFalse(output.exists())

    def test_projects_are_recorded_and_tokenized_by_split(self):
        metadata = prepare(self.manifest, self.root / "out")
        self.assertEqual(metadata["manifest_version"], 2)
        self.assertEqual(
            metadata["projects"],
            {"train": ["calc-tools", "greeter"], "validation": ["shapes"]},
        )
        binary = (self.root / "out/validation.u16le").read_bytes()
        ids = struct.unpack("<" + "H" * (len(binary) // 2), binary)
        area = self.PROJECTS["shapes"][1]["area.txt"].decode("utf-8")
        self.assertEqual(list(ids), encode(area, add_eos=True))

    def test_project_cannot_span_splits(self):
        self.add_file("calc-tools", "validation", "mul.txt", b"def mul(a, b):\n    return a * b\n")
        self.write_manifest()
        self.assert_rejected("project spans train and validation")
        # Order does not matter: a validation project cannot later gain training files.
        self.items.insert(0, self.items.pop())
        self.write_manifest()
        self.assert_rejected("project spans train and validation")

    def test_duplicate_content_is_rejected_across_splits_and_newlines(self):
        copied = self.PROJECTS["calc-tools"][1]["add.txt"]
        self.add_file("shapes", "validation", "copied.txt", copied.replace(b"\n", b"\r\n"))
        self.write_manifest()
        self.assert_rejected("duplicate")

    def test_malformed_paths_are_rejected(self):
        item = self.items[0]
        for bad_path in (
            "samples/greeter/add.txt",          # another project's directory
            "samples/add.txt",                  # missing project directory
            "samples/calc-tools/nested/add.txt",
            "samples/calc-tools/../calc-tools/add.txt",
            "samples/calc-tools/./add.txt",
            "/samples/calc-tools/add.txt",
            "samples\\calc-tools\\add.txt",
            "C:samples/calc-tools/add.txt",
            "samples/calc-tools/add.py",
            "other/calc-tools/add.txt",
            42,
        ):
            with self.subTest(path=bad_path):
                item["path"] = bad_path
                self.write_manifest()
                self.assert_rejected("sample")

    def test_malformed_hashes_are_rejected(self):
        item = self.items[0]
        good = item["sha256"]
        for bad_hash, pattern in (
            (good.upper(), "SHA-256"), (good[:-1], "SHA-256"), (None, "SHA-256"),
            (hashlib.sha256(b"other").hexdigest(), "SHA-256 mismatch"),
        ):
            with self.subTest(sha256=bad_hash):
                item["sha256"] = bad_hash
                self.write_manifest()
                self.assert_rejected(pattern)

    def test_rights_and_project_metadata_are_required(self):
        for field, value, pattern in (
            ("rights_reviewed", False, "reviewed"),
            ("rights_reviewed", "true", "reviewed"),
            ("rights_reviewed", 1, "reviewed"),
            ("license", "", "license"),
            ("license", "CC0 1.0", "license"),
            ("license", None, "license"),
            ("source", "   ", "source"),
            ("source", "x" * 257, "source"),
            ("project", "Calc-Tools", "project"),
            ("project", "../calc-tools", "project"),
            ("project", "", "project"),
            ("split", "test", "split"),
        ):
            with self.subTest(field=field, value=value):
                self.setUp()
                self.items[0][field] = value
                self.write_manifest()
                self.assert_rejected(pattern)
        for missing in ("project", "license", "source", "rights_reviewed"):
            with self.subTest(missing=missing):
                self.setUp()
                del self.items[0][missing]
                self.write_manifest()
                self.assert_rejected("fields")

    def test_unknown_manifest_versions_are_rejected(self):
        for version in (0, 4, True, "2", None):
            with self.subTest(version=version):
                self.manifest.write_bytes(
                    json.dumps({"version": version, "samples": self.items}).encode("utf-8")
                )
                self.assert_rejected("version")


def owned_project(split, repository, **overrides):
    project = {
        "split": split, "repository": repository, "commit": "a" * 40,
        "rights_basis": "owner-authorized-local-use", "license": None,
        "authorized_by": "Repository owner, for local experiments only",
        "rights_reviewed": True,
    }
    project.update(overrides)
    return project


class SourceProvenanceManifestTests(unittest.TestCase):
    """Version 3: per-project source commit and rights, per-file source path."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.projects = {
            "alpha": owned_project("train", "https://example.invalid/owner/alpha"),
            "beta": owned_project("validation", "https://example.invalid/owner/beta"),
        }
        self.samples = []
        self.add_sample("alpha", "pkg/core.py", b"def core():\n    return 1\n")
        self.add_sample("alpha", "pkg/util.py", b"def util(x):\n    return x + 1\n")
        self.add_sample("beta", "main.py", b"def main():\n    print('beta')\n", "redacted")
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def add_sample(self, project, source_path, data, transform="verbatim"):
        path = f"samples/{project}/{source_path.replace('/', '__')}.txt"
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_bytes(data)
        self.samples.append({
            "path": path, "project": project, "source_path": source_path,
            "sha256": hashlib.sha256(data).hexdigest(), "transform": transform,
        })

    def write_manifest(self, **extra):
        manifest = {"version": 3, "projects": self.projects, "samples": self.samples, **extra}
        self.manifest.write_bytes(json.dumps(manifest).encode("utf-8"))

    def assert_rejected(self, pattern):
        self.write_manifest()
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, pattern):
            prepare(self.manifest, output)
        self.assertFalse(output.exists())

    def test_owner_authorized_projects_prepare_as_local_only(self):
        metadata = prepare(self.manifest, self.root / "out")
        self.assertEqual(metadata["manifest_version"], 3)
        self.assertEqual(metadata["projects"], {"train": ["alpha"], "validation": ["beta"]})
        self.assertTrue(metadata["local_only"])
        self.assertEqual(metadata["sources"]["beta"], {
            "repository": "https://example.invalid/owner/beta", "commit": "a" * 40,
            "rights_basis": "owner-authorized-local-use", "license": None,
        })

    def test_public_license_projects_are_not_local_only(self):
        for name in self.projects:
            self.projects[name].update(rights_basis="public-license", license="MIT", authorized_by=None)
        self.write_manifest()
        self.assertFalse(prepare(self.manifest, self.root / "out")["local_only"])

    def test_rights_cannot_be_mislabeled(self):
        for overrides in (
            {"license": "CC0-1.0"},                       # owner-only use is not a public license
            {"authorized_by": None},
            {"authorized_by": "  "},
            {"rights_basis": "public-license"},           # a public license needs an ID
            {"rights_basis": "public-license", "license": "MIT"},  # and no owner note
            {"rights_basis": "unknown"},
            {"rights_reviewed": False},
            {"rights_reviewed": "true"},
        ):
            with self.subTest(overrides=overrides):
                original = dict(self.projects["alpha"])
                self.projects["alpha"].update(overrides)
                self.assert_rejected("rights|license|authorized")
                self.projects["alpha"] = original
        del self.projects["alpha"]["license"]
        self.assert_rejected("fields")

    def test_source_provenance_is_required(self):
        for field, value in (
            ("commit", "a" * 39), ("commit", "A" * 40), ("commit", None),
            ("repository", "http://example.invalid/owner/alpha"),
            ("repository", "https://example.invalid"), ("repository", "file:///tmp/alpha"),
        ):
            with self.subTest(field=field, value=value):
                original = self.projects["alpha"][field]
                self.projects["alpha"][field] = value
                self.assert_rejected("repository and full commit")
                self.projects["alpha"][field] = original

    def test_one_repository_cannot_be_split_into_two_projects(self):
        self.projects["beta"]["repository"] = "https://EXAMPLE.invalid/owner/alpha.git"
        self.assert_rejected("more than one project")

    def test_malformed_source_paths_and_transforms_are_rejected(self):
        item = self.samples[0]
        for bad in ("", "../core.py", "/pkg/core.py", "pkg\\core.py", "C:pkg/core.py",
                    "pkg/./core.py", "pkg//core.py", 7):
            with self.subTest(source_path=bad):
                item["source_path"] = bad
                self.assert_rejected("source_path")
        item["source_path"] = self.samples[1]["source_path"]
        self.assert_rejected("duplicate source file")
        item["source_path"] = "pkg/core.py"
        item["transform"] = "rewritten"
        self.assert_rejected("transform")

    def test_samples_take_split_from_their_declared_project(self):
        self.samples[0]["split"] = "validation"
        self.assert_rejected("fields")
        del self.samples[0]["split"]
        self.samples[0]["project"] = "gamma"
        self.assert_rejected("undeclared project")
        self.samples[0]["project"] = "alpha"
        self.projects["gamma"] = owned_project("train", "https://example.invalid/owner/gamma")
        self.assert_rejected("at least one sample")

    def test_content_cannot_leak_between_projects(self):
        self.add_sample("beta", "copy.py", b"def core():\r\n    return 1\r\n")
        self.samples[-1]["sha256"] = self.samples[0]["sha256"]
        self.assert_rejected("duplicate")
        self.samples.pop()
        self.samples[2]["path"] = self.samples[0]["path"]  # beta file stored under alpha
        self.assert_rejected("duplicate|sample must be")

    def test_unexpected_top_level_fields_are_rejected(self):
        self.write_manifest(license="CC0-1.0")
        with self.assertRaisesRegex(ValueError, "projects and samples"):
            prepare(self.manifest, self.root / "out")


GIT = os.environ.get("PROFCODER_GIT") or shutil.which("git")


@unittest.skipUnless(GIT, "git executable not found; set PROFCODER_GIT")
class BuildCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clones = {}
        self.commits = {}
        for name, files in (
            ("alpha", {"pkg/core.py": b"def core():\r\n    return 1\n",
                       "model.pt": b"version https://git-lfs.github.com/spec/v1\noid sha256:0\n"}),
            ("beta", {"app.py": b"PASSWORD = 'x'\ndef run():\n    return PASSWORD\n"}),
        ):
            clone = self.root / name
            clone.mkdir()
            self.git(clone, "init", "-q")
            for rel, data in files.items():
                (clone / rel).parent.mkdir(parents=True, exist_ok=True)
                (clone / rel).write_bytes(data)
            self.git(clone, "add", ".")
            self.git(clone, "commit", "-q", "-m", "fixture")
            self.clones[name] = clone
            self.commits[name] = self.git(clone, "rev-parse", "HEAD").strip()
        # Uncommitted edits must not reach the corpus.
        (self.clones["alpha"] / "pkg/core.py").write_bytes(b"def core():\n    return 'edited'\n")
        self.redacted = self.root / "app_redacted.py"
        self.redacted.write_bytes(b"PASSWORD = '<redacted>'\ndef run():\n    return PASSWORD\n")
        self.spec = {"projects": {
            "alpha": self.spec_project("alpha", "train", ["pkg/core.py"]),
            "beta": self.spec_project("beta", "validation", ["app.py"],
                                      {"app.py": str(self.redacted)}),
        }}

    def git(self, clone, *args):
        return subprocess.run(
            [GIT, "-C", str(clone), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True, text=True,
        ).stdout

    def spec_project(self, name, split, files, redacted=None):
        project = owned_project(split, f"https://example.invalid/owner/{name}")
        project.update(commit=self.commits[name], clone=str(self.clones[name]),
                       files=files, redacted=redacted or {})
        return project

    def build(self, name="corpus"):
        from model_lab.build_corpus import build

        spec_path = self.root / f"{name}-spec.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        return build(spec_path, self.root / name, git=GIT)

    def test_copies_committed_content_and_prepares(self):
        self.assertEqual(self.build(), {"alpha": 1, "beta": 1})
        corpus = self.root / "corpus"
        self.assertEqual(
            (corpus / "samples/alpha/pkg__core.py.txt").read_bytes(), b"def core():\n    return 1\n"
        )
        self.assertNotIn(b"'x'", (corpus / "samples/beta/app.py.txt").read_bytes())
        manifest = json.loads((corpus / "manifest.json").read_bytes())
        self.assertEqual(manifest["projects"]["alpha"]["commit"], self.commits["alpha"])
        self.assertNotIn("clone", manifest["projects"]["alpha"])
        self.assertEqual(
            [(s["source_path"], s["transform"]) for s in manifest["samples"]],
            [("pkg/core.py", "verbatim"), ("app.py", "redacted")],
        )
        metadata = prepare(corpus / "manifest.json", self.root / "prepared")
        self.assertEqual(metadata["projects"], {"train": ["alpha"], "validation": ["beta"]})
        with self.assertRaises(FileExistsError):
            self.build()

    def test_rejects_unsafe_or_unverifiable_sources(self):
        cases = (
            ("lfs", lambda s: s["alpha"]["files"].append("model.pt"), "LFS pointer"),
            ("untracked", lambda s: s["alpha"]["files"].append("missing.py"), "git cat-file"),
            ("commit", lambda s: s["alpha"].update(commit="b" * 40), "rev-parse"),
            ("same", lambda s: self.redacted.write_bytes(
                (self.clones["beta"] / "app.py").read_bytes()), "identical"),
            ("unselected", lambda s: s["alpha"]["redacted"].update({"x.py": "x"}), "not selected"),
            ("license", lambda s: s["beta"].update(license="CC0-1.0"), "license"),
        )
        for name, mutate, pattern in cases:
            with self.subTest(name):
                self.setUp()
                mutate(self.spec["projects"])
                with self.assertRaisesRegex(ValueError, pattern):
                    self.build(name)
                self.assertFalse((self.root / name).exists())


if __name__ == "__main__":
    unittest.main()
