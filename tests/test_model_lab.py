"""Corpus integrity and offline tokenizer tests; no model or network needed."""

import hashlib
import json
import struct
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


if __name__ == "__main__":
    unittest.main()
