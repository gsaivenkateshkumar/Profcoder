"""Read-only held-out evaluation of local checkpoints."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_lab.prepare import prepare

REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
class CpuEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model_lab.model import ModelConfig
        from model_lab.train import TrainConfig, train

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = cls.root / "data"
        samples = []
        for project, split, text in (
            ("adder", "train", "def add(a, b):\n    return a + b\n" * 4),
            ("multiplier", "validation", "def mul(a, b):\n    return a * b\n" * 2),
        ):
            payload = text.encode("utf-8")
            path = data / "samples" / project / "main.txt"
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            samples.append({
                "path": f"samples/{project}/main.txt", "project": project, "split": split,
                "sha256": hashlib.sha256(payload).hexdigest(), "license": "CC0-1.0",
                "source": "Original synthetic test project", "rights_reviewed": True,
            })
        (data / "manifest.json").write_bytes(json.dumps({"version": 2, "samples": samples}).encode())
        cls.prepared = cls.root / "prepared"
        prepare(data / "manifest.json", cls.prepared)
        cls.model_config = ModelConfig(
            context_length=16, width=16, heads=4, layers=1, feed_forward_width=32
        )
        cls.run_dir = cls.root / "run"
        train(cls.prepared, cls.run_dir, TrainConfig(
            model=cls.model_config, max_steps=2, batch_size=2, seq_length=8,
            cpu_threads=1, eval_every=1, eval_windows=2, max_seconds=120,
        ))
        cls.checkpoint = cls.run_dir / "checkpoint.pt"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        import torch
        from model_lab.generate import load_model

        self.torch = torch
        self.model = load_model(self.checkpoint)
        self.tokens = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 257])

    def test_windows_cover_each_target_once_with_token_weighted_loss(self):
        from model_lab.evaluate import evaluate_tokens

        report = evaluate_tokens(self.model, self.tokens, 3, 64)
        self.assertEqual(
            [(w["input_tokens"], w["target_tokens"]) for w in report["windows"]],
            [([0, 3], [1, 4]), ([3, 6], [4, 7]), ([6, 8], [7, 9])],
        )
        self.assertEqual(report["evaluated_target_tokens"], 8)
        self.assertTrue(report["covered_entire_split"])
        expected = []
        for start, end in ((0, 3), (3, 6), (6, 8)):
            _, loss = self.model(self.tokens[start:end][None], self.tokens[start + 1 : end + 1][None])
            expected.append((loss.item(), end - start))
        for window, (loss, _) in zip(report["windows"], expected):
            self.assertAlmostEqual(window["loss"], loss, places=5)
        weighted = sum(loss * count for loss, count in expected) / 8
        self.assertAlmostEqual(report["loss"], weighted, places=5)

        _, single = self.model(self.tokens[:8][None], self.tokens[1:][None])
        whole = evaluate_tokens(self.model, self.tokens, 8, 1)
        self.assertEqual(len(whole["windows"]), 1)
        self.assertAlmostEqual(whole["loss"], single.item(), places=5)

    def test_window_limit_is_reported_as_partial_coverage(self):
        from model_lab.evaluate import evaluate_tokens

        report = evaluate_tokens(self.model, self.tokens, 3, 2)
        self.assertEqual(len(report["windows"]), 2)
        self.assertEqual(report["evaluated_target_tokens"], 6)
        self.assertFalse(report["covered_entire_split"])

    def test_rejects_out_of_bounds_settings(self):
        from model_lab.evaluate import MAX_WINDOWS, evaluate_tokens

        for seq_length, windows in ((0, 1), (17, 1), (4, 0), (4, MAX_WINDOWS + 1), (4.0, 1)):
            with self.subTest(seq_length=seq_length, windows=windows):
                with self.assertRaises(ValueError):
                    evaluate_tokens(self.model, self.tokens, seq_length, windows)
        with self.assertRaises(ValueError):
            evaluate_tokens(self.model, self.tokens[:1], 4, 1)

    def test_evaluation_changes_no_weights_or_files(self):
        from model_lab.evaluate import evaluate, evaluate_tokens

        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        evaluate_tokens(self.model, self.tokens, 4, 4)
        self.assertTrue(self.model.training)
        for key, value in self.model.state_dict().items():
            self.torch.testing.assert_close(value, before[key], rtol=0, atol=0)

        snapshot = {p.name: p.read_bytes() for p in (*self.run_dir.iterdir(), *self.prepared.iterdir())}
        report = evaluate(self.checkpoint, self.prepared, seq_length=8)
        after = {p.name: p.read_bytes() for p in (*self.run_dir.iterdir(), *self.prepared.iterdir())}
        self.assertEqual(snapshot, after)
        self.assertEqual(
            report["checkpoint_sha256"], hashlib.sha256(snapshot["checkpoint.pt"]).hexdigest()
        )
        self.assertEqual(report["checkpoint_step"], 2)
        self.assertTrue(report["corpus_matches_checkpoint"])
        self.assertEqual(report["split"], "validation")
        validation_tokens = json.loads(snapshot["metadata.json"])["tokens"]["validation"]
        self.assertEqual(report["split_tokens"], validation_tokens)
        self.assertEqual(report["evaluated_target_tokens"], validation_tokens - 1)

    def test_reports_a_different_corpus(self):
        from model_lab.evaluate import evaluate

        other = self.root / "other-prepared"
        shutil.copytree(self.prepared, other)
        raw = bytearray((other / "validation.u16le").read_bytes())
        raw[0] ^= 1
        (other / "validation.u16le").write_bytes(raw)
        self.assertFalse(evaluate(self.checkpoint, other)["corpus_matches_checkpoint"])

    def test_cli_does_not_overwrite_reports_or_import_the_api(self):
        existing = self.root / "report.json"
        existing.write_text("keep", encoding="utf-8")
        command = [
            sys.executable, "-W", "ignore", "-m", "model_lab.evaluate",
            "--checkpoint", str(self.checkpoint), "--data", str(self.prepared),
            "--cpu-threads", "1",
        ]
        refused = subprocess.run(
            command + ["--report", str(existing)], cwd=REPO_ROOT, capture_output=True, text=True
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

        created = self.root / "new-report.json"
        completed = subprocess.run(
            command + ["--report", str(created)], cwd=REPO_ROOT, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(created.read_text(encoding="utf-8")), json.loads(completed.stdout))

        probe = subprocess.run(
            [sys.executable, "-W", "ignore", "-c",
             "import sys, model_lab.evaluate; print(sorted(m for m in sys.modules "
             "if m.split('.')[0] in {'app', 'groq', 'httpx', 'requests'}))"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
