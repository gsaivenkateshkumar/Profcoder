"""CPU model smoke and exact checkpoint-resume regression tests."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from model_lab.prepare import prepare


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
class CpuModelTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from model_lab.model import ModelConfig

        self.torch = torch
        self.model_config = ModelConfig(
            context_length=16, width=16, heads=4, layers=1, feed_forward_width=32
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / "data"
        (data / "samples").mkdir(parents=True)
        samples = []
        for name, text in (
            ("train", "def add(a, b):\n    return a + b\n" * 4),
            ("validation", "def mul(a, b):\n    return a * b\n" * 4),
        ):
            payload = text.encode("utf-8")
            (data / "samples" / f"{name}.txt").write_bytes(payload)
            samples.append({
                "path": f"samples/{name}.txt", "split": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "license": "CC0-1.0", "source": "Original test fixture", "rights_reviewed": True,
            })
        (data / "manifest.json").write_bytes(json.dumps({"version": 1, "samples": samples}).encode())
        self.prepared = self.root / "prepared"
        prepare(data / "manifest.json", self.prepared)

    def test_causality_and_finite_backward_pass(self):
        from model_lab.model import ByteTransformer

        torch = self.torch
        torch.manual_seed(1)
        model = ByteTransformer(self.model_config)
        model.eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        second = first.clone()
        second[0, 7] = 12
        before, loss = model(first, first)
        after, _ = model(second)
        torch.testing.assert_close(before[:, :7], after[:, :7], rtol=0, atol=0)
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertTrue(all(torch.isfinite(param.grad).all() for param in model.parameters()))
        self.assertIs(model.token_embedding.weight, model.output.weight)

    def test_default_model_matches_parameter_budget(self):
        from model_lab.model import ByteTransformer

        model = ByteTransformer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_878_720)

    def test_resume_matches_uninterrupted_optimizer_and_model(self):
        from model_lab.train import TrainConfig, train

        torch = self.torch
        settings = dict(
            model=self.model_config, batch_size=2, seq_length=8,
            cpu_threads=1, eval_every=1, eval_windows=2, max_seconds=120,
        )
        full = train(self.prepared, self.root / "full", TrainConfig(max_steps=2, **settings))
        train(self.prepared, self.root / "partial", TrainConfig(max_steps=1, **settings))
        resumed = train(
            self.prepared, self.root / "partial", TrainConfig(max_steps=2, **settings), resume=True
        )
        self.assertEqual(full["step"], 2)
        self.assertEqual(resumed["steps_this_invocation"], 1)
        self.assertTrue(math.isfinite(full["validation_loss"]))
        self.assertEqual(full["validation_loss"], resumed["validation_loss"])
        uninterrupted = torch.load(self.root / "full/checkpoint.pt", weights_only=True)
        restored = torch.load(self.root / "partial/checkpoint.pt", weights_only=True)
        for key in uninterrupted["model"]:
            torch.testing.assert_close(
                uninterrupted["model"][key], restored["model"][key], rtol=0, atol=0
            )
        for parameter in uninterrupted["optimizer"]["state"]:
            original_state = uninterrupted["optimizer"]["state"][parameter]
            resumed_state = restored["optimizer"]["state"][parameter]
            for name in original_state:
                torch.testing.assert_close(original_state[name], resumed_state[name], rtol=0, atol=0)
        torch.testing.assert_close(
            uninterrupted["sampling_rng_state"], restored["sampling_rng_state"]
        )

    def test_modified_corpus_rejects_resume(self):
        from model_lab.train import TrainConfig, train

        settings = TrainConfig(
            model=self.model_config, max_steps=2, batch_size=2, seq_length=8,
            cpu_threads=1, eval_every=1,
        )
        run_dir = self.root / "run"
        train(self.prepared, run_dir, settings)
        path = self.prepared / "train.u16le"
        raw = bytearray(path.read_bytes())
        raw[0] = (raw[0] + 1) % 255
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
            train(self.prepared, run_dir, settings, resume=True)


if __name__ == "__main__":
    unittest.main()
