"""Offline CPU generation: checkpoint loading, output bounds, and EOS handling."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from model_lab.prepare import prepare
from model_lab.tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE, encode


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
class CpuGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model_lab.model import ModelConfig
        from model_lab.train import TrainConfig, train

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = cls.root / "data"
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
        prepare(data / "manifest.json", cls.root / "prepared")
        cls.model_config = ModelConfig(
            context_length=16, width=16, heads=4, layers=1, feed_forward_width=32
        )
        train(cls.root / "prepared", cls.root / "run", TrainConfig(
            model=cls.model_config, max_steps=2, batch_size=2, seq_length=8,
            cpu_threads=1, eval_every=1, eval_windows=2, max_seconds=120,
        ))
        cls.checkpoint = cls.root / "run" / "checkpoint.pt"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scripted_model(self, script):
        """A stand-in model whose next-token logits follow `script`, one list per call."""
        import torch
        from torch import nn

        config = self.model_config
        calls = []

        class Scripted(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config

            def forward(self, tokens):
                calls.append(tokens.clone())
                logits = torch.zeros(1, tokens.size(1), VOCAB_SIZE)
                for rank, token_id in enumerate(script[min(len(calls), len(script)) - 1]):
                    logits[0, -1, token_id] = 10.0 - rank
                return logits, None

        return Scripted(), calls

    def test_loads_training_checkpoint_weights_exactly(self):
        import torch
        from model_lab.generate import load_model

        model = load_model(self.checkpoint)
        saved = torch.load(self.checkpoint, weights_only=True)["model"]
        self.assertFalse(model.training)
        self.assertEqual(model.config, self.model_config)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_load_uses_weights_only_and_rejects_arbitrary_objects(self):
        import pickle

        import torch
        from model_lab.generate import load_model

        path = self.root / "unsafe.pt"
        torch.save({"version": 1, "payload": Path("not-a-tensor")}, path)
        with self.assertRaises(pickle.UnpicklingError):
            load_model(path)

    def test_load_rejects_mismatched_or_foreign_checkpoints(self):
        import torch
        from model_lab.generate import load_model

        original = torch.load(self.checkpoint, weights_only=True)
        wrong_width = dict(original, model_config=dict(original["model_config"], width=32))
        wrong_version = dict(original, version=2)
        for name, payload in (("width", wrong_width), ("version", wrong_version), ("list", [1])):
            path = self.root / f"bad-{name}.pt"
            torch.save(payload, path)
            with self.subTest(name), self.assertRaises(ValueError):
                load_model(path)
        with self.assertRaises(ValueError):
            load_model(self.root / "missing.pt")

    def test_real_model_output_respects_token_limit_and_long_prompts(self):
        from model_lab.generate import generate, load_model

        model = load_model(self.checkpoint)
        long_prompt = "def add(a, b):\n" * 10  # exceeds the 16-token context
        for limit in (1, 5, 40):
            with self.subTest(limit=limit):
                result = generate(model, long_prompt, limit, temperature=0.8, seed=limit)
                self.assertLessEqual(len(result.token_ids), limit)
                if not result.stopped_at_eos:
                    self.assertEqual(len(result.token_ids), limit)
                self.assertTrue(all(0 <= t < 256 for t in result.token_ids))
        greedy = [generate(model, "def ", 12).token_ids for _ in range(2)]
        self.assertEqual(greedy[0], greedy[1])

    def test_rejects_out_of_bounds_requests(self):
        from model_lab.generate import MAX_NEW_TOKENS, MAX_PROMPT_BYTES, generate

        model, calls = self.scripted_model([[ord("x")]])
        for kwargs in (
            dict(prompt="a" * (MAX_PROMPT_BYTES + 1), max_new_tokens=1),
            dict(prompt="é" * (MAX_PROMPT_BYTES // 2 + 1), max_new_tokens=1),
            dict(prompt="a", max_new_tokens=0),
            dict(prompt="a", max_new_tokens=MAX_NEW_TOKENS + 1),
            dict(prompt="a", max_new_tokens=1, temperature=-1.0),
            dict(prompt="a", max_new_tokens=1, temperature=float("nan")),
            dict(prompt="a", max_new_tokens=1, temperature=1.0, top_k=0),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(model, **kwargs)
        self.assertEqual(calls, [])

    def test_crops_input_to_context_window(self):
        from model_lab.generate import generate

        model, calls = self.scripted_model([[ord("y")]])
        prompt = "abcdefghijklmnopqrstuvwxyz"
        result = generate(model, prompt, 3)
        self.assertEqual(result.text, "yyy")
        self.assertEqual([c.size(1) for c in calls], [16, 16, 16])
        self.assertEqual(calls[-1][0].tolist(), encode(prompt + "yy")[-16:])

    def test_stops_at_eos_without_emitting_it(self):
        from model_lab.generate import generate

        model, calls = self.scripted_model([[ord("o")], [ord("k")], [EOS_ID], [ord("z")]])
        result = generate(model, "say ", 50)
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.token_ids, [ord("o"), ord("k")])
        self.assertTrue(result.stopped_at_eos)
        self.assertEqual(len(calls), 3)

    def test_eos_is_sampled_stop_and_empty_prompt_starts_after_eos(self):
        from model_lab.generate import generate

        model, calls = self.scripted_model([[EOS_ID]])
        result = generate(model, "", 10, temperature=0.5, top_k=1, seed=3)
        self.assertEqual((result.text, result.token_ids, result.stopped_at_eos), ("", [], True))
        self.assertEqual(calls[0][0].tolist(), [EOS_ID])

    def test_never_emits_bos_or_padding(self):
        from model_lab.generate import generate

        model, _ = self.scripted_model([[BOS_ID, PAD_ID, ord("a")]])
        result = generate(model, "x", 4)
        self.assertEqual(result.text, "aaaa")
        self.assertFalse(result.stopped_at_eos)

    def test_invalid_utf8_output_is_replaced_not_raised(self):
        from model_lab.generate import generate

        model, _ = self.scripted_model([[0xFF], [EOS_ID]])
        self.assertEqual(generate(model, "x", 5).text, "�")


if __name__ == "__main__":
    unittest.main()
