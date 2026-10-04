import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model_lab.extract_pairs import MAX_FUNCTIONS_PER_FILE, extract_candidates
from model_lab.tokenizer import encode


def owned_project(split, repository, **overrides):
    project = {
        "split": split, "repository": repository, "commit": "a" * 40,
        "rights_basis": "owner-authorized-local-use", "license": None,
        "authorized_by": "Repository owner, for local experiments only",
        "rights_reviewed": True,
    }
    project.update(overrides)
    return project


class ExtractPairsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.projects = {
            "alpha": owned_project("train", "https://example.invalid/owner/alpha"),
            "beta": owned_project("validation", "https://example.invalid/owner/beta"),
        }
        self.samples = []
        self.manifest = self.root / "manifest.json"
        # Every manifest needs both projects present and both splits non-empty;
        # these harmless fillers satisfy that regardless of what a test adds.
        self.add_file("alpha", "_filler_alpha", (
            "# A harmless filler so this project always has a sample.\n"
            "def filler_alpha(x):\n    return x\n"
        ))
        self.add_file("beta", "_filler_beta", (
            "# A harmless filler so this project always has a sample.\n"
            "def filler_beta(x):\n    return x\n"
        ))

    def add_file(self, project, name, text, source_path=None):
        data = text.encode("utf-8")
        path = f"samples/{project}/{name}.txt"
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_bytes(data)
        self.samples.append({
            "path": path, "project": project, "source_path": source_path or f"{name}.py",
            "sha256": hashlib.sha256(data).hexdigest(), "transform": "verbatim",
        })

    def write_manifest(self):
        manifest = {"version": 3, "projects": self.projects, "samples": self.samples}
        self.manifest.write_bytes(json.dumps(manifest).encode("utf-8"))

    def extract(self):
        self.write_manifest()
        return extract_candidates(self.manifest)

    def find(self, result, qualified_name):
        for candidate in result["candidates"]:
            if candidate.qualified_name == qualified_name:
                return candidate
        self.fail(f"no candidate named {qualified_name}")

    # --- extraction correctness -------------------------------------------------

    def test_docstring_becomes_prompt_and_is_excluded_from_target(self):
        self.add_file("alpha", "a", (
            "def greet(name):\n"
            "    \"\"\"Return a friendly greeting for the given name.\"\"\"\n"
            "    message = f\"hello {name}\"\n"
            "    return message\n"
        ))
        result = self.extract()
        candidate = self.find(result, "greet")
        self.assertEqual(candidate.prompt_source, "docstring")
        self.assertIn("def greet(name):", candidate.prompt)
        self.assertIn("friendly greeting", candidate.prompt)
        self.assertNotIn("friendly greeting", candidate.target)
        self.assertIn("message = f\"hello {name}\"", candidate.target)
        self.assertIn("return message", candidate.target)

    def test_leading_comment_used_when_no_docstring(self):
        self.add_file("alpha", "a", (
            "# Compute the area of a rectangle.\n"
            "def area(width, height):\n"
            "    return width * height\n"
        ))
        result = self.extract()
        candidate = self.find(result, "area")
        self.assertEqual(candidate.prompt_source, "comment")
        self.assertIn("Compute the area", candidate.prompt)
        self.assertIn("def area(width, height):", candidate.prompt)
        self.assertEqual(candidate.target.strip(), "return width * height")

    def test_no_descriptive_prompt_is_rejected(self):
        self.add_file("alpha", "a", "def mystery(x):\n    return x + 1\n")
        result = self.extract()
        names = {c.qualified_name for c in result["candidates"]}
        self.assertNotIn("mystery", names)
        self.assertEqual(result["report"]["rejection_reasons"]["no_descriptive_prompt"], 1)

    def test_trivial_bodies_are_rejected(self):
        self.add_file("alpha", "a", (
            "def stub_pass(x):\n"
            "    \"\"\"Do nothing yet.\"\"\"\n"
            "    pass\n\n"
            "def stub_raise(x):\n"
            "    \"\"\"Not implemented yet.\"\"\"\n"
            "    raise NotImplementedError\n\n"
            "def docstring_only(x):\n"
            "    \"\"\"Just a docstring, no other body.\"\"\"\n"
        ))
        result = self.extract()
        names = {c.qualified_name for c in result["candidates"]}
        self.assertFalse(names & {"stub_pass", "stub_raise", "docstring_only"})
        self.assertEqual(result["report"]["rejection_reasons"]["trivial_target"], 3)

    def test_oneliner_def_is_rejected(self):
        self.add_file("alpha", "a", "def one(x): return x + 1\n")
        result = self.extract()
        self.assertEqual(result["report"]["rejection_reasons"]["oneliner_def_unsupported"], 1)

    def test_unsafe_content_is_rejected(self):
        self.add_file("alpha", "a", (
            "def configure():\n"
            "    \"\"\"Configure the client with a hardcoded token.\"\"\"\n"
            "    token = \"sk-ant-abcdefghijklmnopqrstuvwxyz0123456789\"\n"
            "    return token\n"
        ))
        result = self.extract()
        names = {c.qualified_name for c in result["candidates"]}
        self.assertNotIn("configure", names)
        self.assertEqual(result["report"]["rejection_reasons"]["unsafe_content"], 1)

    def test_token_length_buckets(self):
        self.add_file("alpha", "a", (
            "def tiny(x):\n"
            "    \"\"\"Add one to x.\"\"\"\n"
            "    return x + 1\n"
        ))
        long_body = "\n".join(f"    total += {i}" for i in range(80))
        self.add_file("alpha", "b", (
            "def big(total):\n"
            "    \"\"\"Accumulate a long series of additions.\"\"\"\n"
            f"{long_body}\n"
            "    return total\n"
        ))
        result = self.extract()
        tiny = self.find(result, "tiny")
        big = self.find(result, "big")
        self.assertTrue(tiny.fits[128])
        self.assertFalse(big.fits[128])
        self.assertFalse(big.fits[256])
        self.assertFalse(big.fits[1024])
        self.assertGreaterEqual(result["report"]["fits"][128], 1)
        self.assertEqual(sum(result["report"]["length_buckets"].values()), result["report"]["candidates_accepted"])

    def test_candidates_land_in_257_512_and_513_1024_buckets(self):
        def text_for(name, pad_len):
            return f'# Comment for {name}.\ndef {name}():\n    return "{"A" * pad_len}"\n'

        def base_tokens(name):
            prompt = f'# Comment for {name}.\ndef {name}():'
            target = '    return ""'
            return len(encode(prompt + "\n" + target, add_bos=True, add_eos=True))

        pad_mid = 257 - base_tokens("mid")
        pad_high = 513 - base_tokens("high")
        self.add_file("alpha", "a", text_for("mid", pad_mid))
        self.add_file("alpha", "b", text_for("high", pad_high))
        result = self.extract()
        mid = self.find(result, "mid")
        high = self.find(result, "high")
        self.assertTrue(256 < mid.total_tokens <= 512, mid.total_tokens)
        self.assertTrue(512 < high.total_tokens <= 1024, high.total_tokens)
        self.assertFalse(mid.fits[256])
        self.assertTrue(mid.fits[512])
        self.assertFalse(high.fits[512])
        self.assertTrue(high.fits[1024])
        self.assertEqual(result["report"]["length_buckets"].get("257-512"), 1)
        self.assertEqual(result["report"]["length_buckets"].get("513-1024"), 1)

    # --- bounds -------------------------------------------------------------

    def test_functions_per_file_are_bounded(self):
        import model_lab.extract_pairs as module
        original = module.MAX_FUNCTIONS_PER_FILE
        module.MAX_FUNCTIONS_PER_FILE = 2
        self.addCleanup(setattr, module, "MAX_FUNCTIONS_PER_FILE", original)
        text = "".join(
            f"def f{i}(x):\n    \"\"\"Describe function number {i}.\"\"\"\n    return x + {i}\n\n"
            for i in range(5)
        )
        self.add_file("alpha", "a", text)
        result = self.extract()
        # 2 filler files (1 function each) plus 2 of the 5 functions in "a" (capped).
        self.assertEqual(result["report"]["candidates_considered"], 4)

    # --- provenance -----------------------------------------------------------

    def test_provenance_is_preserved_per_candidate(self):
        self.add_file("beta", "a", (
            "# Check whether a number is prime.\n"
            "def is_prime(n):\n"
            "    return n > 1\n"
        ), source_path="lib/primes.py")
        result = self.extract()
        candidate = self.find(result, "is_prime")
        self.assertEqual(candidate.project, "beta")
        self.assertEqual(candidate.split, "validation")
        self.assertEqual(candidate.source_path, "lib/primes.py")
        self.assertEqual(candidate.repository, "https://example.invalid/owner/beta")
        self.assertEqual(candidate.commit, "a" * 40)
        self.assertEqual(candidate.rights_basis, "owner-authorized-local-use")
        self.assertIsNone(candidate.license)

    def test_class_methods_are_qualified_and_nested_functions_are_skipped(self):
        self.add_file("alpha", "a", (
            "class Box:\n"
            "    def volume(self, w, h, d):\n"
            "        \"\"\"Compute the volume of the box.\"\"\"\n"
            "        return w * h * d\n\n"
            "def outer(x):\n"
            "    \"\"\"Wrap an inner helper that should not become its own candidate.\"\"\"\n"
            "    def inner(y):\n"
            "        return y * 2\n"
            "    return inner(x)\n"
        ))
        result = self.extract()
        names = {c.qualified_name for c in result["candidates"]}
        self.assertIn("Box.volume", names)
        self.assertIn("outer", names)
        self.assertNotIn("inner", names)
        self.assertNotIn("outer.inner", names)

    # --- split safety -----------------------------------------------------------

    def test_duplicate_across_splits_is_rejected_not_silently_kept(self):
        shared = (
            "# Return the larger of two values.\n"
            "def bigger(a, b):\n"
            "    return a if a > b else b\n"
        )
        # Distinct surrounding files (so the manifest's whole-file dedup does not
        # itself reject them) that happen to contain an identical function.
        self.add_file("alpha", "a", shared + "\n# Unrelated marker one.\ndef marker_one(x):\n    return x\n")
        self.add_file("beta", "a", shared + "\n# Unrelated marker two.\ndef marker_two(x):\n    return x\n")
        result = self.extract()
        matches = [c for c in result["candidates"] if c.qualified_name == "bigger"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(result["report"]["rejection_reasons"]["duplicate_cross_split"], 1)

    def test_duplicate_within_same_split_is_rejected_separately(self):
        shared = (
            "# Return the smaller of two values.\n"
            "def smaller(a, b):\n"
            "    return a if a < b else b\n"
        )
        self.add_file("alpha", "a", shared + "\n# Unrelated marker one.\ndef marker_one(x):\n    return x\n")
        self.add_file("alpha", "b", shared + "\n# Unrelated marker two.\ndef marker_two(x):\n    return x\n")
        result = self.extract()
        matches = [c for c in result["candidates"] if c.qualified_name == "smaller"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(result["report"]["rejection_reasons"]["duplicate_same_split"], 1)

    def test_accepted_candidate_split_always_matches_its_project(self):
        self.add_file("alpha", "a", "# Add two numbers.\ndef add(a, b):\n    return a + b\n")
        self.add_file("beta", "a", "# Subtract two numbers.\ndef sub(a, b):\n    return a - b\n")
        result = self.extract()
        for candidate in result["candidates"]:
            expected_split = self.projects[candidate.project]["split"]
            self.assertEqual(candidate.split, expected_split)

    # --- never executes source ---------------------------------------------

    def test_module_level_side_effects_are_never_executed(self):
        self.add_file("alpha", "a", (
            "raise RuntimeError('this file must never be executed by the extractor')\n\n"
            "# Add two numbers safely.\n"
            "def add(a, b):\n"
            "    return a + b\n"
        ))
        result = self.extract()  # must not raise
        candidate = self.find(result, "add")
        self.assertEqual(candidate.target.strip(), "return a + b")

    def test_token_counts_use_the_project_byte_tokenizer(self):
        self.add_file("alpha", "a", "# Add one to x.\ndef inc(x):\n    return x + 1\n")
        result = self.extract()
        candidate = self.find(result, "inc")
        combined = candidate.prompt + "\n" + candidate.target
        self.assertEqual(candidate.total_tokens, len(encode(combined, add_bos=True, add_eos=True)))


if __name__ == "__main__":
    unittest.main()
