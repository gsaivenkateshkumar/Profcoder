"""Pre-training corpus audit: synthetic findings, false positives, bounds, and duplicates.

All sensitive-looking values here are synthetic and are assembled at runtime so
that no complete credential pattern appears in the repository.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_lab.audit import audit, scan_text
from model_lab.prepare import MAX_FILE_BYTES

REPO_ROOT = Path(__file__).resolve().parents[1]

# Synthetic values, split so the source never contains a whole token pattern.
FAKE_GITHUB = "gh" + "p_" + "Zq7Lm2Xv9Rt4" * 3
FAKE_GROQ = "gs" + "k_" + "Q4w8E2r6T1y5U3i7O9p0"
FAKE_KEY_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"
FAKE_PASSWORD = "Tr0ub4dor" + "-synthetic"
FAKE_CODE = "q9" + "Zk2"
FAKE_EMAIL = "jordan.sample" + "@" + "mailhost.co"
FAKE_CARD = "4111 1111" + " 1111 1111"
FAKE_MOBILE = "98765" + "43210"
FAKE_ENTROPY = "Xk9fQ2mLp8Rz4TvB7nWc3YhJ" + "5sDe1Ua6"
FAKE_HOME = "C:\\\\Users\\\\" + "jordan" + "\\\\projects"
FAKE_ID = "2345 6789" + " 0123"

SENSITIVE = "\n".join([
    f'TOKEN = "{FAKE_GITHUB}"',
    f"client = Client(api_key='{FAKE_GROQ}')",
    f'KEY = """{FAKE_KEY_HEADER}',
    '"""',
    f'db = connect(host="db", user="app", password="{FAKE_PASSWORD}")',
    f'url = "postgres://app:{FAKE_PASSWORD}@db.internal/app"',
    'code = input("Secret code: ")',
    f'if code == "{FAKE_CODE}":',
    "    pass",
    f'contact = "{FAKE_EMAIL}"',
    f'card = "{FAKE_CARD}"',
    f'national = "{FAKE_ID}"',
    f'mobile = "{FAKE_MOBILE}"',
    'phone = "+1 415-555-0100"',
    'server = "8.8.4.4"',
    f'home = "{FAKE_HOME}"',
    f'blob = "{FAKE_ENTROPY}"',
    "",
])
SENSITIVE_VALUES = (
    FAKE_GITHUB, FAKE_GROQ, FAKE_PASSWORD, FAKE_CODE, FAKE_EMAIL, FAKE_CARD, FAKE_MOBILE,
    FAKE_ENTROPY, FAKE_ID, "jordan", "415-555-0100", "8.8.4.4", "Client(", "connect(",
)

BENIGN = "\n".join([
    "import os, secrets",
    'API_KEY = os.environ.get("API_KEY", "")',
    'password = "<redacted>"',
    'password = ""',
    'token = f"{prefix}-{suffix}"',
    "run_id = 'run-' + secrets.token_hex(8)",
    'tokenizer = "byte-v1"',
    'prompt = input("Enter password: ")',
    'password_field = "password"',
    'name = input("Your name: ")',
    'if name == "alice1":',
    "    pass",
    'for token in ("alpha", "beta"):',
    "    pass",
    'owner = "fixture@example.invalid"',
    'docs = "someone@example.com"',
    'host = "127.0.0.1"',
    'lan = "192.168.1.20"',
    'digest = "3f786850e387550fdab836ed7e6dc881de23001b3f786850e387550fdab836ed"',
    'stamp = "2026-09-27T12:30:00.123Z"',
    'uuid = "123e4567-e89b-12d3-a456-426614174000"',
    'shared = "C:\\\\Users\\\\Public\\\\Documents"',
    "count = 1234567890123",
    'version = "1.2.3"',
    "",
])

MODULE_A = "\n".join(
    [f"def step_{i}(values, scale):\n    total = 0\n    for value in values:\n"
     f"        total += value * scale + {i}\n    return total\n" for i in range(12)]
)
UNRELATED = "\n".join(
    [f"class Shape{i}:\n    def area(self, w, h):\n        return w * h - {i}\n\n"
     f"    def label(self):\n        return 'shape-{i}'\n" for i in range(12)]
)


class ScanTextTests(unittest.TestCase):
    def test_detects_synthetic_credentials_and_personal_data(self):
        self.assertEqual(dict(scan_text(SENSITIVE)), {
            "credential:known-token-format": 2,
            "credential:private-key-block": 1,
            "credential:hardcoded-assignment": 3,     # TOKEN, api_key=, password=
            "credential:url-with-password": 1,
            "credential:hardcoded-secret-comparison": 1,
            "personal:email-address": 1,
            "personal:payment-card-like": 1,
            "personal:national-id-like": 1,
            "personal:phone-number": 2,
            "personal:public-ip-address": 1,
            "personal:home-directory-path": 1,
            "review:high-entropy-literal": 1,
        })

    def test_common_code_patterns_are_not_flagged(self):
        self.assertEqual(dict(scan_text(BENIGN)), {})

    def test_passphrases_with_spaces_count_but_prompts_do_not(self):
        phrase = "violet " + "harbor lantern"
        self.assertEqual(
            dict(scan_text(f'p = input("Enter password: ")\nif p == "{phrase}":\n')),
            {"credential:hardcoded-secret-comparison": 1},
        )
        self.assertEqual(
            dict(scan_text(f'db_password = "{phrase}"\n')),
            {"credential:hardcoded-assignment": 1},
        )
        prompts = (
            'password = "Enter your password:"\n'
            'secret = "Please type the code"\n'
            'token = "Bearer "\n'
            'passcode = "Code --> "\n'
        )
        self.assertEqual(dict(scan_text(prompts)), {})

    def test_comparison_needs_a_secret_prompt(self):
        self.assertEqual(dict(scan_text('n = input("Name: ")\nif n == "q9Zk2x":\n')), {})
        self.assertEqual(
            dict(scan_text('p = getpass.getpass()\nif p == "q9Zk2x":\n')),
            {"credential:hardcoded-secret-comparison": 1},
        )

    def test_overlong_lines_are_counted_not_scanned(self):
        line = "x = '" + "a" * 20_000 + f"' # {FAKE_GITHUB}"
        self.assertEqual(dict(scan_text(line)), {"review:overlong-line": 1})


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = []

    def add(self, project, split, name, text):
        data = text.encode("utf-8")
        path = self.root / "samples" / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.items.append({
            "path": f"samples/{project}/{name}", "project": project, "split": split,
            "sha256": hashlib.sha256(data).hexdigest(), "license": "CC0-1.0",
            "source": "Synthetic audit fixture", "rights_reviewed": True,
        })
        return path

    def manifest(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"version": 2, "samples": self.items}), encoding="utf-8")
        return path

    def test_report_lists_paths_categories_counts_but_no_values(self):
        self.add("alpha", "train", "settings.txt", SENSITIVE)
        self.add("beta", "validation", "clean.txt", BENIGN)
        report = audit(self.manifest())
        self.assertTrue(report["review_required"])
        self.assertEqual(report["files_scanned"], 2)
        self.assertEqual(report["files_flagged"], 1)
        self.assertEqual({f["path"] for f in report["findings"]}, {"samples/alpha/settings.txt"})
        for finding in report["findings"]:
            self.assertEqual(set(finding), {"path", "project", "category", "count"})
        serialized = json.dumps(report)
        for value in SENSITIVE_VALUES:
            self.assertNotIn(value, serialized)

    def test_unlisted_files_are_never_scanned(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        self.add("beta", "validation", "b.txt", UNRELATED)
        (self.root / "samples/alpha/unlisted.txt").write_text(SENSITIVE, encoding="utf-8")
        (self.root / "loose.txt").write_text(SENSITIVE, encoding="utf-8")
        report = audit(self.manifest())
        self.assertEqual(report["files_scanned"], 2)
        self.assertEqual(report["findings"], [])
        self.assertFalse(report["review_required"])
        self.assertIn("not proof", report["note"])

    def test_cross_project_near_duplicate_is_flagged_for_review_only(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        self.add("alpha", "train", "a-copy.txt", MODULE_A + "\n# same project edit\n")
        self.add("beta", "validation", "b.txt", "import math\n\n" + MODULE_A + "\nVALUE = 3\n")
        self.add("gamma", "train", "c.txt", UNRELATED)
        pairs = audit(self.manifest())["near_duplicates"]["pairs"]
        flagged = {tuple(sorted(p["files"])) for p in pairs}
        self.assertEqual(flagged, {
            ("samples/alpha/a-copy.txt", "samples/beta/b.txt"),
            ("samples/alpha/a.txt", "samples/beta/b.txt"),
        })
        for pair in pairs:
            self.assertTrue(pair["crosses_splits"])
            self.assertEqual(pair["status"], "needs-manual-review")
            self.assertGreaterEqual(pair["estimated_containment"], 0.7)

    def test_exact_duplicates_remain_preparation_errors(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        self.add("beta", "validation", "b.txt", MODULE_A.replace("\n", "\r\n"))
        self.items[1]["sha256"] = self.items[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(self.manifest())

    def test_refuses_unsafe_paths_links_and_oversized_files(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        self.add("beta", "validation", "b.txt", UNRELATED)
        manifest = self.manifest()
        for bad in ("samples/beta/../alpha/a.txt", "../outside.txt", "samples/beta"):
            with self.subTest(path=bad):
                self.items[1]["path"] = bad
                with self.assertRaisesRegex(ValueError, "sample must be"):
                    audit(self.manifest())
        self.items[1]["path"] = "samples/beta/b.txt"
        big = "x = 1\n" * (MAX_FILE_BYTES // 6 + 1)
        self.items.pop()
        self.add("beta", "validation", "b.txt", big)
        with self.assertRaisesRegex(ValueError, "size limit"):
            audit(self.manifest())
        with self.assertRaises(ValueError):
            audit(self.root)  # a directory is not a manifest
        self.assertTrue(manifest.exists())

    def test_hard_linked_samples_are_refused(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        target = self.add("beta", "validation", "b.txt", UNRELATED)
        os.link(target, self.root / "elsewhere.txt")
        with self.assertRaisesRegex(ValueError, "single-link"):
            audit(self.manifest())

    def test_linked_samples_are_refused(self):
        self.add("alpha", "train", "a.txt", MODULE_A)
        target = self.add("beta", "validation", "b.txt", UNRELATED)
        target.unlink()
        try:
            os.symlink(self.root / "samples/alpha/a.txt", target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this Windows configuration")
        with self.assertRaisesRegex(ValueError, "linked|single-link"):
            audit(self.manifest())

    def test_cli_exit_codes_and_refusals_never_echo_content(self):
        command = [sys.executable, "-m", "model_lab.audit", "--manifest"]
        self.add("alpha", "train", "a.txt", MODULE_A)
        self.add("beta", "validation", "b.txt", UNRELATED)
        clean = subprocess.run(command + [str(self.manifest())], cwd=REPO_ROOT,
                               capture_output=True, text=True)
        self.assertEqual(clean.returncode, 0, clean.stderr)

        self.items.pop()
        self.add("beta", "validation", "b.txt", SENSITIVE)
        report = self.root / "report.json"
        flagged = subprocess.run(command + [str(self.manifest()), "--report", str(report)],
                                 cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(flagged.returncode, 1)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8")), json.loads(flagged.stdout))
        again = subprocess.run(command + [str(self.manifest()), "--report", str(report)],
                               cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)

        (self.root / "samples/beta/b.txt").write_text(SENSITIVE + "# tampered\n", encoding="utf-8")
        refused = subprocess.run(command + [str(self.manifest())], cwd=REPO_ROOT,
                                 capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("SHA-256 mismatch", refused.stderr)
        for value in SENSITIVE_VALUES:
            self.assertNotIn(value, refused.stdout + refused.stderr + flagged.stdout)


if __name__ == "__main__":
    unittest.main()
