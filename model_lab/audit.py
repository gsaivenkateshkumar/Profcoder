"""Offline pre-training audit of the files an explicit corpus manifest selects.

Reads only the files listed in a manifest, through the same path, link, size,
UTF-8, SHA-256, and exact-duplicate checks as preparation. Reports file paths,
finding categories, and counts; never matched values or surrounding text.
Heuristic: a report with no findings is not proof that a corpus is safe.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from model_lab.prepare import SelectedSample, read_selection

MAX_LINE_CHARS = 10_000
SHINGLE_TOKENS = 8
SAMPLE_MODULUS = 4          # keep 1 in 4 shingle hashes, bounding memory and time
MAX_POSTING_FILES = 25      # shingles in more files are treated as common boilerplate
MIN_SHARED_SHINGLES = 5
JACCARD_THRESHOLD = 0.5
CONTAINMENT_THRESHOLD = 0.7
MAX_REPORTED_PAIRS = 200

STRING_LITERAL = re.compile(r"""(?:[rRbBuUfF]{0,2})("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')""")
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
CREDENTIAL_NAME = re.compile(
    r"(?:^|_)(?:password|passwd|pwd|pass|passcode|secret|token|api_?key|apikey|access_?key|"
    r"secret_?key|private_?key|client_?secret|credentials?)$"
)
ASSIGNMENT = re.compile(
    rf"""(?P<name>{IDENTIFIER})["']?\s*(?::\s*[\w.\[\], |]+?\s*)?(?:==|=|:)\s*"""
    r"""(?:[rRbBuU]{0,2})(?P<quote>["'])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"""
)
INPUT_ASSIGNMENT = re.compile(
    rf"""(?P<name>{IDENTIFIER})\s*=\s*(?:getpass\.)?(?:input|getpass)\s*\((?P<args>[^)]*)\)"""
)
PROMPT_WORDS = re.compile(r"(?i)pass|secret|pin\b|code|token")
KNOWN_TOKENS = re.compile(
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}|"
    r"xox[abprs]-[A-Za-z0-9-]{10,}|gsk_[A-Za-z0-9]{20,}|sk-(?:ant-)?[A-Za-z0-9_-]{20,}|"
    r"AIza[0-9A-Za-z_-]{35}|[sr]k_live_[0-9A-Za-z]{20,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY(?: BLOCK)?-----")
URL_CREDENTIALS = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@'\"]+:(?P<secret>[^/\s@'\"]+)@")
EMAIL = re.compile(r"(?<![\w.+:/%-])[A-Za-z0-9._%+-]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})(?![\w-])")
SAFE_EMAIL_DOMAIN = re.compile(r"(?i)(?:^|\.)(?:example\.(?:com|org|net)|example|invalid|test|localhost|local)$")
PHONE = re.compile(
    r"(?<![\w+])\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?![\w])"
    r"|(?<![\w-])\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])"
)
BARE_MOBILE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HOME_PATH = re.compile(r"(?i)(?:\b[a-z]:(?:\\{1,2}|/)users(?:\\{1,2}|/)|/home/|/users/)(?P<name>[^\\/\s'\"<>$%{}]+)")
GENERIC_HOME_NAMES = {"public", "default", "all users", "user", "username", "name", "runner", "you", "me"}
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Standalone groups only, so a card number's first 12 digits are not also counted here.
NATIONAL_ID = re.compile(
    r"(?<!\d)(?<!\d[ -])(?:[2-9]\d{3}[ -]\d{4}[ -]\d{4}|\d{3}-\d{2}-\d{4})(?![ -]?\d)"
)
HIGH_ENTROPY = re.compile(r"[A-Za-z0-9+/=_-]{24,512}")
CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")


def _placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return (
        len(lowered) < 4
        # Prompt or label text, not a secret; passphrases with spaces still count.
        or lowered.endswith((":", "?", "->"))
        or lowered.startswith(("enter ", "please "))
        or lowered in {"bearer", "basic"}
        or (lowered.startswith("<") and lowered.endswith(">"))
        or "{" in lowered or "%(" in lowered or lowered.startswith("$")
        or len(set(lowered)) == 1
        or bool(CREDENTIAL_NAME.search(lowered.replace("-", "_")))
        or lowered in {"redacted", "placeholder", "example", "dummy", "none", "null", "secret_here"}
        or lowered.startswith(("your_", "your-", "env:", "os.environ"))
    )


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values())


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char) * (2 if index % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def scan_text(text: str) -> Counter:
    """Return counts per finding category; the matched text is never retained."""
    found: Counter = Counter()
    secret_inputs: set[str] = set()
    for line in text.split("\n"):
        if len(line) > MAX_LINE_CHARS:
            found["review:overlong-line"] += 1
            continue
        literals = [m.group(1)[1:-1] for m in STRING_LITERAL.finditer(line)]
        found["credential:private-key-block"] += len(PRIVATE_KEY.findall(line))
        found["credential:known-token-format"] += len(KNOWN_TOKENS.findall(line))
        found["credential:url-with-password"] += sum(
            not _placeholder(m.group("secret")) for m in URL_CREDENTIALS.finditer(line)
        )
        for match in INPUT_ASSIGNMENT.finditer(line):
            if PROMPT_WORDS.search(match.group("args")) or "getpass" in match.group(0):
                secret_inputs.add(match.group("name"))
        for match in ASSIGNMENT.finditer(line):
            name, value = match.group("name"), match.group("value")
            if _placeholder(value):
                continue
            if CREDENTIAL_NAME.search(_snake(name)):
                found["credential:hardcoded-assignment"] += 1
            elif name in secret_inputs and "==" in match.group(0):
                found["credential:hardcoded-secret-comparison"] += 1
        for literal in literals:
            for candidate in HIGH_ENTROPY.findall(literal):
                if (
                    any(c.isupper() for c in candidate) and any(c.islower() for c in candidate)
                    and any(c.isdigit() for c in candidate) and _entropy(candidate) >= 4.3
                    and not KNOWN_TOKENS.search(candidate)
                ):
                    found["review:high-entropy-literal"] += 1
            found["personal:phone-number"] += len(BARE_MOBILE.findall(literal))
            for match in CARD.finditer(literal):
                digits = re.sub(r"\D", "", match.group(0))
                if 13 <= len(digits) <= 19 and len(set(digits)) > 1 and _luhn(digits):
                    found["personal:payment-card-like"] += 1
            found["personal:national-id-like"] += len(NATIONAL_ID.findall(literal))
        found["personal:email-address"] += sum(
            not SAFE_EMAIL_DOMAIN.search(m.group("domain")) for m in EMAIL.finditer(line)
        )
        found["personal:phone-number"] += len(PHONE.findall(line))
        for match in IPV4.finditer(line):
            try:
                address = ipaddress.IPv4Address(match.group(0))
            except ValueError:
                continue
            if address.is_global:
                found["personal:public-ip-address"] += 1
        found["personal:home-directory-path"] += sum(
            m.group("name").lower() not in GENERIC_HOME_NAMES for m in HOME_PATH.finditer(line)
        )
    return +found  # drop zero counts


def _sampled_shingles(text: str) -> set[int]:
    tokens = [t.lower() if t[0].isalpha() or t[0] == "_" else ("0" if t.isdigit() else t)
              for t in CODE_TOKEN.findall(text)]
    sampled = set()
    for start in range(len(tokens) - SHINGLE_TOKENS + 1):
        shingle = "\x1f".join(tokens[start : start + SHINGLE_TOKENS]).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        if value % SAMPLE_MODULUS == 0:
            sampled.add(value)
    return sampled


def near_duplicates(samples: list[SelectedSample]) -> dict[str, object]:
    """Flag cross-project pairs with high shingle overlap for manual review only."""
    group = [s.project if s.project is not None else f"split:{s.split}" for s in samples]
    shingles = [_sampled_shingles(s.data.decode("utf-8")) for s in samples]
    postings: dict[int, list[int]] = defaultdict(list)
    for index, values in enumerate(shingles):
        for value in values:
            postings[value].append(index)
    shared: Counter = Counter()
    common = 0
    for files in postings.values():
        if len(files) > MAX_POSTING_FILES:
            common += 1
            continue
        for i, a in enumerate(files):
            for b in files[i + 1 :]:
                if group[a] != group[b]:
                    shared[(a, b)] += 1
    pairs = []
    for (a, b), count in shared.items():
        if count < MIN_SHARED_SHINGLES:
            continue
        jaccard = count / len(shingles[a] | shingles[b])
        containment = count / min(len(shingles[a]), len(shingles[b]))
        if jaccard >= JACCARD_THRESHOLD or containment >= CONTAINMENT_THRESHOLD:
            pairs.append({
                "files": [samples[a].path, samples[b].path],
                "projects": [group[a], group[b]],
                "crosses_splits": samples[a].split != samples[b].split,
                "estimated_jaccard": round(jaccard, 3),
                "estimated_containment": round(containment, 3),
                "status": "needs-manual-review",
            })
    pairs.sort(key=lambda p: (-p["estimated_containment"], p["files"]))
    return {
        "method": (
            f"{SHINGLE_TOKENS}-token shingles, 1/{SAMPLE_MODULUS} hash-sampled; "
            f"flag if >= {MIN_SHARED_SHINGLES} shared and Jaccard >= {JACCARD_THRESHOLD} "
            f"or containment >= {CONTAINMENT_THRESHOLD}"
        ),
        "files_too_short_to_compare": sum(len(s) < MIN_SHARED_SHINGLES for s in shingles),
        "common_shingles_ignored": common,
        "pairs": pairs[:MAX_REPORTED_PAIRS],
        "pairs_truncated": len(pairs) > MAX_REPORTED_PAIRS,
    }


def audit(manifest_path: Path) -> dict[str, object]:
    selection = read_selection(manifest_path)
    findings = []
    totals: Counter = Counter()
    for sample in selection.samples:
        counts = scan_text(sample.data.decode("utf-8"))
        totals.update(counts)
        for category, count in sorted(counts.items()):
            findings.append({
                "path": sample.path, "project": sample.project,
                "category": category, "count": count,
            })
    duplicates = near_duplicates(selection.samples)
    flagged = len({f["path"] for f in findings})
    return {
        "manifest_sha256": hashlib.sha256(
            selection.manifest_bytes.replace(b"\r\n", b"\n")
        ).hexdigest(),
        "files_scanned": len(selection.samples),
        "bytes_scanned": selection.total_bytes,
        "files_flagged": flagged,
        "category_totals": dict(sorted(totals.items())),
        "findings": findings,
        "near_duplicates": duplicates,
        "review_required": bool(findings or duplicates["pairs"]),
        "note": "Heuristic scan; no findings is not proof that the corpus is safe.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="write JSON here; never overwrites")
    args = parser.parse_args()
    try:
        result = audit(args.manifest)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        # Messages come from fixed validation strings, never from file content.
        print(f"audit refused: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    text = json.dumps(result, indent=2) + "\n"
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as target:
            target.write(text)
    print(text, end="")
    raise SystemExit(1 if result["review_required"] else 0)


if __name__ == "__main__":
    main()
