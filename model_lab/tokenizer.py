"""Version 1: reversible UTF-8 byte tokenizer with fixed special-token IDs."""

from collections.abc import Iterable

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
TOKENIZER_VERSION = "byte-v1"


def encode(text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    ids = list(text.encode("utf-8"))
    if add_bos:
        ids.insert(0, BOS_ID)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode(ids: Iterable[int], *, skip_special: bool = False) -> str:
    raw = bytearray()
    for token_id in ids:
        if type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
            raise ValueError("invalid token ID")
        if token_id >= 256:
            if not skip_special:
                raise ValueError("special token encountered; pass skip_special=True")
            continue
        raw.append(token_id)
    return raw.decode("utf-8", errors="strict")
