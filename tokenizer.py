# tokenizer.py
# Tiny but more fine-grained tokenizer:
# - keeps whitespace tokens for round-trip decode()
# - splits CJK into single characters
# - supports URL/email, numbers, contractions, emojis/symbols, punctuation

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    # CJK Unified Ideographs + Extensions (partial but practical)
    return (
        0x4E00 <= o <= 0x9FFF or
        0x3400 <= o <= 0x4DBF or
        0x20000 <= o <= 0x2A6DF or
        0x2A700 <= o <= 0x2B73F or
        0x2B740 <= o <= 0x2B81F or
        0x2B820 <= o <= 0x2CEAF or
        0xF900 <= o <= 0xFAFF or
        0x2F800 <= o <= 0x2FA1F
    )


# One pass regex for non-CJK chunks; CJK handled char-by-char in tokenize().
_TOKEN_RE = re.compile(
    r"""
    \s+                                                | # whitespace (kept)
    https?://[^\s]+                                    | # url
    www\.[^\s]+                                        | # url-ish
    [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}      | # email
    (?:[A-Za-z]+(?:'[A-Za-z]+)?)                       | # words + contractions
    (?:\d{1,3}(?:,\d{3})+(?:\.\d+)?)%?                 | # 1,234 / 1,234.56 / optional %
    (?:\d+(?:\.\d+)?)%?                                | # 12 / 12.3 / optional %
    [\u2600-\u27BF\U0001F000-\U0001FAFF]               | # emoji/symbol ranges (rough)
    [^\w\s]                                              # punctuation/symbol
    """,
    re.VERBOSE,
)


@dataclass
class Tokenizer:
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

    token_to_id: Dict[str, int] = None
    id_to_token: List[str] = None

    def __post_init__(self) -> None:
        if self.token_to_id is None or self.id_to_token is None:
            self.id_to_token = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    @property
    def pad_id(self) -> int: return self.token_to_id[self.pad_token]
    @property
    def unk_id(self) -> int: return self.token_to_id[self.unk_token]
    @property
    def bos_id(self) -> int: return self.token_to_id[self.bos_token]
    @property
    def eos_id(self) -> int: return self.token_to_id[self.eos_token]

    def tokenize(self, text: str) -> List[str]:
        # Split into runs of CJK and non-CJK; CJK -> single char tokens
        out: List[str] = []
        buf: List[str] = []

        def flush_buf() -> None:
            if not buf:
                return
            chunk = "".join(buf)
            out.extend(_TOKEN_RE.findall(chunk))
            buf.clear()

        for ch in text:
            if _is_cjk(ch):
                flush_buf()
                out.append(ch)  # single char
            else:
                buf.append(ch)
        flush_buf()
        return out

    def fit(self, texts: Iterable[str], min_freq: int = 1, max_vocab: Optional[int] = None) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))

        specials = set(self.token_to_id.keys())
        items = [(tok, c) for tok, c in counter.items() if c >= min_freq and tok not in specials]
        items.sort(key=lambda x: (-x[1], x[0]))

        if max_vocab is not None:
            keep = max(0, max_vocab - len(self.id_to_token))
            items = items[:keep]

        for tok, _ in items:
            self._add_token(tok)

    def _add_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        idx = len(self.id_to_token)
        self.id_to_token.append(tok)
        self.token_to_id[tok] = idx
        return idx

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = {self.pad_id, self.unk_id, self.bos_id, self.eos_id} if skip_special_tokens else set()
        toks: List[str] = []
        for i in ids:
            if i in specials:
                continue
            if 0 <= i < len(self.id_to_token):
                toks.append(self.id_to_token[i])
            else:
                toks.append(self.unk_token)
        return "".join(toks)

    def save(self, path: str) -> None:
        data = {
            "id_to_token": self.id_to_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
        )
        tok.id_to_token = list(data["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


def _read_text_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny fine-grained tokenizer (keeps whitespace, CJK char-level).")
    ap.add_argument("--fit", help="Text file to build vocab from (one document per line).")
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--max-vocab", type=int, default=None)
    ap.add_argument("--save", help="Where to save vocab JSON.")
    ap.add_argument("--load", help="Load vocab JSON.")
    ap.add_argument("--encode", help="Encode this text (prints list of ids).")
    ap.add_argument("--decode", help="Decode comma-separated ids (e.g. 2,10,11,3).")
    ap.add_argument("--special", action="store_true", help="Add BOS/EOS on encode.")
    args = ap.parse_args()

    tok = Tokenizer.load(args.load) if args.load else Tokenizer()

    if args.fit:
        tok.fit(_read_text_file(args.fit), min_freq=args.min_freq, max_vocab=args.max_vocab)
        if args.save:
            tok.save(args.save)

    if args.encode is not None:
        print(tok.encode(args.encode, add_special_tokens=args.special))

    if args.decode is not None:
        ids = [int(x.strip()) for x in args.decode.split(",") if x.strip()]
        print(tok.decode(ids))


if __name__ == "__main__":
    main()
