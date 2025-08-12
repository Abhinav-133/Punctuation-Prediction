from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

from .constants import END_LABELS

PUNCT_RE = re.compile(r"([,.;:!?])\s*(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONMARK|SEMICOLON|COLON)\b", re.IGNORECASE)
SIL_RE = re.compile(r"<\s*sil\s*=\s*[\d.]+>", re.IGNORECASE)


def collapse_spaced_letters(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i
            buf = []
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                buf.append(tokens[j])
                j += 1
            if len(buf) >= 3:
                out.append("".join(buf))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return out


def parse_raw_lines(text: str) -> List[Tuple[str, str]]:
    text = SIL_RE.sub(" ", text)
    tokens = text.split()
    tokens = collapse_spaced_letters(tokens)

    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        m = PUNCT_RE.fullmatch(tok)
        if m:
            label = m.group(2).upper()
            if pairs:
                t, _ = pairs[-1]
                pairs[-1] = (t, label)
            i += 1
            continue
        pairs.append((tok, "O"))
        i += 1
    return pairs


def remap_labels(pairs: List[Tuple[str, str]], mapping: Dict[str, str]) -> List[Tuple[str, str]]:
    if not mapping:
        return pairs
    out = []
    for tok, lab in pairs:
        out.append((tok, mapping.get(lab, lab)))
    return out


def chunk_pairs(pairs: List[Tuple[str, str]], max_len: int = 120):
    chunks = []
    cur_tokens = []
    cur_labels = []
    for tok, lab in pairs:
        cur_tokens.append(tok)
        cur_labels.append(lab)
        if lab in END_LABELS or len(cur_tokens) >= max_len:
            chunks.append({"tokens": cur_tokens, "labels": cur_labels})
            cur_tokens, cur_labels = [], []
    if cur_tokens:
        chunks.append({"tokens": cur_tokens, "labels": cur_labels})
    return chunks


def split_train_val(items, val_ratio: float = 0.1):
    n = len(items)
    val_n = max(1, int(n * val_ratio))
    return items[val_n:], items[:val_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--suppress_rare", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    pairs = parse_raw_lines(raw)
    mapping = {}
    if args.suppress_rare:
        mapping = {"EXCLAMATIONMARK": "PERIOD", "QUESTIONMARK": "PERIOD"}
    pairs = remap_labels(pairs, mapping)
    chunks = chunk_pairs(pairs, max_len=args.max_len)
    train_items, val_items = split_train_val(chunks, val_ratio=args.val_ratio)

    for name, data in [("train.jsonl", train_items), ("val.jsonl", val_items)]:
        with open(outdir / name, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_items)} train and {len(val_items)} val items to {outdir}")
