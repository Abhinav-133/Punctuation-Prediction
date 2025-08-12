from __future__ import annotations

import argparse
from pathlib import Path
import json

from .preprocess import parse_raw_lines, remap_labels, chunk_pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--suppress_rare", action="store_true")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    pairs = parse_raw_lines(raw)
    mapping = {}
    if args.suppress_rare:
        mapping = {"EXCLAMATIONMARK": "PERIOD", "QUESTIONMARK": "PERIOD"}
    pairs = remap_labels(pairs, mapping)
    chunks = chunk_pairs(pairs, max_len=args.max_len)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for ex in chunks:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} examples to {outp}")
    
if __name__ == "__main__":
    main()
