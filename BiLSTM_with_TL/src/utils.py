from __future__ import annotations

import numpy as np
from pathlib import Path

def set_seed(seed: int):
    import random, os, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_glove_embeddings(path: str, word_vocab, emb_dim: int = 100):
    """
    Load GloVe embeddings (.txt) and align to vocab.
    Returns a torch.FloatTensor of shape (vocab_size, emb_dim).
    """
    import torch

    mat = np.random.uniform(-0.05, 0.05, size=(len(word_vocab), emb_dim)).astype(np.float32)
    found = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < emb_dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:1+emb_dim], dtype=np.float32)
            idx = word_vocab.stoi.get(word)
            if idx is not None:
                mat[idx] = vec
                found += 1
    print(f"Loaded embeddings for {found} / {len(word_vocab)} tokens from {path}")
    return torch.tensor(mat)


def download_glove_6B(dim: int = 100, out_dir: str = "embeddings"):
    import zipfile, io, requests
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = f"glove.6B.{dim}d.txt"
    z.extract(name, path=out)
    print(f"Saved to {out / name}")
    return out / name


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_glove", type=int, default=0, help="If set (e.g., 100), downloads GloVe.6B.{dim}d.txt")
    args = ap.parse_args()
    if args.download_glove:
        download_glove_6B(args.download_glove)
