from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import torch

from .vocab import Vocab, label_vocab
from .model import BiLSTMPunct
from .constants import LABEL_TO_CHAR


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    toks = text.strip().split()
    return [t.lower() for t in toks] if lowercase else toks


@torch.no_grad()
def predict_labels(model, word_vocab: Vocab, tokens: List[str], device) -> List[str]:
    ids = torch.tensor([word_vocab.stoi.get(t, word_vocab.stoi.get("<unk>")) for t in tokens], dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    ids, lengths = ids.to(device), lengths.to(device)
    logits = model(ids, lengths)
    preds = logits.argmax(-1).squeeze(0).tolist()
    lv = label_vocab()
    return [lv.itos[p] for p in preds]


def restore(tokens: List[str], labels: List[str]) -> str:
    out = []
    for t, lab in zip(tokens, labels):
        punct = LABEL_TO_CHAR.get(lab, "")
        if punct:
            out.append(f"{t}{punct}")
        else:
            out.append(t)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lower", type=bool, default=True)
    ap.add_argument("--text", type=str, default="", help="If provided, use this text; else read stdin.")
    args = ap.parse_args()

    runs_dir = Path("runs/bilstm")
    word_vocab = Vocab.load(runs_dir / "word_vocab.json")
    lv = label_vocab()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPunct(
        vocab_size=len(word_vocab),
        num_labels=len(lv),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    raw = args.text if args.text else sys.stdin.read()
    tokens = tokenize(raw, lowercase=args.lower)
    labels = predict_labels(model, word_vocab, tokens, device)
    print(restore(tokens, labels))


if __name__ == "__main__":
    main()
