from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .dataset import PunctDataset, pad_collate
from .vocab import Vocab
from .model import BiLSTMPunct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed dir containing *.jsonl")
    ap.add_argument("--ckpt", type=str, required=True, help="path to runs/bilstm/best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lower", type=bool, default=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / f"{args.split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found. Did you preprocess {args.split}?")

    # Load vocabs
    runs_dir = Path("runs/bilstm")
    word_vocab = Vocab.load(runs_dir / "word_vocab.json")
    lab_vocab = Vocab.load(runs_dir / "lab_vocab.json")

    ds = PunctDataset(jsonl_path, word_vocab, lab_vocab, lowercase=args.lower)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPunct(
        vocab_size=len(word_vocab),
        num_labels=len(lab_vocab),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for ids, labs, lengths in loader:
            ids, labs, lengths = ids.to(device), labs.to(device), lengths.to(device)
            logits = model(ids, lengths)
            preds = logits.argmax(-1)
            for i in range(ids.size(0)):
                L = lengths[i].item()
                y_true.extend(labs[i, :L].tolist())
                y_pred.extend(preds[i, :L].tolist())

    labels = list(range(1, len(lab_vocab.itos)))   # skip PAD
    names = lab_vocab.itos[1:]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0, digits=4))


if __name__ == "__main__":
    main()
