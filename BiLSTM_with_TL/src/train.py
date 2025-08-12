from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .dataset import PunctDataset, pad_collate, build_word_vocab
from .vocab import label_vocab, Vocab
from .model import BiLSTMPunct
from .utils import load_glove_embeddings, set_seed

RUNS_DIR = Path("runs/bilstm")


def compute_class_weights(ds: PunctDataset, pad_idx: int = 0):
    counts = Counter()
    for item in ds.items:
        for l in item["labels"]:
            counts[l] += 1
    lv = label_vocab()
    weights = torch.ones(len(lv), dtype=torch.float32)
    total = sum(counts.values())
    for lab, cnt in counts.items():
        idx = lv.stoi[lab]
        weights[idx] = total / (cnt + 1e-6)
    weights[pad_idx] = 0.0  # don't learn PAD
    weights = weights / weights.mean()
    return weights


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for ids, labs, lengths in loader:
        ids, labs, lengths = ids.to(device), labs.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(ids, lengths)  # (B,T,C)
        loss = criterion(logits.view(-1, logits.size(-1)), labs.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, lab_names):
    model.eval()
    y_true, y_pred = [], []
    for ids, labs, lengths in loader:
        ids, labs, lengths = ids.to(device), labs.to(device), lengths.to(device)
        logits = model(ids, lengths)
        preds = logits.argmax(-1)
        for i in range(ids.size(0)):
            L = lengths[i].item()
            y_true.extend(labs[i, :L].tolist())
            y_pred.extend(preds[i, :L].tolist())
    labels = list(range(1, len(lab_names)))   # skip PAD (index 0)
    names = lab_names[1:]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0, digits=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed")
    ap.add_argument("--emb_path", type=str, default="", help="embeddings/glove.6B.100d.txt (optional)")
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze_emb", action="store_true")
    ap.add_argument("--lower", type=bool, default=True)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weights", choices=["none","balanced"], default="balanced")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    word_vocab = build_word_vocab([train_path, val_path], lowercase=args.lower, min_freq=args.min_freq)
    lab_vocab = label_vocab()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    Vocab(stoi=word_vocab.stoi, itos=word_vocab.itos).save(RUNS_DIR / "word_vocab.json")
    Vocab(stoi=lab_vocab.stoi, itos=lab_vocab.itos).save(RUNS_DIR / "lab_vocab.json")

    train_ds = PunctDataset(train_path, word_vocab, lab_vocab, lowercase=args.lower)
    val_ds = PunctDataset(val_path, word_vocab, lab_vocab, lowercase=args.lower)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    embeddings = None
    if args.emb_path:
        embeddings = load_glove_embeddings(args.emb_path, word_vocab, emb_dim=args.emb_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMPunct(
        vocab_size=len(word_vocab),
        num_labels=len(lab_vocab),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        pad_idx=0,
        embeddings=embeddings,
        freeze_emb=args.freeze_emb,
    ).to(device)

    if args.class_weights == "balanced":
        weights = compute_class_weights(train_ds, pad_idx=0).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep} | train_loss={train_loss:.4f}")
        evaluate(model, val_loader, device, lab_vocab.itos)
        torch.save({"model_state": model.state_dict(), "config": vars(args)}, RUNS_DIR / "best.pt")
    print(f"Saved best checkpoint at {RUNS_DIR / 'best.pt'}.")
