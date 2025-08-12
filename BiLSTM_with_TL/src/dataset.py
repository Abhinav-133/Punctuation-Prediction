from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from .vocab import Vocab


class PunctDataset(Dataset):
    def __init__(self, jsonl_path: Path, word_vocab: Vocab, lab_vocab: Vocab, lowercase: bool = True):
        self.items: List[Dict] = []
        self.word_vocab = word_vocab
        self.lab_vocab = lab_vocab
        self.lowercase = lowercase
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                toks = obj["tokens"]
                labs = obj["labels"]
                if lowercase:
                    toks = [t.lower() for t in toks]
                self.items.append({"tokens": toks, "labels": labs})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        ids = self.word_vocab.encode(ex["tokens"])
        lab_ids = [self.lab_vocab.stoi[l] for l in ex["labels"]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lab_ids, dtype=torch.long)


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    ids, labs = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    ids_padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    labs_padded = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=0)  # PAD idx 0
    return ids_padded, labs_padded, lengths


def build_word_vocab(jsonl_paths: List[Path], lowercase: bool = True, min_freq: int = 1) -> Vocab:
    from collections import Counter
    counter = Counter()
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                toks = obj["tokens"]
                if lowercase:
                    toks = [t.lower() for t in toks]
                counter.update(toks)
    # feed counts into Vocab.build
    return Vocab.build((t for t, c in counter.items() for _ in range(c)), min_freq=min_freq)
