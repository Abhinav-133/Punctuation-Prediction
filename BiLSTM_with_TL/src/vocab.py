from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
from collections import Counter
import json
from pathlib import Path

from .constants import PAD_TOKEN, UNK_TOKEN, PUNCT_LABELS

class Vocab:
    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, tokens_iter: Iterable[str], min_freq: int = 1, specials: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN)):
        counter = Counter(tokens_iter)
        itos = list(specials)
        for tok, c in counter.most_common():
            if c >= min_freq and tok not in specials:
                itos.append(tok)
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        u = self.stoi.get(UNK_TOKEN)
        return [self.stoi.get(t, u) for t in tokens]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        itos = obj["itos"]
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


def label_vocab() -> Vocab:
    itos = PUNCT_LABELS[:]  # PAD first
    stoi = {s: i for i, s in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)
