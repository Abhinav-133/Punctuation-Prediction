from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMPunct(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        emb_dim: int = 100,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        embeddings: torch.Tensor | None = None,
        freeze_emb: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if embeddings is not None:
            if embeddings.size(0) != vocab_size or embeddings.size(1) != emb_dim:
                raise ValueError("Embeddings shape mismatch with vocab/emb_dim")
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = not freeze_emb

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
