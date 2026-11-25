"""SASRec with causal attention, external-catalog padding, CE or SCE."""

import torch
from torch import nn

from recsys.data.batches import sequence_batches
from recsys.losses.sequence import build_sequence_loss


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding, causal):
        q = self.attention_norm(x)
        attended, _ = self.attention(
            q, x, x, key_padding_mask=padding, attn_mask=causal, need_weights=False
        )
        x = self.ffn_norm(q + attended)
        return (x + self.ffn(x)).masked_fill(padding.unsqueeze(-1), 0)


class SASRec(nn.Module):
    def __init__(self, history, config):
        super().__init__()
        self.pad_id = history.n_items
        self.length = history.max_seq_length or int(
            max(len(history.sequence(u)) for u in history.users)
        )
        dim = config["embedding_dim"]
        if dim % config["heads"]:
            raise ValueError("embedding_dim must be divisible by heads")
        self.next_item_loss = build_sequence_loss(config["loss"])
        self.item_embedding = nn.Embedding(history.n_items + 1, dim, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(self.length, dim)
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(dim, config["heads"], config["dropout"])
                for _ in range(config["blocks"])
            ]
        )
        self.final_norm = nn.LayerNorm(dim, eps=1e-8)
        self.register_buffer(
            "active_items", torch.from_numpy(history.items.copy()), persistent=False
        )
        target_map = torch.full((history.n_items,), -1, dtype=torch.long)
        target_map[self.active_items] = torch.arange(len(self.active_items))
        self.register_buffer("target_map", target_map, persistent=False)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_id].zero_()

    def forward(self, sequences):
        padding = sequences == self.pad_id
        if sequences.shape[1] > self.length or padding.all(dim=1).any():
            raise ValueError("Sequences must fit max length and have at least one real event")
        x = self.item_embedding(sequences) * self.item_embedding.embedding_dim**0.5
        x = self.dropout(
            x + self.position_embedding(torch.arange(sequences.shape[1], device=x.device))
        )
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        causal = torch.ones(
            sequences.shape[1], sequences.shape[1], device=x.device, dtype=torch.bool
        ).triu(1)
        for block in self.blocks:
            x = block(x, padding, causal)
        return self.final_norm(x).masked_fill(padding.unsqueeze(-1), 0)

    @property
    def max_position_length(self):
        """Keep the positional embedding length in checkpoint compatibility checks."""
        return self.length

    def training_batches(self, history, batch_size, target_start=None):
        return sequence_batches(
            history, batch_size, self.item_embedding.weight.device, target_start
        )

    def loss(self, sequences, targets):
        hidden = self(sequences)
        valid = targets != self.pad_id
        if not valid.any():
            raise ValueError("Batch has no supervised targets")
        # Loss classes are dense positions in the active catalog, while targets use dataset IDs.
        x, target = hidden[valid], self.target_map[targets[valid]]
        if (target < 0).any():
            raise ValueError("Target is outside effective training catalog")
        weights = self.item_embedding.weight[self.active_items]
        return self.next_item_loss(x, target, weights)

    def prepare_scoring(self, history):
        self.eval()

    def encode(self, users, history):
        device = self.item_embedding.weight.device
        sequences = torch.as_tensor(history.padded(users, self.length), device=device)
        lengths = (sequences != self.pad_id).sum(1)
        hidden = self(sequences)
        return hidden[torch.arange(len(users), device=device), lengths - 1]

    def score(self, encoded, items):
        items = torch.as_tensor(items, device=encoded.device)
        return encoded @ self.item_embedding(items).T
