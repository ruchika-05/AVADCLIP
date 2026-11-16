import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class VisualTransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, attention_window):
        super().__init__()
        self.attention_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            _QuickGELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.attention_window = attention_window

    def forward(self, x):
        length = x.shape[1]
        positions = torch.arange(length, device=x.device)
        window_ids = positions // self.attention_window
        attention_mask = window_ids[:, None] != window_ids[None, :]

        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if in_features == out_features:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Conv1d(
                in_features,
                out_features,
                kernel_size=5,
                padding=2,
            )

    def forward(self, x, adjacency):
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adjacency, support)
        if isinstance(self.residual_projection, nn.Identity):
            residual = self.residual_projection(x)
        else:
            residual = self.residual_projection(x.transpose(1, 2)).transpose(1, 2)
        return output + residual

class VisualTemporalEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        max_length=768,
        num_heads=1,
        transformer_layers=2,
        attention_window=4,
    ):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if attention_window < 1:
            raise ValueError("attention_window must be positive")

        self.max_length = max_length
        self.position_embeddings = nn.Embedding(max_length, feature_dim)
        self.transformer = nn.ModuleList(
            VisualTransformerBlock(feature_dim, num_heads, attention_window)
            for _ in range(transformer_layers)
        )
        self.similarity_projection = nn.Parameter(
            torch.empty(feature_dim, feature_dim)
        )
        nn.init.xavier_uniform_(self.similarity_projection)
        hidden_dim = feature_dim // 2
        self.similarity_gc1 = GraphConvolution(feature_dim, hidden_dim)
        self.similarity_gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.distance_gc1 = GraphConvolution(feature_dim, hidden_dim)
        self.distance_gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.activation = _QuickGELU()

    def _build_adjacencies(self, x):
        projected = torch.matmul(x, self.similarity_projection)
        normalized = F.normalize(projected, p=2, dim=-1)
        similarities = torch.bmm(normalized, normalized.transpose(1, 2))
        similarity_adjacency = F.softmax(
            similarities.masked_fill(similarities <= 0.7, 0.0),
            dim=1,
        )

        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        distances = torch.abs(positions[:, None] - positions[None, :])
        distance_adjacency = torch.exp(-distances / math.e)
        distance_adjacency = distance_adjacency.unsqueeze(0).expand(
            x.shape[0], -1, -1
        )
        return similarity_adjacency, distance_adjacency

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("visual features must have shape [T, D] or [B, T, D]")
        if x.shape[-1] != self.position_embeddings.embedding_dim:
            raise ValueError(
                "visual feature dimension does not match feature_dim"
            )
        if x.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds max_length "
                f"{self.max_length}"
            )

        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embeddings(positions).unsqueeze(0)
        for layer in self.transformer:
            x = layer(x)

        similarity_adjacency, distance_adjacency = self._build_adjacencies(x)
        similarity_features = self.activation(
            self.similarity_gc1(x, similarity_adjacency)
        )
        similarity_features = self.activation(
            self.similarity_gc2(similarity_features, similarity_adjacency)
        )
        distance_features = self.activation(
            self.distance_gc1(x, distance_adjacency)
        )
        distance_features = self.activation(
            self.distance_gc2(distance_features, distance_adjacency)
        )

        x = torch.cat((similarity_features, distance_features), dim=-1)
        return self.output_projection(x)
