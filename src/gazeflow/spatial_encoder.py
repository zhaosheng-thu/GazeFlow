import torch.nn as nn


class SpatialEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        B, T, N, D = tokens.shape
        x = tokens.reshape(B * T, N, D)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x).reshape(B, T, N, D)
