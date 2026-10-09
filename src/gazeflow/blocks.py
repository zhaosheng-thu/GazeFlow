import math

import torch
import torch.nn as nn


def make_2d_sinusoidal_pe(height: int, width: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(height, width, d_model)
    d_half = d_model // 2
    pos_h = torch.arange(height).unsqueeze(1).float()
    pos_w = torch.arange(width).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))
    pe[:, :, 0:d_half:2] = torch.sin(pos_h * div).unsqueeze(1).expand(-1, width, -1)
    pe[:, :, 1:d_half:2] = torch.cos(pos_h * div).unsqueeze(1).expand(-1, width, -1)
    pe[:, :, d_half::2] = torch.sin(pos_w * div).unsqueeze(0).expand(height, -1, -1)
    pe[:, :, d_half + 1::2] = torch.cos(pos_w * div).unsqueeze(0).expand(height, -1, -1)
    return pe


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self.cos[:T], self.sin[:T]  # position = frame index
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, T, D))


class FlowBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RoPESelfAttention(d_model, n_heads, max_seq_len)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm_task = nn.LayerNorm(d_model)
        self.task_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.task_gate = nn.Parameter(torch.full((d_model,), -2.0))

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, visual_tokens, task_tokens):
        B, T, D = x.shape
        x = x + self.self_attn(self.norm1(x))

        x_flat = x.reshape(B * T, 1, D)  # each frame attends only to its own visual tokens
        x = (x_flat + self.cross_attn(self.norm2(x_flat), visual_tokens, visual_tokens)[0]).reshape(B, T, D)

        h = self.task_cross_attn(self.norm_task(x), task_tokens, task_tokens)[0]
        x = x + torch.sigmoid(self.task_gate) * h

        return x + self.ffn(self.norm3(x))
