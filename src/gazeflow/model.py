import math

import torch
import torch.nn as nn

from gazeflow.blocks import FlowBlock, make_2d_sinusoidal_pe
from gazeflow.spatial_encoder import SpatialEncoder
from gazeflow.time_embed import SinusoidalEmbedding


class GazeFlowNet(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        feat_dim: int = 1024,
        window_size: int = 64,
        n_task_queries: int = 4,
        n_spatial_layers: int = 2,
        grid_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        self.coord_encoder = nn.Sequential(
            nn.Linear(4, d_model),  # x_t and the self-conditioning estimate of x_1
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.visual_proj = nn.Linear(feat_dim, d_model)
        self.register_buffer("spatial_pe", make_2d_sinusoidal_pe(grid_size, grid_size, d_model), persistent=False)
        self.spatial_encoder = SpatialEncoder(d_model, n_heads, n_spatial_layers)

        self.task_queries = nn.Parameter(torch.zeros(n_task_queries, d_model))
        self.task_extract_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.task_extract_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([FlowBlock(d_model, n_heads, window_size) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 2)

    def encode_video(self, feats: torch.Tensor):
        B, T, N = feats.shape[:3]
        g = math.isqrt(N)
        v = self.visual_proj(feats).reshape(B, T, g, g, self.d_model) + self.spatial_pe
        v = self.spatial_encoder(v.reshape(B, T, N, self.d_model))

        frame_tokens = v.mean(dim=2)
        pool_token = frame_tokens.mean(dim=1, keepdim=True)
        q = self.task_extract_norm(self.task_queries.unsqueeze(0).expand(B, -1, -1))
        learned, _ = self.task_extract_attn(q, frame_tokens, frame_tokens)
        task_tokens = torch.cat([pool_token, learned], dim=1)  # top-down condition

        return v.reshape(B * T, N, self.d_model), task_tokens

    def forward(self, x_t, t, x1_hat, visual_tokens, task_tokens):
        x = self.coord_encoder(torch.cat([x_t, x1_hat], dim=-1))
        x = x + self.time_embed(t * 1000.0)[:, None, :]
        for layer in self.layers:
            x = layer(x, visual_tokens, task_tokens)
        return self.output_head(self.final_norm(x))


class GazeFlowSampler(nn.Module):
    def __init__(self, flow_net: GazeFlowNet):
        super().__init__()
        self.flow_net = flow_net

    @torch.no_grad()
    def sample(self, feats: torch.Tensor, num_samples: int = 1, num_steps: int = 50, batch_size: int = 16):
        T = feats.shape[1]
        dt = 1.0 / num_steps

        chunks = []
        for start in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - start)
            visual_tokens, task_tokens = self.flow_net.encode_video(feats.expand(bs, -1, -1, -1))

            x = torch.randn(bs, T, 2, device=feats.device)
            x1_hat = torch.zeros_like(x)
            for k in range(num_steps):
                t_k = k * dt
                t = torch.full((bs,), t_k, device=feats.device)
                v = self.flow_net(x, t, x1_hat, visual_tokens, task_tokens)
                x1_hat = (x + (1.0 - t_k) * v).clamp(-1, 1)
                x = x + dt * v  # Euler step from noise (t=0) to gaze (t=1)

            chunks.append(x.clamp(-1, 1))

        return torch.cat(chunks, dim=0)
