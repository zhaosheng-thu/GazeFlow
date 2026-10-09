import numpy as np
import torch

from gazeflow.model import GazeFlowSampler


def sample_clip(
    model: GazeFlowSampler,
    frame_feats: torch.Tensor,
    n_frames: int,
    window_size: int,
    stride: int,
    num_samples: int = 50,
    num_ode_steps: int = 50,
    sample_batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    T = n_frames

    if T <= window_size:
        feat_padded = torch.zeros(window_size, frame_feats.shape[1], frame_feats.shape[2])
        feat_padded[:T] = frame_feats[:T]
        feat_padded[T:] = frame_feats[T - 1]  # pad short clips with the last frame
        samples = model.sample(
            feat_padded.unsqueeze(0).float().to(device),
            num_samples=num_samples,
            num_steps=num_ode_steps,
            batch_size=sample_batch_size,
        )
        return samples[:, :T, :].cpu().numpy()

    overlap = window_size - stride
    starts = list(range(0, T - window_size + 1, stride))
    if starts[-1] + window_size < T:
        starts.append(T - window_size)

    out_chunks = []
    for batch_start in range(0, num_samples, sample_batch_size):
        bs = min(sample_batch_size, num_samples - batch_start)
        accum = np.zeros((bs, T, 2), dtype=np.float64)
        weights = np.zeros((bs, T), dtype=np.float64)

        for start in starts:
            end = start + window_size
            feat_win = frame_feats[start:end].unsqueeze(0).float().to(device)
            win_out = model.sample(
                feat_win,
                num_samples=bs,
                num_steps=num_ode_steps,
                batch_size=bs,
            ).cpu().numpy()

            win_w = np.ones(window_size, dtype=np.float64)
            if overlap > 0:
                if start > 0:
                    win_w[:overlap] = np.linspace(0.0, 1.0, overlap)
                if end < T:
                    win_w[-overlap:] = np.linspace(1.0, 0.0, overlap)

            for b in range(bs):
                accum[b, start:end] += win_out[b] * win_w[:, None]
                weights[b, start:end] += win_w

        valid = weights > 0
        for b in range(bs):
            accum[b][valid[b]] /= weights[b][valid[b]][:, None]
        out_chunks.append(accum.astype(np.float32))

    return np.concatenate(out_chunks, axis=0)


def denormalize_scanpath(scanpath: np.ndarray, w: int = 640, h: int = 480) -> np.ndarray:
    out = np.copy(scanpath)
    out[:, 0] = (out[:, 0] + 1.0) / 2.0 * (w - 1)
    out[:, 1] = (out[:, 1] + 1.0) / 2.0 * (h - 1)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out
