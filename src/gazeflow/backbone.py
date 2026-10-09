import numpy as np
import torch
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(ckpt: dict, backbone_path: str, device: torch.device):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        backbone_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    lora = ckpt["config"]["lora"]
    cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"],
        bias="none",
    )
    model = get_peft_model(model, cfg)
    set_peft_model_state_dict(model, ckpt["lora"])
    return model.to(device).eval()


def _preprocess(frames, target_size, device):
    arr = np.stack([np.asarray(f) for f in frames])
    x = (
        torch.from_numpy(arr)
        .to(device=device, non_blocking=True)
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
    )
    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.repeat_interleave(2, dim=0)  # V-JEPA 2 tubelet size is 2: duplicate each frame
    return x.unsqueeze(0).to(dtype=torch.bfloat16)


@torch.no_grad()
def extract_clip_features(
    backbone,
    video_path: str,
    n_frames: int,
    target_size=(256, 256),
    layer_idx: int = 23,
    chunk_max: int = 32,
    chunk_overlap: int = 4,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    import decord
    from PIL import Image

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    indices = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
    raw = vr.get_batch(indices.tolist()).asnumpy()
    pil_frames = [Image.fromarray(raw[i]) for i in range(len(raw))]

    T = len(pil_frames)
    if T <= chunk_max:
        chunks = [(0, T)]
    else:
        step = chunk_max - chunk_overlap
        chunks = []
        s = 0
        while s < T:
            e = min(s + chunk_max, T)
            chunks.append((s, e))
            if e == T:
                break
            s += step

    per_frame = {}
    for cs, ce in chunks:
        chunk = pil_frames[cs:ce]
        video_tensor = _preprocess(chunk, target_size, device)
        out = backbone(pixel_values_videos=video_tensor, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx + 1].squeeze(0)
        n = ce - cs
        tpf = hidden.shape[0] // n
        for i in range(n):
            per_frame.setdefault(cs + i, []).append(hidden[i * tpf:(i + 1) * tpf])

    merged = []
    for i in range(T):
        f = per_frame[i]
        merged.append(f[0] if len(f) == 1 else torch.stack(f).mean(dim=0))  # frames in two chunks are averaged
    return torch.stack(merged).float()
