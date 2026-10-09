import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gazeflow.backbone import extract_clip_features, load_backbone
from gazeflow.model import GazeFlowNet, GazeFlowSampler
from gazeflow.sampling import denormalize_scanpath, sample_clip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/gazeflow.pt")
    p.add_argument("--manifest", default="data/manifests/egtea_val.jsonl")
    p.add_argument("--egtea-root", default="data/egtea")
    p.add_argument("--output-dir", default="runs/egtea_val")
    p.add_argument("--backbone-path", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--num-ode-steps", type=int, default=50)
    p.add_argument("--sample-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-clips", type=int, default=None)
    return p.parse_args()


def build_sampler(ckpt: dict, device: torch.device) -> GazeFlowSampler:
    c = ckpt["config"]
    flow_net = GazeFlowNet(
        d_model=c["d_model"],
        n_layers=c["n_layers"],
        n_heads=c["n_heads"],
        feat_dim=c["feat_dim"],
        window_size=c["window_size"],
        n_task_queries=c["n_task_queries"],
        n_spatial_layers=c["n_spatial_layers"],
    )
    flow_net.load_state_dict(ckpt["flow_net"])
    return GazeFlowSampler(flow_net).to(device).eval()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    backbone = load_backbone(ckpt, args.backbone_path, device)
    model = build_sampler(ckpt, device)

    with open(args.manifest) as f:
        clips = [json.loads(line) for line in f if line.strip()]
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"[load] {len(clips)} clips from {args.manifest}")

    generated = {}
    for i, rec in enumerate(clips):
        n_frames = rec["num_frames"]
        feats = extract_clip_features(
            backbone, str(Path(args.egtea_root) / rec["video_path"]), n_frames,
            target_size=(cfg["resize"], cfg["resize"]), layer_idx=cfg["backbone_layer"], device=device,
        )
        norm = sample_clip(
            model, feats, n_frames,
            window_size=cfg["window_size"], stride=cfg["stride"],
            num_samples=args.num_samples,
            num_ode_steps=args.num_ode_steps,
            sample_batch_size=args.sample_batch_size,
            device=device,
        )
        generated[rec["id"]] = np.stack([denormalize_scanpath(s) for s in norm])

        if (i + 1) % 10 == 0 or i == len(clips) - 1:
            print(f"  [{i + 1}/{len(clips)}] {rec['id']}  T={n_frames}", flush=True)

    out_path = out_dir / "generated_scanpaths.npz"
    np.savez_compressed(str(out_path), **generated)
    print(f"[save] {out_path}  ({len(generated)} clips)")


if __name__ == "__main__":
    main()
