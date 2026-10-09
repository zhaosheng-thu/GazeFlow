import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gazeflow.egtea import VIDEO_H, VIDEO_W, clip_ground_truth, parse_gaze_file
from gazeflow.metrics import (
    adaptive_f1_frame,
    adaptive_f1_from_averages,
    auc_judd,
    average_angular_error,
    heatmap_center_of_mass,
    points_to_fixation_map,
    points_to_heatmap,
)

SIGMA = 25.0
N_EVAL_FRAMES = 8
F1_THRESHOLDS = np.linspace(0, 1, 51)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="runs/egtea_val/generated_scanpaths.npz")
    p.add_argument("--manifest", default="data/manifests/egtea_val.jsonl")
    p.add_argument("--egtea-root", default="data/egtea")
    p.add_argument("--output", default=None)
    return p.parse_args()


def evaluate(preds, records, gaze_dir: Path) -> dict:
    h, w = VIDEO_H, VIDEO_W
    gaze_cache = {}
    clip_auc, clip_aae = [], []
    frame_recall, frame_precision = [], []

    for rec in records:
        if rec["id"] not in preds:
            continue
        samples = preds[rec["id"]]
        T = rec["num_frames"]

        uid = rec["clip_uid"]
        if uid not in gaze_cache:
            path = str(gaze_dir / f"{uid}.txt")
            gaze_cache[uid] = (parse_gaze_file(path, drop_blinks=True), parse_gaze_file(path, drop_blinks=False))
        gt_points, fixations = clip_ground_truth(*gaze_cache[uid], rec["start_frame"], rec["end_frame"], T)

        f_auc, f_aae = [], []
        for n in np.round(np.linspace(0, T - 1, min(N_EVAL_FRAMES, T))).astype(int):
            if fixations and n not in fixations:  # GLC protocol: only fixation frames are scored
                continue
            pts = gt_points[n]
            if pts.shape[0] == 0:
                continue

            pred_map = points_to_heatmap(samples[:, n], h, w, SIGMA)
            gt_map = points_to_heatmap(pts, h, w, SIGMA)
            f_auc.append(auc_judd(pred_map, points_to_fixation_map(pts, h, w)))

            r, p = adaptive_f1_frame(pred_map, gt_map, F1_THRESHOLDS)
            frame_recall.append(r)
            frame_precision.append(p)

            gt_xy = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            f_aae.append(average_angular_error(heatmap_center_of_mass(pred_map), gt_xy, h, w))

        if f_auc:
            clip_auc.append(np.mean(f_auc))
            clip_aae.append(np.mean(f_aae))

    if not frame_recall:
        raise SystemExit(f"No ground-truth gaze found in {gaze_dir}. Check --egtea-root.")
    f1, precision, recall = adaptive_f1_from_averages(
        np.mean(np.stack(frame_recall), axis=0), np.mean(np.stack(frame_precision), axis=0),
    )
    return {
        "num_clips": len(clip_auc),
        "AUC": float(np.mean(clip_auc)),
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AAE": float(np.mean(clip_aae)),
    }


def main():
    args = parse_args()
    with open(args.manifest) as f:
        records = [json.loads(line) for line in f if line.strip()]
    z = np.load(args.predictions)
    preds = {k: z[k] for k in z.files}

    m = evaluate(preds, records, Path(args.egtea_root) / "gaze_data")
    print(f"clips={m['num_clips']}  AUC={m['AUC']:.3f}  F1={m['F1']:.3f}  "
          f"P={m['Precision']:.3f}  R={m['Recall']:.3f}  AAE={m['AAE']:.2f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
