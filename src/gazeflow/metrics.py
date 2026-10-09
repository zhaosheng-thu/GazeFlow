import math

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter

_EPS = 1e-8
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def points_to_heatmap(points: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    hmap = np.zeros((h, w), dtype=np.float64)
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            hmap[iy, ix] += 1.0
    hmap = gaussian_filter(hmap, sigma=sigma)
    total = hmap.sum()
    if total > 0:
        hmap /= total
    return hmap


def points_to_fixation_map(points: np.ndarray, h: int, w: int) -> np.ndarray:
    fix = np.zeros((h, w), dtype=np.uint8)
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            fix[iy, ix] = 1
    return fix


def auc_judd(pred_map: np.ndarray, fixation_map: np.ndarray) -> float:
    pred = pred_map.astype(np.float64)
    fix = fixation_map.astype(bool)
    if fix.sum() == 0:
        return 0.5
    pred = (pred - pred.min()) / (pred.max() - pred.min() + _EPS)
    thresholds = np.sort(pred[fix])[::-1]

    num_fix = fix.sum()
    num_nonfix = (~fix).sum()
    if num_nonfix == 0:
        return 1.0

    tpr, fpr = [0.0], [0.0]
    for th in thresholds:
        above = pred >= th
        tpr.append(np.logical_and(above, fix).sum() / (num_fix + _EPS))
        fpr.append(np.logical_and(above, ~fix).sum() / (num_nonfix + _EPS))
    tpr.append(1.0)
    fpr.append(1.0)

    order = np.argsort(fpr)
    return float(_trapezoid(np.array(tpr)[order], np.array(fpr)[order]))


def adaptive_f1_frame(pred_map: np.ndarray, gt_map: np.ndarray, thresholds: np.ndarray):
    pred = pred_map.astype(np.float64)
    pmin, pmax = pred.min(), pred.max()
    pred = (pred - pmin) / (pmax - pmin) if pmax - pmin > _EPS else np.zeros_like(pred)

    gt_bin = gt_map > 0.001 * (64 * 64) / gt_map.size  # GLC's 0.001 at 64x64, rescaled to this resolution
    fg_labels = float(gt_bin.sum())

    recall = np.zeros(len(thresholds), dtype=np.float64)
    precision = np.zeros(len(thresholds), dtype=np.float64)
    for i, th in enumerate(thresholds):
        pred_bin = pred >= th
        tp = float(np.logical_and(pred_bin, gt_bin).sum())
        recall[i] = tp / (fg_labels + _EPS)
        precision[i] = tp / (float(pred_bin.sum()) + _EPS)
    return recall, precision


def adaptive_f1_from_averages(mean_recall, mean_precision):
    f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + _EPS)
    best = int(np.argmax(f1))
    return float(f1[best]), float(mean_precision[best]), float(mean_recall[best])


def heatmap_center_of_mass(heatmap: np.ndarray):
    h, w = heatmap.shape
    if heatmap.sum() == 0:
        return (w / 2.0, h / 2.0)
    row, col = center_of_mass(heatmap)
    return (float(col), float(row))


def average_angular_error(pred_point, gt_point, image_h: int, image_w: int,
                          fov_half: float = math.pi / 6) -> float:
    cx, cy = image_w / 2.0, image_h / 2.0
    d = (image_h / 2.0) / math.tan(fov_half)
    r1 = np.array([pred_point[0] - cx, pred_point[1] - cy, d])
    r2 = np.array([gt_point[0] - cx, gt_point[1] - cy, d])
    return math.degrees(math.atan2(np.linalg.norm(np.cross(r1, r2)), np.dot(r1, r2)))
