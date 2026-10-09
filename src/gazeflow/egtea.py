import numpy as np

GAZE_W, GAZE_H = 1280, 960
VIDEO_W, VIDEO_H = 640, 480


def _timestamp_to_frame(ts: str, fps: int = 24) -> int:
    h, m, s, f = ts.split(":")
    return (3600 * int(h) + 60 * int(m) + int(s)) * fps + int(f)


def parse_gaze_file(path: str, drop_blinks: bool) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    begaze_v1 = len(lines) > 10 and "## Number of Samples:" in lines[9]  # OP* sessions; P* sessions use v2

    frames = {}
    for line in lines:
        if line.startswith("#") or line.startswith("Time"):
            continue
        try:
            if begaze_v1:
                parts = line.strip().split("\t")
                if len(parts) < 7:
                    continue
                x, y, frame, event = float(parts[3]), float(parts[4]), int(parts[5]), parts[-1].strip()
            else:
                vals = line.strip().split()
                if len(vals) < 26:
                    continue
                x, y, frame, event = float(vals[5]), float(vals[6]), _timestamp_to_frame(vals[-2]), vals[-1].strip()
        except (ValueError, IndexError):
            continue
        if drop_blinks and (event == "Blink" or not (0 <= x < GAZE_W and 0 <= y < GAZE_H)):
            continue
        if frame in frames:
            px, py, _ = frames[frame]
            frames[frame] = ((px + x) / 2.0, (py + y) / 2.0, event)
        else:
            frames[frame] = (x, y, event)

    return {
        f: (min(max(0.0, x), GAZE_W - 1), min(max(0.0, y), GAZE_H - 1), e)
        for f, (x, y, e) in sorted(frames.items())
    }


def clip_ground_truth(positions: dict, labels: dict, start_frame: int, end_frame: int, num_frames: int):
    points = []
    for f in range(start_frame, start_frame + num_frames):
        if f < end_frame and f in positions:
            x, y, _ = positions[f]
            x = min(max(x * VIDEO_W / GAZE_W, 0), VIDEO_W - 1)
            y = min(max(y * VIDEO_H / GAZE_H, 0), VIDEO_H - 1)
            points.append(np.array([[x, y]], dtype=np.float32))
        else:
            points.append(np.empty((0, 2), dtype=np.float32))
    fixations = {
        f - start_frame for f, (_, _, e) in labels.items()
        if start_frame <= f < end_frame and e == "Fixation"
    }
    return points, fixations
