# Manifest format

`egtea_val.jsonl` has one EGTEA Gaze+ clip per line:

- `id`: clip name (key in the output npz)
- `video_path`: path relative to `--egtea-root`
- `num_frames`: number of 24 fps frames sampled from the clip
- `clip_uid`: recording the clip comes from (name of its gaze file)
- `start_frame`, `end_frame`: frame range of the clip in that recording
