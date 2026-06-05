import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics import YOLO


def process_video(video_path, output_dir, model, skip_existing, conf, iou):
    stem = Path(video_path).stem
    out_name = f"{stem}_alphapose_tracked_person.json"
    out_path = os.path.join(output_dir, out_name)

    if skip_existing and os.path.exists(out_path):
        tqdm.write(f"  Skipping {stem} (already exists)")
        return

    results = model.track(
        source=str(video_path),
        persist=True,
        stream=True,
        verbose=False,
        conf=conf,
        iou=iou,
        tracker="bytetrack.yaml",
        device=0 if torch.cuda.is_available() else "cpu",
    )

    data = {}

    for frame_idx, result in enumerate(tqdm(results, desc=f"  {stem}", leave=False)):
        frame_key = str(frame_idx).zfill(6)

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes
        track_ids = boxes.id.cpu().numpy().astype(int)
        keypoints = result.keypoints

        for pid, kp in zip(track_ids, keypoints.data.cpu().numpy()):
            pid_str = str(pid)
            flat_kp = kp.flatten().tolist()
            score = float(kp[:, 2].mean())

            if pid_str not in data:
                data[pid_str] = {}
            data[pid_str][frame_key] = {
                "keypoints": flat_kp,
                "scores": score,
            }

    if not data:
        tqdm.write(f"  WARNING: No detections in {stem}, writing empty JSON")
    else:
        tqdm.write(
            f"  {stem}: {len(data)} tracks, {sum(len(v) for v in data.values())} frames"
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)


def main():
    ap = argparse.ArgumentParser(
        description="YOLO-pose + ByteTrack extraction for STG-NF"
    )
    ap.add_argument("--input_dir", required=True, help="Directory with video files")
    ap.add_argument("--output_dir", required=True, help="Directory for JSON output")
    ap.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model (default: yolo11s-pose.pt)",
    )
    ap.add_argument(
        "--conf", type=float, default=0.3, help="Detection confidence threshold"
    )
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument(
        "--extensions",
        default=".mp4,.avi,.mov",
        help="Comma-separated video extensions",
    )
    ap.add_argument(
        "--skip_existing", action="store_true", help="Skip already-processed videos"
    )
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    model = YOLO(args.model)

    exts = tuple(args.extensions.split(","))
    videos = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))

    if not videos:
        print(f"No videos found in {args.input_dir}")
        sys.exit(1)

    for v in tqdm(videos, desc="Videos", unit="video"):
        process_video(
            os.path.join(args.input_dir, v),
            args.output_dir,
            model,
            args.skip_existing,
            args.conf,
            args.iou,
        )


if __name__ == "__main__":
    main()
