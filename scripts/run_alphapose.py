import argparse
import json
import os
import subprocess
from pathlib import Path
from posixpath import basename


def convert_data_format(data):
    data_new = {}
    for item in data:
        frame_idx = str(item['image_id']).rsplit('.', 1)[0].zfill(6)
        person_idx = str(item['idx'])
        kp = item['keypoints']
        sc = item['score']
        if person_idx not in data_new:
            data_new[person_idx] = {}
        data_new[person_idx][frame_idx] = {'keypoints': kp, 'scores': sc}
    return data_new


def process_video(video_path, output_dir, alphapose_dir, skip_existing):
    stem = Path(video_path).stem
    out_name = f"{stem}_alphapose_tracked_person.json"
    out_path = os.path.join(output_dir, out_name)

    if skip_existing and os.path.exists(out_path):
        print(f"  Skipping {stem} (already exists)")
        return

    alphapose_results = os.path.join(output_dir, 'alphapose-results.json')

    cmd = [
        'python', 'scripts/demo_inference.py',
        '--cfg', os.path.join(alphapose_dir, 'configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml'),
        '--checkpoint', os.path.join(alphapose_dir, 'pretrained_models/fast_421_res152_256x192.pth'),
        '--outdir', output_dir,
        '--sp', '--pose_track',
        '--video', video_path,
    ]

    print(f"  Processing {stem}...")
    result = subprocess.run(cmd, cwd=alphapose_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR processing {stem}: {result.stderr[:200]}")
        return

    if not os.path.exists(alphapose_results):
        print(f"  ERROR: alphapose-results.json not found for {stem}")
        return

    with open(alphapose_results) as f:
        raw = json.load(f)

    converted = convert_data_format(raw)
    with open(out_path, 'w') as f:
        json.dump(converted, f)

    os.remove(alphapose_results)
    poseflow_dir = os.path.join(output_dir, 'poseflow')
    if os.path.isdir(poseflow_dir):
        for fname in os.listdir(poseflow_dir):
            os.remove(os.path.join(poseflow_dir, fname))
        os.rmdir(poseflow_dir)

    print(f"  Saved {out_name}")


def main():
    ap = argparse.ArgumentParser(description='Batch AlphaPose extraction for STG-NF')
    ap.add_argument('--input_dir', required=True, help='Directory with video files')
    ap.add_argument('--output_dir', required=True, help='Directory for JSON output')
    ap.add_argument('--alphapose_dir', required=True, help='AlphaPose installation directory')
    ap.add_argument('--extensions', default='.mp4,.avi,.mov', help='Comma-separated video extensions (default: .mp4,.avi,.mov)')
    ap.add_argument('--skip_existing', action='store_true', help='Skip already-processed videos')

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    exts = tuple(args.extensions.split(','))
    videos = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    videos.sort()

    if not videos:
        print(f"No videos found in {args.input_dir}")
        return

    print(f"Found {len(videos)} videos")
    for v in videos:
        process_video(os.path.join(args.input_dir, v), args.output_dir, args.alphapose_dir, args.skip_existing)


if __name__ == '__main__':
    main()
