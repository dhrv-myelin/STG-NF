"""
Split pose JSON files into train (normal frames only) and test (all frames).

Theft timings (0-indexed frame numbers):
  Vid1: 100-200 (pocketing)
  Vid2: 150-200 (hiding under phone)
  Vid3: 120-250 (pocketing)
  Vid4: 130-150 (hiding under phone)

Train = ALL frames EXCEPT the theft window (before + after = normal cashier behavior).
Test  = ALL frames (to see the theft spike in scores).

Usage:
    python scripts/split_pose_by_frame.py \
        --input_dir data/Cashier/pose/all/ \
        --train_dir data/Cashier/pose/train/ \
        --test_dir  data/Cashier/pose/test/

The script reads every *_alphapose_tracked_person.json in input_dir,
splits each by the theft timings below, and writes the two subsets.
"""
import argparse
import json
import os

# 0-indexed frame ranges where theft occurs (inclusive)
THEFT_TIMINGS = {
    "Vid1": (100, 200),
    "Vid2": (150, 200),
    "Vid3": (120, 250),
    "Vid4": (130, 150),
}


def split_json(json_path, theft_start, theft_end):
    """Return (normal_dict, full_dict) where normal excludes the theft window."""
    with open(json_path, "r") as f:
        data = json.load(f)

    normal = {}
    for person_id, frames in data.items():
        normal_frames = {}
        for frame_key, frame_data in frames.items():
            frame_idx = int(frame_key)
            if frame_idx < theft_start or frame_idx > theft_end:
                normal_frames[frame_key] = frame_data
        if normal_frames:
            normal[person_id] = normal_frames

    return normal, data


def main():
    ap = argparse.ArgumentParser(description="Split pose JSONs by theft timing")
    ap.add_argument("--input_dir", required=True, help="Directory with all pose JSONs")
    ap.add_argument("--train_dir", required=True, help="Output dir for normal-only JSONs")
    ap.add_argument("--test_dir", required=True, help="Output dir for full JSONs (symlink/copy)")
    args = ap.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith("_alphapose_tracked_person.json")
    )

    for jf in json_files:
        vid_name = jf.split("_alphapose_tracked_person")[0]
        if vid_name not in THEFT_TIMINGS:
            print(f"  Skipping {jf} (no theft timing defined for {vid_name})")
            continue

        theft_start, theft_end = THEFT_TIMINGS[vid_name]
        src = os.path.join(args.input_dir, jf)
        print(f"  {vid_name}: theft frames {theft_start}-{theft_end}")

        normal_data, full_data = split_json(src, theft_start, theft_end)

        # Count frames
        total_frames = sum(len(v) for v in full_data.values())
        normal_frames = sum(len(v) for v in normal_data.values())
        print(f"    Total frames: {total_frames}, Normal frames: {normal_frames}")

        if normal_frames < 20:
            print(f"    WARNING: Only {normal_frames} normal frames — training may be unstable")

        # Write normal-only version to train
        train_path = os.path.join(args.train_dir, jf)
        with open(train_path, "w") as f:
            json.dump(normal_data, f)
        print(f"    -> {train_path}")

        # Write full version to test
        test_path = os.path.join(args.test_dir, jf)
        with open(test_path, "w") as f:
            json.dump(full_data, f)
        print(f"    -> {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
