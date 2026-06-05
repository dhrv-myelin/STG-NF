"""
Filter pose JSONs to keep only the cashier (person with most frames).

The cashier is typically the person present for the entire video, while
customers/passersby appear briefly. The script auto-detects the cashier
as the track ID with the highest frame count.

Usage:
    python scripts/filter_cashier.py \
        --input_dir data/Cashier/pose/all/ \
        --output_dir data/Cashier/pose/filtered/
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Filter pose JSONs to cashier only")
    ap.add_argument("--input_dir", required=True, help="Directory with pose JSONs")
    ap.add_argument("--output_dir", required=True, help="Output directory for filtered JSONs")
    ap.add_argument("--min_frame_ratio", type=float, default=0.3,
                    help="Minimum fraction of total frames to be considered cashier (default: 0.3)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith("_alphapose_tracked_person.json")
    )

    for jf in json_files:
        src = os.path.join(args.input_dir, jf)
        with open(src, "r") as f:
            data = json.load(f)

        if not data:
            print(f"  {jf}: empty JSON, skipping")
            continue

        # Count frames per track ID
        track_frames = {pid: len(frames) for pid, frames in data.items()}
        total_frames = sum(track_frames.values())
        cashier_id = max(track_frames, key=track_frames.get)
        cashier_frames = track_frames[cashier_id]
        ratio = cashier_frames / total_frames if total_frames > 0 else 0

        # Check if the "cashier" is actually dominant enough
        if ratio < args.min_frame_ratio:
            print(f"  {jf}: WARNING — top track {cashier_id} has only "
                  f"{cashier_frames}/{total_frames} frames ({ratio:.1%}). "
                  f"Keeping it anyway.")

        # Keep only the cashier
        filtered = {cashier_id: data[cashier_id]}

        dst = os.path.join(args.output_dir, jf)
        with open(dst, "w") as f:
            json.dump(filtered, f)

        # Report all tracks for debugging
        tracks_str = ", ".join(f"id={pid}: {n} frames" for pid, n in
                               sorted(track_frames.items(), key=lambda x: -x[1]))
        print(f"  {jf}: kept track {cashier_id} ({cashier_frames} frames). "
              f"All tracks: {tracks_str}")

    print(f"\nFiltered JSONs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
