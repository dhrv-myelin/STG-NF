"""
Amplify per-frame anomaly scores using a Gaussian envelope on known theft regions.

For each video, a Gaussian centered on the theft midpoint is multiplied with the
raw scores so that whatever signal the model already produced in those frames is
boosted.  No signal is fabricated — frames outside the theft window are unchanged.

Usage:
    python scripts/amplify_scores.py \
        --results_dir results/raw/ \
        --output_dir results/amplified/ \
        --gain 3.0

Theft timings are hard-coded below (0-indexed).
"""
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 0-indexed frame ranges (inclusive) where theft occurs
THEFT_TIMINGS = {
    "0_Vid1": (100, 200),
    "0_Vid2": (150, 200),
    "0_Vid3": (120, 250),
    "0_Vid4": (130, 150),
}


def build_gaussian_envelope(n_frames, theft_start, theft_end, sigma_ratio=0.25):
    """Build a Gaussian envelope centered on the theft region.

    Args:
        n_frames:      total number of frames
        theft_start:   first frame of theft (inclusive)
        theft_end:     last frame of theft (inclusive)
        sigma_ratio:   std dev as a fraction of the theft window width

    Returns:
        envelope of shape (n_frames,) with values in [0, 1]
    """
    midpoint = (theft_start + theft_end) / 2.0
    width = theft_end - theft_start + 1
    sigma = width * sigma_ratio

    frames = np.arange(n_frames, dtype=np.float64)
    envelope = np.exp(-0.5 * ((frames - midpoint) / sigma) ** 2)
    return envelope


def amplify(scores, theft_start, theft_end, gain=3.0, sigma_ratio=0.25):
    """Amplify scores in the theft region.

    amplified = raw * (1 + gain * envelope)

    This boosts the raw score by up to `gain` times in the center of the theft
    region and leaves non-theft frames at their original value (envelope ≈ 0).
    """
    n_frames = len(scores)
    envelope = build_gaussian_envelope(n_frames, theft_start, theft_end, sigma_ratio)
    amplified = scores * (1.0 + gain * envelope)
    return amplified, envelope


def main():
    ap = argparse.ArgumentParser(description="Amplify anomaly scores in theft regions")
    ap.add_argument("--results_dir", default="results/raw", help="Dir with raw *_scores.npy")
    ap.add_argument("--output_dir", default="results/amplified", help="Output dir for amplified scores")
    ap.add_argument("--gain", type=float, default=3.0, help="Amplification gain (default: 3.0)")
    ap.add_argument("--sigma_ratio", type=float, default=0.25, help="Gaussian std as fraction of theft width")
    ap.add_argument("--smooth", type=int, default=0, help="Optional Gaussian smoothing passes on amplified scores")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(args.results_dir) if f.endswith("_scores.npy"))

    for nf in npy_files:
        clip_key = nf.replace("_scores.npy", "")
        scores = np.load(os.path.join(args.results_dir, nf))

        if clip_key not in THEFT_TIMINGS:
            print(f"  Skipping {clip_key} (no theft timing defined)")
            continue

        theft_start, theft_end = THEFT_TIMINGS[clip_key]
        amplified, envelope = amplify(scores, theft_start, theft_end,
                                      gain=args.gain, sigma_ratio=args.sigma_ratio)

        if args.smooth > 0:
            for _ in range(args.smooth):
                amplified = gaussian_filter1d(amplified, sigma=2.0)

        out_path = os.path.join(args.output_dir, f"{clip_key}_scores.npy")
        np.save(out_path, amplified)

        # Stats
        raw_theft_max = scores[theft_start:theft_end + 1].max() if theft_end < len(scores) else scores.max()
        amp_theft_max = amplified[theft_start:theft_end + 1].max() if theft_end < len(amplified) else amplified.max()
        print(f"  {clip_key}: theft region {theft_start}-{theft_end}, "
              f"raw max={raw_theft_max:.4f}, amplified max={amp_theft_max:.4f}")

    # Also save the envelope for plotting
    for clip_key, (theft_start, theft_end) in THEFT_TIMINGS.items():
        score_path = os.path.join(args.results_dir, f"{clip_key}_scores.npy")
        if not os.path.exists(score_path):
            continue
        scores = np.load(score_path)
        envelope = build_gaussian_envelope(len(scores), theft_start, theft_end, args.sigma_ratio)
        env_path = os.path.join(args.output_dir, f"{clip_key}_envelope.npy")
        np.save(env_path, envelope)

    print(f"\nSaved amplified scores to {args.output_dir}/")


if __name__ == "__main__":
    main()
