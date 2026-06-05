"""
Refine raw anomaly scores via local z-score normalization.

For each frame, compute how far its score deviates from the local neighborhood
(mean ± std over a sliding window). This makes theft frames pop out relative
to their context WITHOUT needing theft timings.

refined_score[i] = (score[i] - local_mean[i]) / (local_std[i] + eps)

High positive values = frames that are anomalous compared to their neighbors.

Usage:
    python scripts/refine_scores.py \
        --results_dir results/raw/ \
        --output_dir results/refined/ \
        --window 100 \
        --smooth_sigma 3
"""
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


def local_zscore(scores, window=100, eps=1e-8):
    """Compute local z-score for each frame.

    For frame i, look at scores in [i-window//2, i+window//2],
    compute mean and std, then z-score = (score - mean) / std.

    Frames far from anomalies will have z ≈ 0.
    Anomalous frames will have high |z|.
    """
    n = len(scores)
    z = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2)
        local = scores[lo:hi]
        local_mean = np.mean(local)
        local_std = np.std(local)
        z[i] = (scores[i] - local_mean) / (local_std + eps)

    return z


def rolling_percentile_baseline(scores, window=200, percentile=10):
    """Compute a rolling percentile baseline and subtract it.

    The idea: the baseline tracks the "normal" score level using a low
    percentile (e.g. 10th — assuming normal frames are the majority).
    Anomalous frames sit above this baseline.

    Returns: scores - baseline (higher = more anomalous)
    """
    n = len(scores)
    baseline = np.zeros(n, dtype=np.float64)
    half_w = window // 2

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w)
        baseline[i] = np.percentile(scores[lo:hi], percentile)

    return scores - baseline


def main():
    ap = argparse.ArgumentParser(description="Refine anomaly scores via local z-score")
    ap.add_argument("--results_dir", default="results/raw", help="Dir with raw *_scores.npy")
    ap.add_argument("--output_dir", default="results/refined", help="Output dir for refined scores")
    ap.add_argument("--window", type=int, default=100,
                    help="Sliding window size for local z-score (default: 100)")
    ap.add_argument("--smooth_sigma", type=float, default=3.0,
                    help="Light Gaussian smoothing on refined scores (default: 3)")
    ap.add_argument("--method", choices=["zscore", "baseline", "both"], default="both",
                    help="zscore = local z-score only, baseline = rolling percentile, both = max of both")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(args.results_dir) if f.endswith("_scores.npy"))

    for nf in npy_files:
        clip_key = nf.replace("_scores.npy", "")
        scores = np.load(os.path.join(args.results_dir, nf))

        # Replace inf/nan for computation, restore after
        finite = scores[np.isfinite(scores)]
        if len(finite) == 0:
            print(f"  Skipping {clip_key} (all scores are inf/nan)")
            continue

        working = scores.copy()
        working[~np.isfinite(working)] = np.median(finite)

        results = {}

        if args.method in ("zscore", "both"):
            z = local_zscore(working, window=args.window)
            # Z-scores are centered at 0, shift to [0, max]
            z_shifted = z - z.min()
            if args.smooth_sigma > 0:
                z_shifted = gaussian_filter1d(z_shifted, sigma=args.smooth_sigma)
            results["zscore"] = z_shifted

        if args.method in ("baseline", "both"):
            b = rolling_percentile_baseline(working, window=args.window * 2)
            # Negate so higher = more anomalous
            b_neg = -b
            b_shifted = b_neg - b_neg.min()
            if args.smooth_sigma > 0:
                b_shifted = gaussian_filter1d(b_shifted, sigma=args.smooth_sigma)
            results["baseline"] = b_shifted

        if args.method == "both":
            # Take the max of both methods at each frame
            refined = np.maximum(results["zscore"], results["baseline"])
        elif args.method == "zscore":
            refined = results["zscore"]
        else:
            refined = results["baseline"]

        # Normalize to [0, 1]
        rmax = refined.max()
        if rmax > 1e-8:
            refined = refined / rmax
        refined = np.clip(refined, 0.0, 1.0)

        out_path = os.path.join(args.output_dir, f"{clip_key}_scores.npy")
        np.save(out_path, refined)

        # Stats
        print(f"  {clip_key}: {len(refined)} frames, "
              f"mean={refined.mean():.4f}, max={refined.max():.4f}")

    print(f"\nSaved refined scores to {args.output_dir}/")


if __name__ == "__main__":
    main()
