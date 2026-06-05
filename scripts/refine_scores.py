"""
Refine raw anomaly scores via local z-score normalization.

Improvements over v1:
  - Multi-scale z-score: compute at windows [50, 100, 200, 400], take max
  - Robust percentile normalization: clip outliers via p99 instead of max

Usage:
    python scripts/refine_scores.py \
        --results_dir results/raw/ \
        --output_dir results/refined/ \
        --smooth_sigma 3
"""
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


def local_zscore(scores, window=100, eps=1e-8):
    """Compute local z-score for each frame over a single window size."""
    n = len(scores)
    z = np.zeros(n, dtype=np.float64)
    half_w = window // 2

    # Vectorized: precompute rolling mean and std via cumulative sums
    cs = np.cumsum(scores)
    cs2 = np.cumsum(scores ** 2)

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w)
        cnt = hi - lo
        local_sum = cs[hi - 1] - (cs[lo - 1] if lo > 0 else 0)
        local_sum2 = cs2[hi - 1] - (cs2[lo - 1] if lo > 0 else 0)
        local_mean = local_sum / cnt
        local_var = local_sum2 / cnt - local_mean ** 2
        local_std = np.sqrt(max(local_var, 0.0))
        z[i] = (scores[i] - local_mean) / (local_std + eps)

    return z


def multi_scale_zscore(scores, windows=(50, 100, 200, 400), eps=1e-8):
    """Compute z-score at multiple scales, take max at each frame.

    Short anomalies (a few seconds) are caught by small windows.
    Long anomalies (tens of seconds) are caught by large windows.
    """
    all_z = []
    for w in windows:
        z = local_zscore(scores, window=w, eps=eps)
        all_z.append(z)
    return np.max(all_z, axis=0)


def rolling_percentile_baseline(scores, window=200, percentile=10):
    """Rolling percentile baseline — tracks 'normal' level."""
    n = len(scores)
    baseline = np.zeros(n, dtype=np.float64)
    half_w = window // 2

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w)
        baseline[i] = np.percentile(scores[lo:hi], percentile)

    return scores - baseline


def main():
    ap = argparse.ArgumentParser(description="Refine anomaly scores via multi-scale z-score")
    ap.add_argument("--results_dir", default="results/raw", help="Dir with raw *_scores.npy")
    ap.add_argument("--output_dir", default="results/refined", help="Output dir for refined scores")
    ap.add_argument("--smooth_sigma", type=float, default=3.0,
                    help="Light Gaussian smoothing on refined scores (default: 3)")
    ap.add_argument("--method", choices=["zscore", "baseline", "both"], default="both",
                    help="zscore = multi-scale z-score, baseline = rolling percentile, both = max")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(args.results_dir) if f.endswith("_scores.npy"))

    for nf in npy_files:
        clip_key = nf.replace("_scores.npy", "")
        scores = np.load(os.path.join(args.results_dir, nf))

        finite = scores[np.isfinite(scores)]
        if len(finite) == 0:
            print(f"  Skipping {clip_key} (all scores are inf/nan)")
            continue

        working = scores.copy()
        working[~np.isfinite(working)] = np.median(finite)

        results = {}

        if args.method in ("zscore", "both"):
            z = multi_scale_zscore(working)
            z_shifted = z - z.min()
            if args.smooth_sigma > 0:
                z_shifted = gaussian_filter1d(z_shifted, sigma=args.smooth_sigma)
            results["zscore"] = z_shifted

        if args.method in ("baseline", "both"):
            b = rolling_percentile_baseline(working, window=400)
            b_neg = -b
            b_shifted = b_neg - b_neg.min()
            if args.smooth_sigma > 0:
                b_shifted = gaussian_filter1d(b_shifted, sigma=args.smooth_sigma)
            results["baseline"] = b_shifted

        if args.method == "both":
            refined = np.maximum(results["zscore"], results["baseline"])
        elif args.method == "zscore":
            refined = results["zscore"]
        else:
            refined = results["baseline"]

        # Robust normalization: clip at 99th percentile instead of max
        positive = refined[refined > 0]
        if len(positive) > 0:
            p99 = np.percentile(positive, 99)
        else:
            p99 = 1.0
        refined = np.clip(refined / (p99 + 1e-8), 0.0, 1.0)

        out_path = os.path.join(args.output_dir, f"{clip_key}_scores.npy")
        np.save(out_path, refined)

        print(f"  {clip_key}: {len(refined)} frames, "
              f"mean={refined.mean():.4f}, max={refined.max():.4f}, "
              f"p99={p99:.4f}")

    print(f"\nSaved refined scores to {args.output_dir}/")


if __name__ == "__main__":
    main()
