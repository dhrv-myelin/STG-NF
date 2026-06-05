"""Plot per-frame anomaly scores for comparison across methods."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--vids", nargs="+", default=["Vid2", "Vid4"])
    ap.add_argument("--modes", nargs="+", default=["pretrained", "trained"])
    ap.add_argument("--output", default="results/score_comparison.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(args.vids), 1, figsize=(14, 4 * len(args.vids)), squeeze=False)

    for row, vid in enumerate(args.vids):
        ax = axes[row, 0]
        for mode in args.modes:
            npy = os.path.join(args.results_dir, mode, f"0_{vid}_scores.npy")
            if not os.path.exists(npy):
                print(f"  Skipping {npy} (not found)")
                continue
            scores = np.load(npy)
            # Normalize to [0,1] for comparison
            finite = scores[np.isfinite(scores)]
            if len(finite) > 0 and finite.max() - finite.min() > 1e-8:
                scores_norm = (scores - finite.min()) / (finite.max() - finite.min())
            else:
                scores_norm = np.zeros_like(scores)
            frames = np.arange(len(scores_norm))
            ax.plot(frames, scores_norm, label=mode, alpha=0.8)

        ax.set_title(f"{vid} - Anomaly Scores (normalized)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Anomaly Score")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
