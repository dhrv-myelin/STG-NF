"""
Plot per-frame anomaly scores: raw vs amplified, with theft region shading.

Usage:
    python scripts/plot_scores.py \
        --raw_dir results/raw \
        --amp_dir results/amplified \
        --output  results/score_comparison.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 0-indexed theft regions for shading
THEFT_TIMINGS = {
    "0_Vid1": (100, 200),
    "0_Vid2": (150, 200),
    "0_Vid3": (120, 250),
    "0_Vid4": (130, 150),
}

VID_LABELS = {
    "0_Vid1": "Vid1 (pocketing)",
    "0_Vid2": "Vid2 (hiding under phone)",
    "0_Vid3": "Vid3 (pocketing)",
    "0_Vid4": "Vid4 (hiding under phone)",
}


def normalize_scores(scores):
    finite = scores[np.isfinite(scores)]
    if len(finite) > 0 and finite.max() - finite.min() > 1e-8:
        return (scores - finite.min()) / (finite.max() - finite.min())
    return np.zeros_like(scores)


def main():
    ap = argparse.ArgumentParser(description="Plot raw vs amplified anomaly scores")
    ap.add_argument("--raw_dir", default="results/raw", help="Dir with raw *_scores.npy")
    ap.add_argument("--amp_dir", default="results/amplified", help="Dir with amplified *_scores.npy")
    ap.add_argument("--vids", nargs="+", default=["0_Vid1", "0_Vid2", "0_Vid3", "0_Vid4"])
    ap.add_argument("--output", default="results/score_comparison.png")
    ap.add_argument("--no_norm", action="store_true", help="Plot raw values instead of normalizing to [0,1]")
    args = ap.parse_args()

    n_vids = len(args.vids)
    fig, axes = plt.subplots(n_vids, 1, figsize=(16, 3.5 * n_vids), squeeze=False)

    for row, clip_key in enumerate(args.vids):
        ax = axes[row, 0]
        vid_label = VID_LABELS.get(clip_key, clip_key)

        # Load raw scores
        raw_path = os.path.join(args.raw_dir, f"{clip_key}_scores.npy")
        amp_path = os.path.join(args.amp_dir, f"{clip_key}_scores.npy")

        if not os.path.exists(raw_path):
            print(f"  Skipping {clip_key}: raw scores not found at {raw_path}")
            continue

        raw = np.load(raw_path)
        frames = np.arange(len(raw))

        # Plot raw scores
        raw_plot = raw if args.no_norm else normalize_scores(raw)
        ax.plot(frames, raw_plot, color="#2196F3", linewidth=1.2, alpha=0.8, label="Raw score")

        # Plot amplified scores
        if os.path.exists(amp_path):
            amp = np.load(amp_path)
            amp_plot = amp if args.no_norm else normalize_scores(amp)
            ax.plot(frames, amp_plot, color="#F44336", linewidth=1.5, alpha=0.9, label="Amplified score")

        # Shade theft region
        if clip_key in THEFT_TIMINGS:
            ts, te = THEFT_TIMINGS[clip_key]
            ax.axvspan(ts, te, alpha=0.15, color="#F44336", label=f"Theft frames ({ts}-{te})")
            ax.axvline(ts, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(te, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_title(vid_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Anomaly Score")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(-0.05, 1.10 if not args.no_norm else None)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
