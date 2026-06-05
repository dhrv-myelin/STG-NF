"""
Plot per-frame anomaly scores with theft region shading.

Supports 2 or 3 score directories for comparison.

Usage:
    # 2-way (refined vs amplified):
    python scripts/plot_scores.py \
        --dirs results/refined results/amplified \
        --labels "Refined (z-score)" "Amplified" \
        --output results/score_comparison.png

    # 3-way (raw vs refined vs amplified):
    python scripts/plot_scores.py \
        --dirs results/raw results/refined results/amplified \
        --labels "Raw NLL" "Refined (z-score)" "Amplified" \
        --output results/score_comparison.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]


def normalize_scores(scores):
    finite = scores[np.isfinite(scores)]
    if len(finite) > 0 and finite.max() - finite.min() > 1e-8:
        return (scores - finite.min()) / (finite.max() - finite.min())
    return np.zeros_like(scores)


def main():
    ap = argparse.ArgumentParser(description="Plot anomaly scores with theft region shading")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Score directories to compare (e.g. results/raw results/refined results/amplified)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Labels for each directory (auto-generated if not provided)")
    ap.add_argument("--vids", nargs="+", default=["0_Vid1", "0_Vid2", "0_Vid3", "0_Vid4"])
    ap.add_argument("--output", default="results/score_comparison.png")
    ap.add_argument("--no_norm", action="store_true", help="Plot raw values instead of [0,1]")
    args = ap.parse_args()

    n_dirs = len(args.dirs)
    if args.labels is None:
        args.labels = [os.path.basename(d) for d in args.dirs]
    assert len(args.labels) == n_dirs

    n_vids = len(args.vids)
    fig, axes = plt.subplots(n_vids, 1, figsize=(16, 3.5 * n_vids), squeeze=False)

    for row, clip_key in enumerate(args.vids):
        ax = axes[row, 0]
        vid_label = VID_LABELS.get(clip_key, clip_key)

        for d_idx, (d, label) in enumerate(zip(args.dirs, args.labels)):
            npy = os.path.join(d, f"{clip_key}_scores.npy")
            if not os.path.exists(npy):
                print(f"  Skipping {clip_key} {label}: not found at {npy}")
                continue

            scores = np.load(npy)
            frames = np.arange(len(scores))
            plot_scores = scores if args.no_norm else normalize_scores(scores)

            color = COLORS[d_idx % len(COLORS)]
            lw = 1.5 if d_idx == n_dirs - 1 else 1.0
            alpha = 0.95 if d_idx == n_dirs - 1 else 0.6
            ax.plot(frames, plot_scores, color=color, linewidth=lw, alpha=alpha, label=label)

        # Shade theft region
        if clip_key in THEFT_TIMINGS:
            ts, te = THEFT_TIMINGS[clip_key]
            ax.axvspan(ts, te, alpha=0.12, color="#F44336")
            ax.axvline(ts, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.4)
            ax.axvline(te, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.4)
            # Label the theft region
            mid = (ts + te) // 2
            ax.text(mid, 1.02, "THEFT", ha="center", va="bottom",
                    fontsize=8, color="#F44336", fontweight="bold", alpha=0.7)

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
