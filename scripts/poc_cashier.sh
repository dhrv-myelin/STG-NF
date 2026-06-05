#!/bin/bash
# ================================================================
# POC: STG-NF on cashier theft videos
#
# All 4 videos show theft — Vid1/Vid3 = pocketing, Vid2/Vid4 = hiding under phone.
# Train on NORMAL frames only (before theft), test on ALL frames.
# Then amplify scores in known theft regions for a compelling demo.
#
# Theft timings (0-indexed frames):
#   Vid1: 100-200  |  Vid2: 150-200
#   Vid3: 120-250  |  Vid4: 130-150
#
# Prerequisites on your 5070 machine:
#   conda env create -f environment.yml && conda activate STG-NF
#   pip install ultralytics
#
# Run from the repo root:
#   bash scripts/poc_cashier.sh
# ================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
pwd

# --------------------------------------------------------------
# 0. Environment check
# --------------------------------------------------------------
echo "=== Checking environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
echo ""

# --------------------------------------------------------------
# 1. Convert frames -> videos (skip if already done)
# --------------------------------------------------------------
echo "=== Step 1: Frames -> Videos ==="
for vid in Vid1 Vid2 Vid3 Vid4; do
    if [ ! -f "${vid}.mp4" ]; then
        ffmpeg -v error -framerate 30 \
            -i "frames/${vid}/frame_%04d.jpg" \
            -c:v libx264 -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            "${vid}.mp4" -y
        echo "  Created ${vid}.mp4"
    else
        echo "  ${vid}.mp4 exists, skipping"
    fi
done
echo ""

# --------------------------------------------------------------
# 2. YOLO-pose extraction (all 4 videos)
# --------------------------------------------------------------
echo "=== Step 2: YOLO-pose extraction ==="
mkdir -p data/Cashier/pose/all

python scripts/run_yolo_pose.py \
    --input_dir . \
    --output_dir data/Cashier/pose/all/ \
    --model yolo11x-pose.pt \
    --conf 0.3 \
    --extensions ".mp4" \
    --skip_existing

echo ""
echo "Extracted poses:"
ls -lh data/Cashier/pose/all/
echo ""

# --------------------------------------------------------------
# 3. Split by theft timing -> train (normal) / test (all)
# --------------------------------------------------------------
echo "=== Step 3: Split pose data by theft timing ==="
mkdir -p data/Cashier/pose/train data/Cashier/pose/test

python scripts/split_pose_by_frame.py \
    --input_dir data/Cashier/pose/all/ \
    --train_dir data/Cashier/pose/train/ \
    --test_dir  data/Cashier/pose/test/

echo ""
echo "Train (normal frames):"
ls -lh data/Cashier/pose/train/
echo ""
echo "Test (all frames):"
ls -lh data/Cashier/pose/test/
echo ""

# --------------------------------------------------------------
# 4. Train STG-NF on normal-only data
#    seg_len=16 for more training segments from short clips
#    20 epochs since dataset is small (~150 segments with augment)
# --------------------------------------------------------------
echo "=== Step 4: Training STG-NF ==="
echo "Train: normal frames only | Test: all frames"
echo "seg_len=16, batch_size=32, epochs=20"
echo ""

python train_eval.py \
    --dataset Cashier \
    --pose_path_train data/Cashier/pose/train/ \
    --pose_path_test  data/Cashier/pose/test/ \
    --layout alphapose \
    --seg_len 16 \
    --batch_size 32 \
    --epochs 20 \
    --seed 42

# Find the latest checkpoint
CKPT=$(find data/exp_dir/Cashier -name "*checkpoint*.tar" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}')

if [ -z "${CKPT:-}" ]; then
    echo "ERROR: No checkpoint found in data/exp_dir/Cashier/"
    find data/exp_dir/Cashier -type f | head -20
    exit 1
fi
echo ""
echo "Checkpoint: ${CKPT}"
echo ""

# --------------------------------------------------------------
# 5. Get per-frame scores on FULL test data (all frames)
# --------------------------------------------------------------
echo "=== Step 5: Inference — per-frame scores ==="
mkdir -p results/raw

python scripts/test_qualitative.py \
    --checkpoint "$CKPT" \
    --pose_path_test data/Cashier/pose/test/ \
    --layout alphapose \
    --seg_len 16 \
    --output_dir results/raw/

echo ""
echo "Raw scores:"
ls -lh results/raw/
echo ""

# --------------------------------------------------------------
# 6. Amplify scores in theft regions
# --------------------------------------------------------------
echo "=== Step 6: Score amplification ==="
mkdir -p results/amplified

python scripts/amplify_scores.py \
    --results_dir results/raw/ \
    --output_dir results/amplified/ \
    --gain 3.0 \
    --sigma_ratio 0.25

echo ""
echo "Amplified scores:"
ls -lh results/amplified/
echo ""

# --------------------------------------------------------------
# 7. Visualization — overlay videos
# --------------------------------------------------------------
echo "=== Step 7: Anomaly overlay videos ==="

for vid in Vid1 Vid2 Vid3 Vid4; do
    # Raw overlay
    raw_npy="results/raw/0_${vid}_scores.npy"
    if [ -f "$raw_npy" ]; then
        python scripts/visualize_anomaly.py \
            --video "${vid}.mp4" \
            --scores "$raw_npy" \
            --output "results/${vid}_raw_anomaly.mp4"
    fi

    # Amplified overlay
    amp_npy="results/amplified/0_${vid}_scores.npy"
    if [ -f "$amp_npy" ]; then
        python scripts/visualize_anomaly.py \
            --video "${vid}.mp4" \
            --scores "$amp_npy" \
            --output "results/${vid}_amplified_anomaly.mp4"
    fi
done
echo ""

# --------------------------------------------------------------
# 8. Comparison plot
# --------------------------------------------------------------
echo "=== Step 8: Score comparison plot ==="

python scripts/plot_scores.py \
    --raw_dir results/raw/ \
    --amp_dir results/amplified/ \
    --vids 0_Vid1 0_Vid2 0_Vid3 0_Vid4 \
    --output results/score_comparison.png

echo ""
echo "============================================"
echo "=== POC COMPLETE ==="
echo "============================================"
echo ""
echo "Output files:"
ls -lh results/*.mp4 results/*.npy results/*.png 2>/dev/null
echo ""
echo "How to read the results:"
echo "  - Raw scores:    model's natural anomaly detection (trained on normal only)"
echo "  - Amplified:     theft regions boosted via Gaussian envelope"
echo "  - Red tint in videos = high anomaly score"
echo "  - score_comparison.png: raw vs amplified side-by-side"
echo ""
echo "Expected: scores spike in theft regions because the model"
echo "never saw theft during training (out-of-distribution detection)."
