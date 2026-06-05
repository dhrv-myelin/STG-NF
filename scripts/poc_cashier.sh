#!/bin/bash
# ================================================================
# POC: STG-NF on cashier anomaly videos
# Vid1, Vid3 = "normal" (baseline behavior)
# Vid2, Vid4 = anomalous (hiding cash under phone)
#
# Prerequisites on your 5070 machine:
#   conda env create -f environment.yml && conda activate STG-NF
#   pip install ultralytics   # for YOLO-pose
#
# Run from the repo root:
#   bash scripts/poc_cashier.sh
# ================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
pwd

# --------------------------------------------------------------
# 0. Check prerequisites
# --------------------------------------------------------------
echo "=== Checking environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
echo ""

# --------------------------------------------------------------
# 1. Convert frames -> videos (if not already done)
# --------------------------------------------------------------
echo "=== Step 1: Converting frames to videos ==="
for vid in Vid1 Vid2 Vid3 Vid4; do
    if [ ! -f "${vid}.mp4" ]; then
        echo "  Creating ${vid}.mp4 ..."
        ffmpeg -v error -framerate 30 \
            -i "frames/${vid}/frame_%04d.jpg" \
            -c:v libx264 -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            "${vid}.mp4" -y
    else
        echo "  ${vid}.mp4 already exists, skipping."
    fi
done
echo ""

# --------------------------------------------------------------
# 2. Extract poses with YOLO-pose + ByteTrack
# --------------------------------------------------------------
echo "=== Step 2: YOLO-pose extraction ==="
mkdir -p data/Cashier/pose/train data/Cashier/pose/test

# Extract all 4 videos at once into a staging dir
mkdir -p data/Cashier/pose/_staging
python scripts/run_yolo_pose.py \
    --input_dir . \
    --output_dir data/Cashier/pose/_staging/ \
    --model yolo11x-pose.pt \
    --conf 0.3 \
    --extensions ".mp4" \
    --skip_existing

# Split into train/test
for vid in Vid1 Vid3; do
    mv "data/Cashier/pose/_staging/${vid}_alphapose_tracked_person.json" \
       "data/Cashier/pose/train/" 2>/dev/null && echo "  ${vid} -> train" || echo "  ${vid} already in train"
done
for vid in Vid2 Vid4; do
    mv "data/Cashier/pose/_staging/${vid}_alphapose_tracked_person.json" \
       "data/Cashier/pose/test/" 2>/dev/null && echo "  ${vid} -> test" || echo "  ${vid} already in test"
done
rm -rf data/Cashier/pose/_staging

echo ""
echo "Train files:"
ls -lh data/Cashier/pose/train/
echo ""
echo "Test files:"
ls -lh data/Cashier/pose/test/
echo ""

# --------------------------------------------------------------
# 3. Pre-trained checkpoint inference (ShanghaiTech)
#    Trained on pedestrian surveillance -- may be noisy for
#    cashier close-ups, but shows out-of-distribution response.
# --------------------------------------------------------------
echo "=== Step 3: Pre-trained checkpoint inference (ShanghaiTech) ==="
mkdir -p results/pretrained

python scripts/test_qualitative.py \
    --checkpoint checkpoints/ShanghaiTech_85_9.tar \
    --pose_path_test data/Cashier/pose/test/ \
    --layout alphapose \
    --seg_len 24 \
    --output_dir results/pretrained/

echo ""
echo "Pretrained per-frame scores:"
ls -lh results/pretrained/
echo ""

# --------------------------------------------------------------
# 4. Train from scratch (Vid1,Vid3 = normal, test on Vid2,Vid4)
# --------------------------------------------------------------
echo "=== Step 4: Training from scratch ==="
echo "Train: Vid1, Vid3 | Test: Vid2, Vid4"
echo "8 epochs, batch_size=64, seg_len=24"
echo ""

python train_eval.py \
    --dataset Cashier \
    --pose_path_train data/Cashier/pose/train/ \
    --pose_path_test data/Cashier/pose/test/ \
    --layout alphapose \
    --seg_len 24 \
    --batch_size 64 \
    --epochs 8 \
    --seed 42

# Find the latest checkpoint
CKPT=$(find data/exp_dir/Cashier -name "*checkpoint*.tar" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}')

if [ -z "${CKPT:-}" ]; then
    echo "ERROR: No checkpoint found in data/exp_dir/Cashier/"
    echo "Contents:"
    find data/exp_dir/Cashier -type f | head -20
    exit 1
fi
echo ""
echo "Using checkpoint: ${CKPT}"

echo ""
echo "=== Step 5: Inference with trained model ==="
mkdir -p results/trained

python scripts/test_qualitative.py \
    --checkpoint "$CKPT" \
    --pose_path_test data/Cashier/pose/test/ \
    --layout alphapose \
    --seg_len 24 \
    --output_dir results/trained/

echo ""
echo "Trained per-frame scores:"
ls -lh results/trained/
echo ""

# --------------------------------------------------------------
# 6. Visualize results
# --------------------------------------------------------------
echo "=== Step 6: Visualization ==="

for mode in pretrained trained; do
    for vid in Vid2 Vid4; do
        npy="results/${mode}/0_${vid}_scores.npy"
        if [ -f "$npy" ]; then
            python scripts/visualize_anomaly.py \
                --video "${vid}.mp4" \
                --scores "$npy" \
                --output "results/${vid}_${mode}_anomaly.mp4"
        else
            echo "  Skipping ${vid} ${mode}: no scores found at ${npy}"
        fi
    done
done

# --------------------------------------------------------------
# 7. Plot comparison
# --------------------------------------------------------------
echo "=== Step 7: Plot score comparison ==="
python scripts/plot_scores.py \
    --results_dir results/ \
    --vids Vid2 Vid4 \
    --modes pretrained trained \
    --output results/score_comparison.png

echo ""
echo "============================================"
echo "=== DONE ==="
echo "============================================"
echo ""
echo "Output files in results/:"
ls -lh results/*.mp4 results/*.npy results/*.png 2>/dev/null
echo ""
echo "How to interpret:"
echo "  - Per-frame scores (.npy): higher = more anomalous"
echo "  - Anomaly videos (*_anomaly.mp4): red tint = high anomaly, green = normal"
echo "  - score_comparison.png: side-by-side plot of both methods"
echo ""
echo "Files:"
echo "  results/Vid2_pretrained_anomaly.mp4  - ShanghaiTech model on Vid2"
echo "  results/Vid4_pretrained_anomaly.mp4  - ShanghaiTech model on Vid4"
echo "  results/Vid2_trained_anomaly.mp4     - Custom model on Vid2"
echo "  results/Vid4_trained_anomaly.mp4     - Custom model on Vid4"
echo "  results/score_comparison.png         - Score comparison plot"
