# AGENTS.md

## Setup
- Python 3.8 + CUDA 10.2 required.
- `conda env create -f environment.yml && conda activate STG-NF`
- Data must be downloaded separately (Google Drive link in README). Extract into `data/` per the directory structure in README.
- No lint, format, typecheck, or test commands exist in this repo.

## Architecture
- Single entrypoint: `train_eval.py`
- `models/STG_NF/model_pose.py` — main model (normalizing flows over ST-GCN blocks)
- `models/training.py` — Trainer class (train/test loops)
- `dataset.py` — `PoseSegDataset` loads pose JSONs, builds sliding-window segments
- `args.py` — all CLI arguments and sub-arg parsing
- `utils/` — pose transforms, scoring (AUC), data normalization

## Key commands

### Standard training/eval (ShanghaiTech / UBnormal)
```
python train_eval.py --dataset ShanghaiTech              # default seg_len=24
python train_eval.py --dataset UBnormal                  # training (seg_len=16 via default)
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar
python train_eval.py --dataset ShanghaiTech      --checkpoint checkpoints/ShanghaiTech_85_9.tar
python train_eval.py --dataset ShanghaiTech-HR   --checkpoint checkpoints/ShanghaiTech_85_9.tar
```

### Custom dataset
```
# 1. Extract poses from videos via AlphaPose
python scripts/run_alphapose.py \
  --input_dir /path/to/train/videos/ \
  --output_dir data/MyDataset/pose/train/ \
  --alphapose_dir /path/to/AlphaPose/

python scripts/run_alphapose.py \
  --input_dir /path/to/test/videos/ \
  --output_dir data/MyDataset/pose/test/ \
  --alphapose_dir /path/to/AlphaPose/

# 2. Train (no GT needed)
python train_eval.py --dataset MyDataset \
  --pose_path_train data/MyDataset/pose/train/ \
  --pose_path_test data/MyDataset/pose/test/ \
  --layout alphapose --epochs 20

# 3. Qualitative evaluation (per-frame anomaly scores, no GT)
python scripts/test_qualitative.py \
  --checkpoint data/exp_dir/MyDataset/<timestamp>/*.tar \
  --pose_path_test data/MyDataset/pose/test/ \
  --layout alphapose --output_dir results/
```

### Custom data generation (legacy)
Requires AlphaPose repo + pretrained models on disk:
```
python gen_data.py --alphapose_dir /path/to/AlphaPose/ --dir /input/dir/ --outdir /output/dir/ [--video]
```

## Non-obvious details
- `--layout` controls skeleton keypoint count: `alphapose` (17 kp, default), `openpose` (18 kp), `ntu-rgb+d` (25 kp).
- For custom datasets, only `--pose_path_train` and `--pose_path_test` are required. `--vid_path_train/test` are optional.
- Custom dataset filenames are used as-is for `clip_id` (strip `_alphapose_tracked_person.json` suffix). No numeric prefix needed.
- No GT required for custom datasets — AUC computation is skipped when GT directory doesn't exist. Use `scripts/test_qualitative.py` for per-frame anomaly score output.
- `--only_test` flag skips training; it's automatically set when `--checkpoint` is provided (see `train_eval.py:35`).