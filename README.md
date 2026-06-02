# Normalizing Flows for Human Pose Anomaly Detection

Fork of [STG-NF](https://github.com/orhir/STG-NF) (ICCV 2023) by Hirschorn & Avidan.

## Setup

```bash
conda env create -f environment.yml
conda activate STG-NF
```

## Quickstart — 3-Step Workflow

### Step 1: Extract poses from videos

```bash
python scripts/run_yolo_pose.py --input_dir <VIDEOS_DIR> --output_dir <POSE_DIR>
```

Outputs `<videostem>_alphapose_tracked_person.json` files (one per video). Place train/test JSONs in separate directories.

### Step 2: Train STG-NF

```bash
python train_eval.py \
    --dataset MyDataset \
    --pose_path_train ./data/pose/train/ \
    --pose_path_test ./data/pose/test/ \
    --layout alphapose \
    --seg_len 24 \
    --batch_size 256 \
    --epochs 8
```

The checkpoint is saved under `data/exp_dir/MyDataset/<timestamp>/`.

### Step 3: Visualize anomaly scores on a test video

First export per-frame scores:

```bash
python scripts/test_qualitative.py \
    --checkpoint data/exp_dir/MyDataset/<timestamp>/<checkpoint>.tar \
    --pose_path_test ./data/pose/test/ \
    --layout alphapose \
    --seg_len 24 \
    --output_dir results/
```

Then render the overlay video:

```bash
python scripts/visualize_anomaly.py \
    --video ./data/videos/test_video.mp4 \
    --scores results/0_<videostem>_scores.npy \
    --output anomaly_result.mp4
```

## Repository Structure

```
.
├── args.py                        CLI arguments and parser
├── train_eval.py                  Main training / evaluation entrypoint
├── dataset.py                     Dataset loader (JSON pose → segments)
│
├── models/
│   ├── training.py                Trainer class (train loop, test loop, checkpointing)
│   └── STG_NF/
│       ├── model_pose.py          STG_NF model (normalizing flow on pose keypoints)
│       ├── modules_pose.py        Flow layers: actnorm, invertible 1×1, affine, split, squeeze
│       ├── stgcn.py               Spatio-temporal graph convolution (ST-GCN) blocks
│       ├── tgcn.py                Temporal graph convolution (invertible)
│       ├── graph.py               Graph construction & adjacency matrix (alphapose/openpose/ntu)
│       └── utils.py               Padding & tensor split helpers
│
├── utils/
│   ├── pose_utils.py              JSON parsing, keypoint conversion, segment splitting
│   ├── data_utils.py              Pose normalization & data augmentation transforms
│   ├── scoring_utils.py           AUC computation & segment-to-frame score aggregation
│   ├── train_utils.py             Model parameter init, arg dumping, param counting
│   └── optim_init.py              Optimizer & scheduler factory
│
└── scripts/
    ├── run_yolo_pose.py           Pose extraction via YOLO-pose + ByteTrack
    ├── test_qualitative.py        Export per-frame anomaly scores (no GT required)
    └── visualize_anomaly.py       Overlay scores on a video as a color-coded timeline
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_yolo_pose.py` | Pose extraction via YOLO-pose + ByteTrack |
| `scripts/test_qualitative.py` | Export per-frame anomaly scores (no GT required) |
| `scripts/visualize_anomaly.py` | Overlay scores on a video as a color-coded timeline |

## Citation

If you use this work, cite the original paper:

```
@InProceedings{Hirschorn_2023_ICCV,
    author    = {Hirschorn, Or and Avidan, Shai},
    title     = {Normalizing Flows for Human Pose Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13545-13554}
}
```
