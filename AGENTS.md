# AGENTS.md

## Environment and setup
- `conda env create -f environment.yml && conda activate STG-NF`
- Modern stack: Python 3.10, PyTorch + CUDA 12.1, ultralytics (for YOLO-pose).
- No repo-level lint/format/typecheck/test config exists; do not invent those commands.
- Data is external (Google Drive link in `README.md`) and expected under `data/`.

## Real entrypoints
- Main training/eval: `train_eval.py`
- Per-frame score export (no GT required): `scripts/test_qualitative.py`
- YOLO-pose + ByteTrack extraction (modern, recommended): `scripts/run_yolo_pose.py`
- Legacy AlphaPose extraction: `scripts/run_alphapose.py` or `gen_data.py`

## Train/eval behavior that is easy to miss
- `--checkpoint` implies test-only flow (`train_eval.py` builds loader with `only_test=True`).
- AUC is only computed when GT directories exist:
  - UBnormal: `data/UBnormal/gt/`
  - ShanghaiTech/ShanghaiTech-HR: `data/ShanghaiTech/gt/test_frame_mask/`
- Without GT, `train_eval.py` still runs test and prints segment score shape only.
- Default `--seg_len` is `24`; UBnormal eval with provided checkpoints must pass `--seg_len 16`.
- Test loader always uses stride 1 (dense sliding window), regardless of `--seg_stride`.
- `--R` controls the prior radius: default 3.0. Supervised UBnormal checkpoint requires `--R 10`.
- `--seed 999` (default) means use a random seed (`torch.initial_seed()`), not a fixed seed.

## Dataset/CLI constraints
- `args.py` restricts `--dataset` to `ShanghaiTech`, `ShanghaiTech-HR`, `UBnormal`.
- `train_eval.py --dataset MyDataset` is invalid unless `args.py` is changed.
- For custom pose data without GT, use `scripts/test_qualitative.py` for per-frame outputs.
- `--layout` controls keypoint schema: `alphapose` (17), `openpose` (18), `ntu-rgb+d` (25).
- Custom dataset filenames are used as clip IDs after stripping `_alphapose_tracked_person.json`.
- UBnormal filename regex for metadata parsing: `(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*` (`dataset.py:178`).
- ShanghaiTech-HR skips 6 specific broken clips defined in `SHANGHAITECH_HR_SKIP` (`dataset.py:12`).

## Input pipeline (model)
- Model input shape `(B, 2, T, V)` — the confidence channel is always stripped before `model.forward()`.
- When `--layout` is not `alphapose`, 17-keypoint data is converted to 18-keypoint COCO format via `keypoints17_to_coco18()` (`dataset.py:219-222`, `dataset.py:245-257`).
- Normalization in `utils/data_utils.py:normalize_pose()`:
  1. Divide x,y by `[856, 480]` (default `--vid_res`), leave confidence as-is.
  2. Subtract per-sample mean of (x,y) over all frames and keypoints.
  3. Divide by per-sample std of y over all frames and keypoints.
- Segmentation is sliding-window: train stride = `--seg_stride` (default 6), test stride = 1.

## AlphaPose integration (legacy)
- AlphaPose cloned at `./AlphaPose/`. Requires weight files (not in repo).
- CUDA 10.2 cannot compile AlphaPose's CUDA/Cython extensions on Ampere+ GPUs.
  Two source files patched to use `torchvision.ops` as a fallback (see `doc_changes/alphapose_fallback_patches.md`).
- If AlphaPose still fails, run `AlphaPose/scripts/demo_inference.py` directly to isolate issues.

## YOLO-pose pipeline (modern, recommended)
- `scripts/run_yolo_pose.py` uses Ultralytics YOLO-pose + built-in ByteTrack.
- Writes the same `_alphapose_tracked_person.json` schema — no model code changes needed.
- Model auto-downloads on first use (e.g. `yolo11x-pose.pt`).
- Outputs COCO 17-keypoint format, compatible with `--layout alphapose`.

## Useful commands
```bash
# ShanghaiTech eval
python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar

# UBnormal unsupervised eval
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar

# UBnormal supervised eval
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar

# YOLO-pose batch extraction (modern)
python scripts/run_yolo_pose.py --input_dir <videos_dir> --output_dir <pose_json_out> --model yolo11x-pose.pt

# Legacy AlphaPose batch extraction
python scripts/run_alphapose.py --input_dir <videos_dir> --output_dir <pose_json_out> --alphapose_dir ./AlphaPose/

# Qualitative per-frame score export (no GT required)
python scripts/test_qualitative.py --checkpoint <ckpt.tar> --pose_path_test <pose_test_dir> --layout alphapose --output_dir results/
```
