# AGENTS.md

## Environment and setup (verified)
- Use the pinned environment: `conda env create -f environment.yml && conda activate STG-NF`
- Toolchain is old/pinned: Python 3.8, PyTorch 1.10.1, CUDA 10.2 (`environment.yml`).
- No repo-level lint/format/typecheck/test config exists; do not invent those commands.
- Data is external (Google Drive link in `README.md`) and expected under `data/`.

## Real entrypoints
- Main training/eval entrypoint: `train_eval.py`
- Qualitative per-frame score export: `scripts/test_qualitative.py`
- AlphaPose extraction wrapper used by this repo: `scripts/run_alphapose.py`

## Train/eval behavior that is easy to miss
- `--checkpoint` implies test-only flow in `train_eval.py` (loader built with `only_test=True`).
- AUC is only computed when GT directories exist:
  - UBnormal: `data/UBnormal/gt/`
  - ShanghaiTech/ShanghaiTech-HR: `data/ShanghaiTech/gt/test_frame_mask/`
- Without GT, `train_eval.py` still runs test and prints segment score shape only.
- Default `--seg_len` is `24`; UBnormal eval with provided checkpoints must pass `--seg_len 16`.

## Dataset/CLI constraints
- `args.py` restricts `--dataset` to `ShanghaiTech`, `ShanghaiTech-HR`, `UBnormal`.
- `train_eval.py --dataset MyDataset` is invalid unless code is changed.
- For custom pose data without GT, use `scripts/test_qualitative.py` for per-frame outputs.
- `--layout` controls keypoint schema: `alphapose` (17), `openpose` (18), `ntu-rgb+d` (25).
- Custom dataset filenames are used as clip IDs after stripping `_alphapose_tracked_person.json`.

## AlphaPose integration notes in this repo
- This repo expects a separate AlphaPose clone at `./AlphaPose`.
- Required pose checkpoint: `AlphaPose/pretrained_models/fast_421_res152_256x192.pth`.
- YOLO default detector requires `AlphaPose/detector/yolo/data/yolov3-spp.weights`.
- `scripts/run_alphapose.py` now:
  - uses `sys.executable`,
  - normalizes paths with `abspath`,
  - sets `PYTHONPATH=<alphapose_dir>`,
  - sets `ALPHAPOSE_PURE_PY_FALLBACK=1`,
  - prints full subprocess stderr.
- Fallback patches for missing AlphaPose extensions are isolated in:
  - `AlphaPose/alphapose/utils/roi_align/roi_align.py`
  - `AlphaPose/detector/nms/nms_wrapper.py`

## Known local gotchas
- On newer NVIDIA GPUs (e.g. RTX 30xx), this pinned PyTorch/CUDA stack can show unsupported SM warnings; CPU fallback (`--gpus -1` in direct AlphaPose commands) may be required.
- If AlphaPose fails, run `AlphaPose/scripts/demo_inference.py` directly first to isolate detector/model issues before rerunning `scripts/run_alphapose.py`.

## Useful commands
```bash
# ShanghaiTech eval
python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar

# UBnormal eval
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar

# Batch pose extraction via repo wrapper
python scripts/run_alphapose.py --input_dir <videos_dir> --output_dir <pose_json_out> --alphapose_dir ./AlphaPose/

# Qualitative per-frame score export (no GT required)
python scripts/test_qualitative.py --checkpoint <ckpt.tar> --pose_path_test <pose_test_dir> --layout alphapose --output_dir results/
```
