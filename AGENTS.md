# AGENTS.md

## Environment and setup (verified)
- Use the pinned environment: `conda env create -f environment.yml && conda activate STG-NF`
- Toolchain is old/pinned: Python 3.8, PyTorch 1.10.1, CUDA 10.2 (`environment.yml`).
- No repo-level lint/format/typecheck/test config exists; do not invent those commands.
- Data is external (Google Drive link in `README.md`) and expected under `data/`.

## Real entrypoints
- Main training/eval: `train_eval.py`
- Per-frame score export (no GT required): `scripts/test_qualitative.py`
- AlphaPose batch extraction: `scripts/run_alphapose.py`
- Legacy AlphaPose extraction: `gen_data.py`

## Train/eval behavior that is easy to miss
- `--checkpoint` implies test-only flow (`train_eval.py` builds loader with `only_test=True`).
- AUC is only computed when GT directories exist:
  - UBnormal: `data/UBnormal/gt/`
  - ShanghaiTech/ShanghaiTech-HR: `data/ShanghaiTech/gt/test_frame_mask/`
- Without GT, `train_eval.py` still runs test and prints segment score shape only.
- Default `--seg_len` is `24`; UBnormal eval with provided checkpoints must pass `--seg_len 16`.
- Test loader always uses stride 1 (dense sliding window), regardless of `--seg_stride`.
- `--R` controls the prior radius: default 3.0. Supervised UBnormal checkpoint requires `--R 10`.

## Dataset/CLI constraints
- `args.py` restricts `--dataset` to `ShanghaiTech`, `ShanghaiTech-HR`, `UBnormal`.
- `train_eval.py --dataset MyDataset` is invalid unless code is changed.
- For custom pose data without GT, use `scripts/test_qualitative.py` for per-frame outputs.
- `--layout` controls keypoint schema: `alphapose` (17), `openpose` (18), `ntu-rgb+d` (25).
- Custom dataset filenames are used as clip IDs after stripping `_alphapose_tracked_person.json`.

## Preprocessing pipeline (model input)
The model receives `(B, 2, T, V)` where `2 = (x, y)` — confidence is always stripped before
`model.forward()`. Segmentation is sliding-window: train stride = `--seg_stride` (default 6),
test stride = 1. Normalization in `utils/data_utils.py:normalize_pose()`:
1. Divide x,y by `[856, 480]` (default `--vid_res`), leave confidence as-is.
2. Subtract per-sample mean of (x,y) over all frames and keypoints.
3. Divide by per-sample std of y over all frames and keypoints.

## AlphaPose integration
- AlphaPose is cloned at `./AlphaPose/`. Required weight files (not in repo):
  - `AlphaPose/pretrained_models/fast_421_res152_256x192.pth`
  - `AlphaPose/detector/yolo/data/yolov3-spp.weights`
- CUDA 10.2 cannot compile AlphaPose's CUDA/Cython extensions on Ampere+ GPUs.
  Two source files have been patched to use `torchvision.ops` as a fallback:
  - `AlphaPose/detector/nms/nms_wrapper.py` — replaced with `torchvision.ops.nms`
  - `AlphaPose/alphapose/utils/roi_align/roi_align.py` — replaced with `torchvision.ops.roi_align`
  See `doc_changes/alphapose_fallback_patches.md` for details.
- `scripts/run_alphapose.py` sets `ALPHAPOSE_PURE_PY_FALLBACK=1` in subprocess env
  (no-op — AlphaPose does not read this variable; the patches do the work).
- If AlphaPose still fails, run `AlphaPose/scripts/demo_inference.py` directly first
  to isolate detector/model issues before rerunning the wrapper.

## Useful commands
```bash
# ShanghaiTech eval
python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar

# UBnormal unsupervised eval
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar

# UBnormal supervised eval
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar

# Batch pose extraction via repo wrapper
python scripts/run_alphapose.py --input_dir <videos_dir> --output_dir <pose_json_out> --alphapose_dir ./AlphaPose/

# Qualitative per-frame score export (no GT required)
python scripts/test_qualitative.py --checkpoint <ckpt.tar> --pose_path_test <pose_test_dir> --layout alphapose --output_dir results/
```
