# AlphaPose CUDA Extension Fallback Patches

## Problem

AlphaPose requires compiled CUDA/Cython extensions that cannot be built with
CUDA 10.2 (pinned by `environment.yml`) on modern GPUs (Ampere+, SM 86+).

The following five compiled extensions are defined in `setup.py`:

- `detector.nms.nms_cpu` (C++)
- `detector.nms.nms_cuda` (CUDA)
- `detector.nms.soft_nms_cpu` (Cython)
- `alphapose.utils.roi_align.roi_align_cuda` (CUDA)
- `alphapose.models.layers.dcn.deform_conv_cuda` (CUDA)
- `alphapose.models.layers.dcn.deform_pool_cuda` (CUDA)

## Which imports actually crash at runtime

Only two files are **unconditionally** imported at module level when running
`demo_inference.py` with the default config (`256x192_res152_lr1e-3_1x-duc.yaml`):

| File | Import | Type |
|------|--------|------|
| `detector/nms/nms_wrapper.py:4-5` | `nms_cpu`, `nms_cuda`, `soft_nms_cpu` | CUDA + Cython |
| `alphapose/utils/roi_align/roi_align.py:6` | `roi_align_cuda` | CUDA |

The DCN extensions (`deform_conv_cuda`, `deform_pool_cuda`) are imported
**conditionally** inside `if self.with_dcn:` blocks in `SE_Resnet.py:79`,
`Resnet.py:75`, and `ShuffleResnet.py:83/92`. The default config does not
use DCN, so these imports are never triggered.

## Changes made

### 1. `AlphaPose/detector/nms/nms_wrapper.py`

- **Removed** imports of `nms_cpu`, `nms_cuda`, `soft_nms_cpu`
- **Added** `import torchvision.ops` and a pure-Python `_soft_nms_cpu()` function
- **`nms()`**: replaced `nms_cuda.nms()` / `nms_cpu.nms()` with
  `torchvision.ops.nms()`
- **`soft_nms()`**: replaced `soft_nms_cpu()` call with `_soft_nms_cpu()`

### 2. `AlphaPose/alphapose/utils/roi_align/roi_align.py`

- **Removed** `from . import roi_align_cuda`
- **Removed** the entire `RoIAlignFunction` class (CUDA autograd function)
- **Replaced** with `from torchvision.ops import roi_align`
- **Simplified** `RoIAlign.forward()` to always use `torchvision.ops.roi_align`
  (the `use_torchvision` path was already present but unreachable)

## Files NOT changed (but verified safe)

| File | Why safe |
|------|----------|
| `alphapose/models/layers/dcn/deform_conv.py` | Import guarded by `if self.with_dcn:` — default config has `DCN: null` |
| `alphapose/models/layers/dcn/deform_pool.py` | Same — only imported when DCN config is active |
| `alphapose/models/layers/dcn/__init__.py` | Only loaded if `from .dcn import ...` is reached inside a DCN-enabled model |

## Additional setup required

### Pretrained model checkpoint

```bash
# Download fast_421_res152_256x192.pth from AlphaPose model zoo
# Place at: AlphaPose/pretrained_models/fast_421_res152_256x192.pth
```

### YOLO detector weights

```bash
wget -P AlphaPose/detector/yolo/data/ \
  https://pjreddie.com/media/files/yolov3-spp.weights
```

## Usage

```bash
conda activate STG-NF
python scripts/run_alphapose.py \
  --input_dir /path/to/videos \
  --output_dir data/pose/ \
  --alphapose_dir ./AlphaPose/
```

The wrapper already sets `ALPHAPOSE_PURE_PY_FALLBACK=1` in the subprocess
environment and runs without compiled extensions.
