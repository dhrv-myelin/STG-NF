# Normalizing Flows for Human Pose Anomaly Detection [ICCV 2023]
[![arXiv](https://img.shields.io/badge/arXiv-<2211.10946>-<COLOR>.svg)](https://arxiv.org/abs/2211.10946)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=normalizing-flows-for-human-pose-anomaly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-ubnormal)](https://paperswithcode.com/sota/anomaly-detection-on-ubnormal?p=normalizing-flows-for-human-pose-anomaly)



The official PyTorch implementation of the paper [**"Normalizing Flows for Human Pose Anomaly Detection"**](https://arxiv.org/abs/2211.10946).


![Framework_Overview](data/arch.png)

## Citation
If you find this useful, please cite this work as follows:
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

## Getting Started

This code was tested on `Ubuntu 20.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/orhir/STG-NF
cd STG-NF

# Conda environment setup
conda env create -f environment.yml
conda activate STG-NF
```

### Directory Structure
```
.
├── checkpoints
├── data
│   ├── ShanghaiTech
│   │   ├── gt
│   │   │   └── test_frame_mask
│   │   └── pose
│   │       ├── test
│   │       └── train
│   └── UBnormal
│       ├── gt
│       ├── pose
│       │   ├── abnormal_train
│       │   ├── test
│       │   ├── train
│       │   └── validation
│       └── videos
├── models
│   └── STG_NF
└── utils

```

### Data Directory
Data folder, including extracted poses and GT, can be downloaded using the [link](https://drive.google.com/file/d/1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt/view?usp=sharing).

The data directory holds pose graphs and ground truth vectors for the datasets.
A path for the directory may be configured using the arguments:

    --vid_path_train
    --vid_path_test
    --pose_path_train
    --pose_path_train_abnormal
    --pose_path_test

### Custom Dataset

Extract poses using the recommended YOLO-pose pipeline (auto-downloads model on first use):

    python scripts/run_yolo_pose.py --input_dir <VIDEO_DIR> --output_dir <POSE_DIR>
    
Or using the legacy AlphaPose pipeline (requires manual weight downloads — see `doc_changes/`):

    python gen_data.py --alphapose_dir ./AlphaPose/ --dir <VIDEO_DIR> --outdir <POSE_DIR> [--video]

### Quickstart (Custom Dataset)

```bash
# 1. Extract poses from training videos
python scripts/run_yolo_pose.py \
    --input_dir <VIDEOS/train> \
    --output_dir <DATA/pose/train>

# 2. Extract poses from test videos
python scripts/run_yolo_pose.py \
    --input_dir <VIDEOS/test> \
    --output_dir <DATA/pose/test>

# 3. Train & evaluate STG-NF
python train_eval.py \
    --pose_path_train <DATA/pose/train> \
    --pose_path_test <DATA/pose/test>
```

## Training/Testing
Training and Evaluating is run using:
```
python train_eval.py --dataset [ShanghaiTech\UBnormal]
```

Evaluation of our pretrained model can be done using:

ShanghaiTech/ShanghaiTech-HR:
```
python train_eval.py --dataset [ShanghaiTech/ShanghaiTech-HR] --checkpoint checkpoints/ShanghaiTech_85_9.tar
```
Unsupervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar 
```
Supervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar
```

## Acknowledgments
Our code is based on code from:
- [Graph Embedded Pose Clustering for Anomaly Detection](https://github.com/amirmk89/gepc)
- [Glow](https://github.com/y0ast/Glow-PyTorch)

## License
This code is distributed under a [Creative Commons LICENSE](LICENSE).

Note that our code depends on other libraries and uses datasets that each have their own respective licenses that must also be followed.
