import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from args import init_parser
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.train_utils import init_model_params
from utils.optim_init import init_optimizer
from models.training import Trainer


def segment_to_frame_scores(scores, metadata, seg_len, smooth_passes=3):
    metadata_np = np.array(metadata)
    clip_person_frames = defaultdict(lambda: defaultdict(dict))

    for i, (scene, clip, person, start) in enumerate(metadata_np):
        center = int(start) + seg_len // 2
        clip_person_frames[(scene, clip)][int(person)][center] = scores[i]

    clip_scores = {}
    for (scene, clip), persons in clip_person_frames.items():
        max_frame = max(f for frames in persons.values() for f in frames)
        per_frame = np.full(max_frame + 1, np.inf)

        for fs in persons.values():
            for f, s in fs.items():
                per_frame[f] = min(per_frame[f], s)

        finite = per_frame[np.isfinite(per_frame)]
        if len(finite) > 0:
            per_frame[per_frame == np.inf] = finite.max()
            per_frame[per_frame == -np.inf] = finite.min()

        for sigma in range(1, smooth_passes + 1):
            per_frame = gaussian_filter1d(per_frame, sigma=sigma)

        clip_key = f"{scene}_{clip}".replace(' ', '_')
        clip_scores[clip_key] = per_frame

    return clip_scores


def main():
    ap = argparse.ArgumentParser(description='Qualitative anomaly score inference for STG-NF')
    ap.add_argument('--checkpoint', required=True, help='Path to trained checkpoint')
    ap.add_argument('--dataset', default='CustomDataset', help='Dataset name')
    ap.add_argument('--layout', default='alphapose', choices=['openpose', 'alphapose', 'ntu-rgb+d'])
    ap.add_argument('--pose_path_train', default=None)
    ap.add_argument('--pose_path_test', required=True)
    ap.add_argument('--seg_len', type=int, default=24)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output_dir', default='results', help='Directory to save per-clip scores')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--smooth_passes', type=int, default=3,
                    help='Gaussian smoothing passes on frame scores (default: 3, was 7)')

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.device = args.device if torch.cuda.is_available() else 'cpu'
    args.model_hidden_dim = 0
    args.model_confidence = False
    args.headless = False
    args.norm_scale = 0
    args.prop_norm_scale = 1
    args.train_seg_conf_th = 0.0
    args.specific_clip = None
    args.global_pose_segs = True
    args.seg_stride = 1
    args.data_dir = 'data/'
    args.seed = 999
    args.pose_path_train_abnormal = None
    args.K = 8
    args.L = 1
    args.R = 3.0
    args.flow_permutation = 'permute'
    args.edge_importance = False
    args.temporal_kernel = None
    args.adj_strategy = 'uniform'
    args.max_hops = 8
    args.ckpt_dir = args.output_dir
    args.vid_path_train = None
    args.vid_path_test = None
    args.vid_path = {
        'train': getattr(args, 'vid_path_train', None) or '',
        'test': getattr(args, 'vid_path_test', None) or '',
    }
    args.pose_path = {
        'train': getattr(args, 'pose_path_train', None) or '',
        'test': getattr(args, 'pose_path_test', None) or '',
    }

    _, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    dataset = loader['test'].dataset

    model_args = init_model_params(args, {"test": dataset})
    model = STG_NF(**model_args)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found at '{args.checkpoint}'")
        import sys
        sys.exit(1)

    optimizer_f = init_optimizer('adamx', lr=5e-4)

    trainer = Trainer(args, model, None, loader['test'],
                      optimizer_f=optimizer_f)
    trainer.load_checkpoint(args.checkpoint)

    print("Running inference...")
    scores = trainer.test()
    print(f"Per-segment scores: {scores.shape}")

    clip_scores = segment_to_frame_scores(scores, dataset.metadata, args.seg_len,
                                           smooth_passes=args.smooth_passes)

    for clip_key, frame_scores in clip_scores.items():
        out_path = os.path.join(args.output_dir, f"{clip_key}_scores.npy")
        np.save(out_path, frame_scores)
        print(f"  {clip_key}: {len(frame_scores)} frames, "
              f"mean={frame_scores.mean():.4f}, std={frame_scores.std():.4f}")

    print(f"\nSaved per-frame scores to {args.output_dir}/")


if __name__ == '__main__':
    main()
