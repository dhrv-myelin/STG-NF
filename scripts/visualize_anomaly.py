import argparse
import os

import cv2
import numpy as np


def _color_for_score(score):
    score = np.clip(score, 0.0, 1.0)
    if score < 0.5:
        g = 1.0
        r = score * 2.0
    else:
        r = 1.0
        g = 2.0 - score * 2.0
    b = 0.0
    return int(b * 255), int(g * 255), int(r * 255)


def _build_timeline_bar(width, bar_height, scores):
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    for x in range(width):
        idx = int(x / width * len(scores))
        idx = min(idx, len(scores) - 1)
        color = _color_for_score(scores[idx])
        cv2.line(bar, (x, 0), (x, bar_height - 1), color, 1)
    return bar


def main():
    ap = argparse.ArgumentParser(description="Overlay anomaly scores on a video")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--scores", required=True, help="Per-frame scores .npy file")
    ap.add_argument("--output", default=None,
                    help="Output video path (default: <video_stem>_anomaly.mp4)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Tint overlay strength (0-1, default: 0.3)")
    ap.add_argument("--timeline_height", type=int, default=40,
                    help="Height of the score timeline bar in pixels")
    args = ap.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{stem}_anomaly.mp4"

    scores = np.load(args.scores).astype(np.float64)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        print("ERROR: all scores are NaN/inf — cannot visualize")
        return
    vmin, vmax = finite.min(), finite.max()
    if vmax - vmin > 1e-8:
        scores = (scores - vmin) / (vmax - vmin)
    else:
        scores = np.zeros_like(scores)
    scores = np.clip(scores, 0.0, 1.0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps,
                          (width, height + args.timeline_height + 5))

    timeline_bar = _build_timeline_bar(width, args.timeline_height, scores)

    print(f"Frames: {total_frames}  |  Scores: {len(scores)}")
    print(f"Score range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"Output: {args.output}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(scores):
            s = scores[frame_idx]
            tint_color = _color_for_score(s)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), tint_color, -1)
            frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

        vis = np.vstack([frame, np.full((5, width, 3), 32, dtype=np.uint8)])

        bar_copy = timeline_bar.copy()
        if frame_idx < len(scores):
            cx = int(frame_idx / len(scores) * width)
            cv2.line(bar_copy, (cx, 0), (cx, args.timeline_height - 1),
                     (255, 255, 255), 2)
            label = f"{scores[frame_idx]:.3f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = min(cx - tw // 2, width - tw - 4)
            ly = args.timeline_height - 4
            cv2.rectangle(bar_copy, (lx - 2, ly - th - 2),
                          (lx + tw + 2, ly + 2), (0, 0, 0), -1)
            cv2.putText(bar_copy, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        vis = np.vstack([vis, bar_copy])
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done: {frame_idx} frames written to {args.output}")


if __name__ == "__main__":
    main()
