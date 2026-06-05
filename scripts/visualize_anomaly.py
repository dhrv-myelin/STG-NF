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


def _draw_text_with_bg(img, text, org, font_face, font_scale, text_color,
                       bg_color=(0, 0, 0), thickness=1, padding=4):
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness,
                cv2.LINE_AA)


def _open_writer(path, fps, width, height):
    """Try multiple fourcc codes until one works."""
    for fourcc_str in ["avc1", "X264", "mp4v", "MJPG"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"  VideoWriter: using {fourcc_str} codec")
            return writer
    raise RuntimeError(f"Could not open VideoWriter for {path}")


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
    ap.add_argument("--no_tint", action="store_true",
                    help="Disable color tint overlay on frames")
    ap.add_argument("--raw_scores", action="store_true",
                    help="Show actual score values instead of normalizing to [0,1]")
    ap.add_argument("--font_scale", type=float, default=1.2,
                    help="Font scale for the score text overlay (default: 1.2)")
    args = ap.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        suffix = "_raw" if args.raw_scores else "_anomaly"
        args.output = f"{stem}{suffix}.mp4"

    raw_scores = np.load(args.scores).astype(np.float64)
    finite = raw_scores[np.isfinite(raw_scores)]
    if len(finite) == 0:
        print("ERROR: all scores are NaN/inf — cannot visualize")
        return
    vmin, vmax = finite.min(), finite.max()

    if args.raw_scores:
        display_scores = raw_scores.copy()
    else:
        if vmax - vmin > 1e-8:
            display_scores = (raw_scores - vmin) / (vmax - vmin)
        else:
            display_scores = np.zeros_like(raw_scores)
        display_scores = np.clip(display_scores, 0.0, 1.0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scores = len(raw_scores)

    # Pad scores to match total_frames so we never have a size mismatch
    if n_scores < total_frames:
        pad_count = total_frames - n_scores
        raw_scores = np.concatenate([raw_scores, np.full(pad_count, raw_scores[-1])])
        display_scores = np.concatenate([display_scores, np.full(pad_count, display_scores[-1])])
        print(f"  Padded scores: {n_scores} -> {total_frames} (last score repeated)")
        n_scores = total_frames

    separator_h = 3
    out_height = height + separator_h + args.timeline_height

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = _open_writer(args.output, fps, width, out_height)

    timeline_bar = _build_timeline_bar(width, args.timeline_height, display_scores)
    separator = np.full((separator_h, width, 3), 32, dtype=np.uint8)

    print(f"Frames: {total_frames}  |  Scores: {n_scores}")
    if args.raw_scores:
        print(f"Raw score range: [{vmin:.6f}, {vmax:.6f}]")
    else:
        print(f"Normalized score range: [0.0, 1.0]  (from raw [{vmin:.6f}, {vmax:.6f}])")
    print(f"Output: {args.output}")

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = args.font_scale
    font_thickness = 2

    abs_max = max(abs(vmin), abs(vmax))
    if abs_max >= 100:
        score_fmt = "{:.1f}"
    elif abs_max >= 1:
        score_fmt = "{:.3f}"
    else:
        score_fmt = "{:.6f}"

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()

        if not args.no_tint and frame_idx < n_scores:
            s = display_scores[frame_idx]
            tint_color = _color_for_score(s)
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), tint_color, -1)
            vis = cv2.addWeighted(overlay, args.alpha, vis, 1 - args.alpha, 0)

        if frame_idx < n_scores:
            score_val = raw_scores[frame_idx]
            label = f"Score: {score_fmt.format(score_val)}"
            _draw_text_with_bg(vis, label, (15, 40), font_face, font_scale,
                               text_color=(255, 255, 255), bg_color=(0, 0, 0),
                               thickness=font_thickness, padding=6)

            frame_label = f"Frame {frame_idx}/{n_scores}"
            _draw_text_with_bg(vis, frame_label, (width - 300, 40), font_face, 0.7,
                               text_color=(200, 200, 200), bg_color=(0, 0, 0),
                               thickness=1, padding=4)

        bar_copy = timeline_bar.copy()
        if frame_idx < n_scores:
            cx = int(frame_idx / n_scores * width)
            cv2.line(bar_copy, (cx, 0), (cx, args.timeline_height - 1),
                     (255, 255, 255), 2)
            lbl = f"{score_fmt.format(raw_scores[frame_idx])}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = min(cx - tw // 2, width - tw - 4)
            ly = args.timeline_height - 4
            cv2.rectangle(bar_copy, (lx - 2, ly - th - 2),
                          (lx + tw + 2, ly + 2), (0, 0, 0), -1)
            cv2.putText(bar_copy, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        vis = np.vstack([vis, separator, bar_copy])
        assert vis.shape == (out_height, width, 3), \
            f"Frame size mismatch: {vis.shape} != ({out_height}, {width}, 3)"
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done: {frame_idx} frames written to {args.output}")


if __name__ == "__main__":
    main()
