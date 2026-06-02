import numpy as np
import torch
import torchvision.ops


def _soft_nms_cpu(dets, iou_thr, method=1, sigma=0.5, min_score=1e-3):
    """Pure-Python soft-NMS fallback (replaces Cython soft_nms_cpu)."""
    N = dets.shape[0]
    for i in range(N):
        max_score = dets[i, 4]
        max_pos = i
        pos = i + 1
        while pos < N:
            if max_score < dets[pos, 4]:
                max_score = dets[pos, 4]
                max_pos = pos
            pos += 1
        dets[i, :5], dets[max_pos, :5] = dets[max_pos, :5], dets[i, :5].copy()
        if i == N - 1:
            break
        pos = i + 1
        while pos < N:
            x1 = max(dets[i, 0], dets[pos, 0])
            y1 = max(dets[i, 1], dets[pos, 1])
            x2 = min(dets[i, 2], dets[pos, 2])
            y2 = min(dets[i, 3], dets[pos, 3])
            w = max(0.0, x2 - x1 + 1)
            h = max(0.0, y2 - y1 + 1)
            inter = w * h
            area_i = (dets[i, 2] - dets[i, 0] + 1) * (dets[i, 3] - dets[i, 1] + 1)
            area_j = (dets[pos, 2] - dets[pos, 0] + 1) * (dets[pos, 3] - dets[pos, 1] + 1)
            ovr = inter / (area_i + area_j - inter)
            if method == 1:
                weight = 1 - ovr if ovr > iou_thr else 1
            else:
                weight = np.exp(-(ovr * ovr) / sigma)
            dets[pos, 4] *= weight
            if dets[pos, 4] < min_score:
                dets[pos, 4] = dets[i, 4]
                dets[i, 4] = dets[pos, 4]
                pos += 1
                continue
            pos += 1
    keep = np.where(dets[:, 4] >= min_score)[0]
    return dets[keep, :], keep


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets.to('cpu')
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        inds = torchvision.ops.nms(dets_th[:, :4], dets_th[:, 4], iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = _soft_nms_cpu(
        dets_np, iou_thr, method=method_codes[method], sigma=sigma, min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
