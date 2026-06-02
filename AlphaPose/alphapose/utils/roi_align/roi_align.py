import torch.nn as nn
from torch.nn.modules.utils import _pair

from torchvision.ops import roi_align as tv_roi_align
roi_align = tv_roi_align


class RoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale=1,
                 sample_num=0,
                 use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        return roi_align(features, rois, _pair(self.out_size),
                         self.spatial_scale, self.sample_num)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str
